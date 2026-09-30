import os
import io
import re
import uuid
import datetime
import logging
import warnings
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from PIL import Image

# Suppress insightface deprecated skimage transform warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

import unified_memory

logger = logging.getLogger("face_engine")

_app = None
_recognition_session = None

DEFAULT_DISTANCE_THRESHOLD = 0.42  # Cosine distance <= 0.42 means cosine similarity >= 0.58


def _get_app():
    """Ленивая инициализация FaceAnalysis (buffalo_s) только на CPU."""
    global _app
    if _app is None:
        try:
            from insightface.app import FaceAnalysis
            logger.info("[face_engine] Initializing FaceAnalysis (buffalo_s)...")
            app = FaceAnalysis(
                name="buffalo_s",
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            _app = app
            logger.info("[face_engine] FaceAnalysis initialized successfully.")
        except Exception as e:
            logger.error(f"[face_engine] Failed to initialize FaceAnalysis: {e}")
            raise e
    return _app


def _bytes_to_rgb_array(img_bytes: bytes) -> Tuple[np.ndarray, int, int]:
    """Декодирует байты в RGB numpy-массив и возвращает (array, width, height)."""
    with Image.open(io.BytesIO(img_bytes)) as pil_img:
        # Автоматическая ориентация по EXIF
        try:
            from PIL import ImageOps
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception:
            pass
        pil_rgb = pil_img.convert("RGB")
        w, h = pil_rgb.size
        # Ограничиваем максимальное разрешение для ускорения детекции
        max_dim = 1920
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_w, new_h = int(w * scale), int(h * scale)
            pil_rgb = pil_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = new_w, new_h
        return np.array(pil_rgb), w, h


def _crop_face_avatar_b64(rgb_array: np.ndarray, bbox: List[int], size: int = 256) -> str:
    """Вырезает квадратную область лица с запасом и возвращает data:image/jpeg;base64,..."""
    try:
        import base64
        h, w, _ = rgb_array.shape
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        box_side = int(max(bw, bh) * 1.3)
        half = box_side // 2

        cx1 = max(0, cx - half)
        cy1 = max(0, cy - half)
        cx2 = min(w, cx + half)
        cy2 = min(h, cy + half)

        crop_np = rgb_array[cy1:cy2, cx1:cx2]
        crop_pil = Image.fromarray(crop_np).resize((size, size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        crop_pil.save(buf, format="JPEG", quality=92, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logger.warning(f"[face_engine] Failed to generate face avatar: {e}")
        return ""


def detect_faces(
    img_bytes: bytes,
    owner: Optional[str] = None,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Обнаруживает все лица на изображении, извлекает эмбеддинги и ищет
    совпадения среди уже известных людей в коллекции omd_faces.
    """
    if not img_bytes:
        return []

    app = _get_app()
    rgb_img, img_w, img_h = _bytes_to_rgb_array(img_bytes)

    # Детекция и извлечение признаков (InsightFace ожидает BGR при передаче через cv2,
    # но FaceAnalysis внутри app.get делает нужные преобразования)
    # Переводим RGB -> BGR для InsightFace
    bgr_img = rgb_img[:, :, ::-1]
    detected = app.get(bgr_img)

    results = []
    faces_coll = unified_memory.get_faces_collection()

    for idx, face in enumerate(detected):
        bbox = [int(round(coord)) for coord in face.bbox]
        # Ограничиваем границами картинки
        bbox = [
            max(0, bbox[0]),
            max(0, bbox[1]),
            min(img_w, bbox[2]),
            min(img_h, bbox[3]),
        ]
        score = round(float(face.det_score), 3)

        # 512-мерный вектор лица
        emb = face.embedding
        emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
        emb_list = emb_norm.tolist()

        avatar_b64 = _crop_face_avatar_b64(rgb_img, bbox)

        match_name = None
        match_person_id = None
        similarity = 0.0

        # Поиск в ChromaDB omd_faces
        try:
            where_filter = {"owner": {"$eq": owner}} if owner else None
            q_res = faces_coll.query(
                query_embeddings=[emb_list],
                n_results=1,
                where=where_filter,
                include=["metadatas", "distances"],
            )
            if q_res and q_res.get("distances") and len(q_res["distances"][0]) > 0:
                dist = q_res["distances"][0][0]
                if dist <= distance_threshold:
                    meta = q_res["metadatas"][0][0]
                    match_name = meta.get("name")
                    match_person_id = meta.get("person_id")
                    similarity = round(max(0.0, 1.0 - dist), 3)
        except Exception as e:
            logger.warning(f"[face_engine] Chroma query error: {e}")

        results.append({
            "index": idx,
            "bbox": bbox,
            "score": score,
            "name": match_name,
            "person_id": match_person_id,
            "similarity": similarity,
            "avatar_b64": avatar_b64,
            "img_width": img_w,
            "img_height": img_h,
        })

    return results


def register_face(
    owner: str,
    name: str,
    img_bytes: bytes,
    bbox: Optional[List[int]] = None,
    person_id: Optional[str] = None,
    document_id: str = "",
) -> Dict[str, Any]:
    """
    Регистрирует образ лица для персоны в коллекции omd_faces.
    Если передан bbox, находит соответствующее лицо (или кропает и извлекает эмбеддинг).
    """
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Name cannot be empty")

    owner = owner or "owner"
    if not person_id:
        slug = re.sub(r"[^a-zA-Z0-9_\u0400-\u04FF]", "_", clean_name.lower()).strip("_")
        person_id = f"p_{slug}_{uuid.uuid4().hex[:6]}"

    app = _get_app()
    rgb_img, img_w, img_h = _bytes_to_rgb_array(img_bytes)
    bgr_img = rgb_img[:, :, ::-1]

    detected = app.get(bgr_img)
    target_face = None

    if bbox and len(bbox) == 4:
        # Находим лицо с максимальным пересечением (IoU) или содержащее центр bbox
        bx1, by1, bx2, by2 = bbox
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        best_dist = float("inf")

        for f in detected:
            fx1, fy1, fx2, fy2 = f.bbox
            fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
            dist = (fcx - bcx) ** 2 + (fcy - bcy) ** 2
            if dist < best_dist:
                best_dist = dist
                target_face = f
    elif detected:
        # Если bbox не передан, берем самое крупное лицо
        target_face = max(detected, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    if target_face is None:
        # Автодетектор не нашел лицо (например, сложный ракурс).
        # Если передан bbox, попробуем запустить детекцию на вырезанном кропе
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(round(c)) for c in bbox]
            crop_bgr = bgr_img[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
            if crop_bgr.size > 0:
                crop_detected = app.get(crop_bgr)
                if crop_detected:
                    target_face = crop_detected[0]

    if target_face is None:
        raise ValueError("No face detected in the selected area")

    # Нормализованный эмбеддинг
    emb = target_face.embedding
    emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
    emb_list = emb_norm.tolist()

    f_bbox = [int(round(c)) for c in target_face.bbox]
    eff_bbox = bbox or f_bbox
    avatar_b64 = _crop_face_avatar_b64(rgb_img, eff_bbox)

    face_id = f"face_{uuid.uuid4().hex[:12]}"
    created_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    metadata = {
        "name": clean_name,
        "person_id": person_id,
        "owner": owner,
        "document_id": document_id or "",
        "bbox": f"{eff_bbox[0]},{eff_bbox[1]},{eff_bbox[2]},{eff_bbox[3]}",
        "avatar_b64": avatar_b64,
        "created": created_iso,
    }

    faces_coll = unified_memory.get_faces_collection()
    faces_coll.add(
        ids=[face_id],
        embeddings=[emb_list],
        documents=[clean_name],
        metadatas=[metadata],
    )

    logger.info(f"[face_engine] Registered face {face_id} for person '{clean_name}' ({person_id})")

    return {
        "face_id": face_id,
        "person_id": person_id,
        "name": clean_name,
        "avatar_b64": avatar_b64,
        "bbox": eff_bbox,
    }


def get_known_people(owner: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Возвращает список всех известных персон с их аватарами и списком связанных документов.
    """
    faces_coll = unified_memory.get_faces_collection()
    where_filter = {"owner": {"$eq": owner}} if owner else None

    try:
        all_records = faces_coll.get(where=where_filter, include=["metadatas"])
    except Exception as e:
        logger.warning(f"[face_engine] get_known_people error: {e}")
        return []

    people_map: Dict[str, Dict[str, Any]] = {}

    if all_records and all_records.get("metadatas"):
        for meta in all_records["metadatas"]:
            pid = meta.get("person_id")
            if not pid:
                continue
            name = meta.get("name", "Unknown")
            avatar_b64 = meta.get("avatar_b64", "")
            doc_id = meta.get("document_id", "")

            if pid not in people_map:
                people_map[pid] = {
                    "person_id": pid,
                    "name": name,
                    "avatar_b64": avatar_b64,
                    "sample_count": 0,
                    "photos": set(),
                }

            people_map[pid]["sample_count"] += 1
            if not people_map[pid]["avatar_b64"] and avatar_b64:
                people_map[pid]["avatar_b64"] = avatar_b64
            if doc_id:
                people_map[pid]["photos"].add(doc_id)

    results = []
    for p in people_map.values():
        photos_list = sorted(list(p["photos"]))
        results.append({
            "person_id": p["person_id"],
            "name": p["name"],
            "avatar_b64": p["avatar_b64"],
            "sample_count": p["sample_count"],
            "photos_count": len(photos_list),
            "photos": photos_list,
        })

    # Сортируем: сначала те, у кого больше фото, затем по алфавиту
    results.sort(key=lambda x: (-x["photos_count"], x["name"].lower()))
    return results


def rename_person(person_id: str, new_name: str, owner: Optional[str] = None) -> int:
    """Переименовывает человека во всех записях коллекции omd_faces."""
    clean_name = new_name.strip()
    if not clean_name:
        raise ValueError("Name cannot be empty")

    faces_coll = unified_memory.get_faces_collection()
    where_filter = {"person_id": {"$eq": person_id}}
    if owner:
        where_filter = {"$and": [{"person_id": {"$eq": person_id}}, {"owner": {"$eq": owner}}]}

    records = faces_coll.get(where=where_filter, include=["metadatas", "embeddings"])
    if not records or not records.get("ids"):
        return 0

    ids = records["ids"]
    metadatas = records["metadatas"]
    embeddings = records["embeddings"]

    for m in metadatas:
        m["name"] = clean_name

    faces_coll.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=[clean_name] * len(ids),
        metadatas=metadatas,
    )
    logger.info(f"[face_engine] Renamed person {person_id} to '{clean_name}' in {len(ids)} records")
    return len(ids)


def delete_person(person_id: str, owner: Optional[str] = None) -> int:
    """Удаляет все образцы лица персоны из omd_faces."""
    faces_coll = unified_memory.get_faces_collection()
    where_filter = {"person_id": {"$eq": person_id}}
    if owner:
        where_filter = {"$and": [{"person_id": {"$eq": person_id}}, {"owner": {"$eq": owner}}]}

    records = faces_coll.get(where=where_filter)
    count = len(records.get("ids", []))
    if count > 0:
        faces_coll.delete(where=where_filter)
        logger.info(f"[face_engine] Deleted person {person_id} ({count} face records)")
    return count


def match_names_for_image(
    img_bytes: bytes,
    owner: Optional[str] = None,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> List[str]:
    """
    Быстрый поиск имен известных людей на фото для пайплайна индексации (recognize_image_readme).
    Возвращает отсортированный список уникальных имен, например ['Алексей', 'Наташа'].
    """
    try:
        detected = detect_faces(img_bytes, owner=owner, distance_threshold=distance_threshold)
        names = set()
        for f in detected:
            if f.get("name"):
                names.add(f["name"])
        return sorted(list(names))
    except Exception as e:
        logger.warning(f"[face_engine] match_names_for_image error: {e}")
        return []
