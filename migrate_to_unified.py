#!/usr/bin/env python3
"""
migrate_to_unified.py — перенос данных из старых ChromaDB баз (omd, omd_search, qa_index)
в unifiedMemory (omd_unified) с тегом "omd".

Запуск:  python migrate_to_unified.py [--dry-run]
"""

import os
import sys
import logging
import chromadb
from chromadb.config import Settings

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_INDEX_DIR = "memory_index"

OLD_OMD_DIR    = os.path.join(BASE_INDEX_DIR, "chroma_db")
OLD_SEARCH_DIR = os.path.join(BASE_INDEX_DIR, "search_index")
OLD_QA_DIR     = os.path.join(BASE_INDEX_DIR, "chroma_qa")
NEW_UNIFIED_DIR = os.path.join(BASE_INDEX_DIR, "unified_index")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("migrate")

DRY_RUN = "--dry-run" in sys.argv


def get_collection(db_dir: str, name: str):
    client = chromadb.PersistentClient(
        path=db_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        col = client.get_collection(name=name, metadata={"hnsw:space": "cosine"})
    except Exception:
        col = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    return col


def map_old_meta(meta: dict, record_type: str, tags: str) -> dict:
    """Преобразует старую metadata в формат unified_memory (с t_ полями для тегов)."""
    new_meta = {
        "type":      record_type,
        "owner":     meta.get("owner", "alexey"),
        "relevance": meta.get("relevance", "contextual"),
        "timestamp": meta.get("timestamp", ""),
    }

    if not meta.get("document_id"):
        # Старая база omd_search хранила путь в itemPath
        if meta.get("itemPath"):
            new_meta["document_id"] = meta["itemPath"]
        elif meta.get("url"):
            new_meta["document_id"] = meta["url"]

    if new_meta.get("document_id") and not new_meta["document_id"].startswith("http"):
        new_meta["document_id"] = norm_document_path(new_meta["document_id"])

    if meta.get("document_id"):
        new_meta["document_id"] = meta["document_id"]

    if meta.get("chunk_id") is not None:
        new_meta["chunk_id"] = str(meta["chunk_id"])

    # timestamp — из last_modified, если не задан
    if not new_meta["timestamp"] and meta.get("last_modified"):
        new_meta["timestamp"] = meta["last_modified"]

    # title — из имени файла или из старого title
    title = meta.get("title", "")
    if not title and meta.get("document_id"):
        title = meta["document_id"].split("/")[-1]
    new_meta["title"] = title

    # Теги как отдельные t_ поля + строковое поле tags
    tag_list = [t.strip().lower() for t in tags.split(",") if t.strip()]
    for t in tag_list:
        new_meta[f"t_{t}"] = 1
    new_meta["tags"] = ",".join(tag_list)

    return new_meta


def norm_document_path(path: str) -> str:
    """Убирает префикс владельца: /alexey/beelink/doc.txt -> /beelink/doc.txt."""
    if path.startswith("/") and path.count("/") >= 2:
        return "/" + path.lstrip("/").split("/", 1)[1]
    return path


def migrate_collection(
    src_col,
    dst_col,
    record_type: str,
    tags: str,
    id_prefix: str,
    label: str,
):
    """Читает все записи из src_col, конвертирует и upsert в dst_col."""
    total = src_col.count()
    if total == 0:
        log.info(f"[{label}] Пустая коллекция, пропуск.")
        return 0

    log.info(f"[{label}] Найдено {total} записей, миграция...")

    # Читаем пачками по 1000 (ChromaDB get() без limit тоже ок, но для больших баз надёжнее)
    migrated = 0
    offset = 0
    batch_size = 1000

    while offset < total:
        batch = src_col.get(
            include=["documents", "metadatas", "embeddings"],
            limit=batch_size,
            offset=offset,
        )
        if not batch["ids"]:
            break

        ids        = batch["ids"]
        documents  = batch["documents"]
        metadatas  = batch["metadatas"]
        embeddings = batch["embeddings"]

        new_ids        = [f"{id_prefix}{i}" for i in ids]
        new_metadatas  = [map_old_meta(m, record_type, tags) for m in metadatas]
        new_documents  = documents

        if not DRY_RUN:
            dst_col.upsert(
                ids=new_ids,
                embeddings=embeddings,
                documents=new_documents,
                metadatas=new_metadatas,
            )

        migrated += len(ids)
        offset += batch_size
        log.info(f"[{label}] Перенесено {migrated}/{total}")

    return migrated


def main():
    log.info("=" * 60)
    log.info("Миграция старых ChromaDB → unified_memory (omd_unified)")
    log.info(f"Режим: {'DRY RUN (ничего не записывается)' if DRY_RUN else 'REAL WRITE'}")
    log.info("=" * 60)

    # Unified collection (destination)
    unified_client = chromadb.PersistentClient(
        path=NEW_UNIFIED_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    dst = unified_client.get_or_create_collection(
        name="omd_unified",
        metadata={"hnsw:space": "cosine"},
    )
    log.info(f"Текущий размер omd_unified: {dst.count()} записей")

    # ── 1. Старая база omd (memory cards + file chunks) ──
    try:
        src_omd = get_collection(OLD_OMD_DIR, "omd")
        count_omd = src_omd.count()
        log.info(f"\n[1/3] Старая база 'omd': {count_omd} записей")

        if count_omd > 0:
            # Определяем тип записи по relevance / chunk_id
            # Читаем все и группируем
            all_data = src_omd.get(
                include=["documents", "metadatas", "embeddings"],
            )
            total = len(all_data["ids"])
            log.info(f"[omd] Читаю {total} записей...")

            migrated = 0
            for i in range(total):
                old_id   = all_data["ids"][i]
                doc      = all_data["documents"][i]
                meta     = all_data["metadatas"][i]
                emb      = all_data["embeddings"][i]

                # Определяем тип
                relevance = meta.get("relevance", "contextual")
                has_chunk = meta.get("chunk_id") is not None

                if relevance == "document_chunk" or has_chunk:
                    rec_type = "file_chunk"
                else:
                    rec_type = "memory_card"

                new_meta = map_old_meta(meta, rec_type, "omd")
                new_id = f"migrated_omd_{old_id}"

                if not DRY_RUN:
                    dst.upsert(
                        ids=[new_id],
                        embeddings=[emb],
                        documents=[doc],
                        metadatas=[new_meta],
                    )
                migrated += 1

                if migrated % 500 == 0:
                    log.info(f"[omd] Перенесено {migrated}/{total}")

            log.info(f"[omd] Готово: {migrated} записей")

    except Exception as e:
        log.error(f"[omd] Ошибка: {e}")

    # ── 2. Старая база omd_search (file search index) ──
    try:
        src_search = get_collection(OLD_SEARCH_DIR, "omd_search")
        count_search = src_search.count()
        log.info(f"\n[2/3] Старая база 'omd_search': {count_search} записей")

        if count_search > 0:
            # Личные документы пользователя — без тегов (searchindex без тегов)
            migrated = migrate_collection(
                src_search, dst,
                record_type="file_chunk",
                tags="",
                id_prefix="migrated_search_",
                label="omd_search",
            )
            log.info(f"[omd_search] Готово: {migrated} записей")

    except Exception as e:
        log.error(f"[omd_search] Ошибка: {e}")

    # ── 3. Старая база qa_index (Q&A pairs) ──
    try:
        src_qa = get_collection(OLD_QA_DIR, "qa_index")
        count_qa = src_qa.count()
        log.info(f"\n[3/3] Старая база 'qa_index': {count_qa} записей")

        if count_qa > 0:
            all_qa = src_qa.get(
                include=["documents", "metadatas", "embeddings"],
            )
            total = len(all_qa["ids"])
            migrated = 0

            for i in range(total):
                old_id = all_qa["ids"][i]
                doc    = all_qa["documents"][i]   # question text
                meta   = all_qa["metadatas"][i]
                emb    = all_qa["embeddings"][i]

                new_meta = {
                    "type":    "qa",
                    "owner":   "system",
                    "answer":  meta.get("answer", ""),
                    "t_qa":    1,
                    "t_omd":   1,
                    "tags":    "qa,omd",
                }
                new_id = f"migrated_qa_{old_id}"

                if not DRY_RUN:
                    dst.upsert(
                        ids=[new_id],
                        embeddings=[emb],
                        documents=[doc],
                        metadatas=[new_meta],
                    )
                migrated += 1

            log.info(f"[qa_index] Готово: {migrated} записей")

    except Exception as e:
        log.error(f"[qa_index] Ошибка: {e}")

    # ── Итого ──
    final_count = dst.count() if not DRY_RUN else "?"
    log.info(f"\n{'=' * 60}")
    log.info(f"Итого в omd_unified: {final_count} записей")
    log.info("Миграция завершена.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
