#!/usr/bin/env python
"""Одноразовая миграция: все document_id в omd_unified начинаются с /beelink.

Зачем
-----
Старый `_norm_doc_path()` отбрасывал первый сегмент пути, поэтому в базе
лежали `/Pictures/...`, `/Data/...` и т.п. без указания устройства. Полный
логический путь должен быть `/<устройство>/<путь-к-файлу>`, где сегмент
устройства — его hostname-префикс (напр. `/beelink`), а не внутренний linkId
(см. `local-client.js`: "hostname must be the user-visible device name").

Что делает
----------
* memory_card / qa — UUID-id не трогаем, чиним только metadata.document_id
* file_chunk          — id вида `<doc_id>:chunk:<N>` переписываем вместе с путем
* эмбеддинги переиспользуются (повторного эмбеддинга нет)

Безопасность
------------
* По умолчанию DRY RUN — ничего не пишется.
* Перед записью делается полный бэкап каталога unified_index.
* Писать в базу, пока живёт процесс onmychat, НЕЛЬЗЯ (PersistentClient
  рассчитан на один процесс) — скрипт это проверяет и отказывается работать.

Запуск
------
    ./venv/bin/python migrate_doc_ids_to_beelink.py            # dry run
    ./venv/bin/python migrate_doc_ids_to_beelink.py --apply    # с бэкапом
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chromadb.config import Settings  # noqa: E402
import chromadb  # noqa: E402
from config import BASE_INDEX_DIR  # noqa: E402

DB_DIR = os.path.join(BASE_INDEX_DIR, "unified_index")
COLLECTION = "omd_unified"
NAMESPACE = "beelink"
BATCH = 200


def service_pids() -> list:
    """PID-ы запущенного onmychat (его нельзя писать в обход)."""
    try:
        out = subprocess.run(
            ["pgrep", "-af", "uvicorn api:app"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return []
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith(str(os.getpid())):
            continue
        if "migrate_doc_ids_to_beelink" in line:
            continue
        m = re.match(r"^(\d+)\s", line)
        if m and "uvicorn api:app" in line:
            pids.append(int(m.group(1)))
    return pids


def set_namespace(ns: str) -> None:
    global NAMESPACE
    NAMESPACE = ns


def normalize_doc_id(doc: str) -> str:
    """Приводит любой исторический document_id к /<namespace>/<path>."""
    if not doc:
        return doc
    doc = str(doc).strip()

    # Полный URL -> берём путь, host отбрасываем.
    if doc.startswith("http://") or doc.startswith("https://"):
        m = re.match(r"^https?://[^/]+(/.*)?$", doc)
        path = (m.group(1) if m and m.group(1) else "/") or "/"
        doc = path
        # Например https://onmydisk.net/beelink/Documents/f.pdf
        # -> /beelink/Documents/f.pdf — уже готов, namespace не дублируем.

    if not doc.startswith("/"):
        doc = "/" + doc

    # Нормализация слэшей, без потери регистра кириллицы.
    doc = re.sub(r"/{2,}", "/", doc)

    head = doc.split("/")[1] if len(doc.split("/")) > 1 else ""
    if head == NAMESPACE:
        return doc
    return "/" + NAMESPACE + doc


def new_id_for(old_id: str, old_doc: str, new_doc: str) -> str:
    """Переписывает путь внутри id, но только для канонического вида
    `<document_id>:chunk:<N>` (id начинается с document_id).

    Legacy-записи `migrated_omd_*` / `migrated_search_*` не трогаем: там id —
    произвольная строка из старого скрипта миграции, где путь сидит в середине
    URL (`migrated_search_http://localhost:8080/tester/beelink/Documents/...`).
    Наивный replace дал бы `/tester/beelink/beelink/Documents/...`. У них
    правим только metadata.document_id.
    """
    if not old_doc or not new_doc or old_doc == new_doc:
        return old_id
    if old_id.startswith(old_doc):
        return new_doc + old_id[len(old_doc):]
    return old_id


def backup_dir(tag: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst = f"{DB_DIR}.bak-{tag}-{stamp}"
    shutil.copytree(DB_DIR, dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="реальная запись (иначе dry run)")
    ap.add_argument("--namespace", default=NAMESPACE)
    args = ap.parse_args()

    set_namespace(args.namespace)

    print("=" * 66)
    print("Миграция document_id -> /%s/..." % NAMESPACE)
    print("Режим: %s" % ("REAL WRITE" if args.apply else "DRY RUN (ничего не пишется)"))
    print("=" * 66)

    if args.apply:
        pids = service_pids()
        if pids:
            print("\nОТКАЗ: onmychat запущен (PID %s)." % ", ".join(map(str, pids)))
            print("PersistentClient рассчитан на один процесс — конкурентная запись")
            print("в sqlite может заблокировать или повредить базу.")
            print("Останови сервис, затем повтори:  systemctl stop onmychat")
            return 2

    client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

    total = col.count()
    print("\nКоллекция %s: %d записей" % (COLLECTION, total))

    if args.apply:
        bak = backup_dir(args.namespace)
        print("Бэкап: %s" % bak)

    # ── Собираем план: читаем id+metadatas (без эмбеддингов — они не нужны для оценки)
    data = col.get(include=["metadatas"])
    ids = data.get("ids") or []
    metas = data.get("metadatas") or []

    plan = []          # (old_id, new_id, old_doc, new_doc, type)
    seen_new_ids = set()
    for rid, meta in zip(ids, metas):
        meta = meta or {}
        old_doc = meta.get("document_id") or ""
        new_doc = normalize_doc_id(old_doc) if old_doc else old_doc
        nid = new_id_for(rid, old_doc, new_doc)
        if nid != rid or new_doc != old_doc:
            plan.append((rid, nid, old_doc, new_doc, meta.get("type")))

    # Проверка коллизий: новый id не должен совпадать с чужим существующим.
    existing = set(ids)
    collisions = [p for p in plan if p[1] in existing and p[1] != p[0]]

    changed_docs = len({(p[2], p[3]) for p in plan})
    print("\nК изменению: %d записей, %d уникальных document_id" % (len(plan), changed_docs))
    print("Коллизий: %d" % len(collisions))
    if collisions:
        print("  СТОП, есть коллизии:")
        for p in collisions[:10]:
            print("    %s -> %s" % (p[0][:60], p[1][:60]))
        return 3

    by_type = {}
    for p in plan:
        by_type[p[4]] = by_type.get(p[4], 0) + 1
    print("По типам: %s" % by_type)

    print("\nПримеры (старый id -> новый id):")
    for p in plan[:8]:
        print("  [%s]" % p[4])
        print("     doc: %s" % p[2])
        print("       -> %s" % p[3])
        print("     id : %s" % p[0][:70])
        print("       -> %s" % p[1][:70])

    already = sum(1 for m in metas if (m or {}).get("document_id", "").startswith("/%s/" % NAMESPACE))
    print("\nУже корректных document_id: %d" % already)

    if not args.apply:
        print("\nDry run завершён. Для записи: --apply (сервис должен быть остановлен)")
        return 0

    # ── Запись батчами: эмбеддинги переиспользуем
    print("\nЗапись...")
    done = 0
    for start in range(0, len(plan), BATCH):
        batch = plan[start:start + BATCH]
        want_old = [b[0] for b in batch]
        src = col.get(ids=want_old, include=["documents", "metadatas", "embeddings"])

        new_ids, docs, metas_out, embs = [], [], [], []
        for j, rid in enumerate(src.get("ids") or []):
            new_ids.append(new_id_for(rid, batch[j][2], batch[j][3]))
            docs.append((src.get("documents") or [None] * len(src["ids"]))[j])
            m = dict((src.get("metadatas") or [{}] * len(src["ids"]))[j] or {})
            old_doc = m.get("document_id") or ""
            if old_doc:
                m["document_id"] = normalize_doc_id(old_doc)
            metas_out.append(m or {})
            e = (src.get("embeddings") if src.get("embeddings") is not None else None)
            embs.append(e[j] if e is not None else None)

        col.upsert(ids=new_ids, documents=docs, metadatas=metas_out, embeddings=embs)
        # Удалять ТОЛЬКО переименованные id. У memory_card id — это UUID и он
        # не меняется: такой же delete снёс бы только что записанную запись.
        stale = [b[0] for b in batch if b[0] != b[1]]
        if stale:
            col.delete(ids=stale)
        done += len(batch)
        print("  %d/%d" % (done, len(plan)))

    print("\nГотово: %d записей. Проверка:" % done)
    after = col.get(include=["metadatas"])
    bad = [
        (m or {}).get("document_id")
        for m in (after.get("metadatas") or [])
        if (m or {}).get("document_id")
        and not (m or {})["document_id"].startswith("/%s/" % NAMESPACE)
    ]
    final = col.count()
    print("  записей: %d (было %d)" % (final, total))
    print("  document_id без /%s/: %d" % (NAMESPACE, len(bad)))
    for b in bad[:10]:
        print("    осталось: %s" % b)

    # Потеря записей — всегда баг в скрипте (например delete по неизменённому id).
    if final < total:
        print("\nОШИБКА: потеряно %d записей. Откатись из бэкапа." % (total - final))
        return 5
    return 0 if not bad else 4


if __name__ == "__main__":
    sys.exit(main())
