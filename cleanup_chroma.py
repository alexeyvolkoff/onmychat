#!/usr/bin/env python3
"""
Удаляет из ChromaDB все записи, чей owner ≠ NODE_OWNER,
за исключением записей с t_omd=1 (публичные знания владельца).

Использование:
    python3 cleanup_chroma.py                   # dry-run (покажет что будет удалено)
    python3 cleanup_chroma.py --apply           # реальная очистка
    python3 cleanup_chroma.py --owner=<user_id>  # задать owner явно
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from config import SETTINGS
import unified_memory


def main():
    parser = argparse.ArgumentParser(description="Cleanup guest data from ChromaDB")
    parser.add_argument("--apply", action="store_true", help="Actually delete (default: dry-run)")
    parser.add_argument("--owner", default=SETTINGS.get("NODE_OWNER", ""), help="Node owner user_id to keep")
    args = parser.parse_args()

    owners_to_keep = [args.owner] if args.owner else []
    coll = unified_memory.get_collection()
    total = coll.count()
    print(f"ChromaDB total records: {total}")

    if not args.owner:
        # Show all unique owners so the user can pick
        results = coll.get(include=["metadatas"])
        owners = {}
        for meta in results.get("metadatas", []):
            o = meta.get("owner", "(none)")
            has_omd = meta.get("t_omd", 0) == 1
            key = f"{ o}{' [omd]' if has_omd else ''}"
            owners[key] = owners.get(key, 0) + 1

        print("\nExisting owners:")
        for k, v in sorted(owners.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v} records")

        print("\nUsage: python3 cleanup_chroma.py --owner=<user_id> --apply")
        return

    # Dry-run: count what would be deleted
    results = coll.get(include=["metadatas"])
    would_delete = 0
    would_keep = 0
    for meta in results.get("metadatas", []):
        owner = meta.get("owner", "")
        has_omd = meta.get("t_omd", 0) == 1
        if owner not in owners_to_keep and not has_omd:
            would_delete += 1
        else:
            would_keep += 1

    print(f"\nOwner to keep: {args.owner}")
    print(f"Would delete: {would_delete} guest records")
    print(f"Would keep:   {would_keep} records")

    if not args.apply:
        print("\n[DRY-RUN] No changes made. Use --apply to execute.")
        return

    deleted = unified_memory.cleanup_guest_data(owners_to_keep)
    print(f"\nDone. Deleted {deleted} records.")


if __name__ == "__main__":
    main()
