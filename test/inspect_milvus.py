"""查看 Milvus / Zilliz 中已入库的全部 RAG 节点数据。

用法:
  uv run python -m test.inspect_milvus
  uv run python -m test.inspect_milvus --collection rag_nodes_vec
  uv run python -m test.inspect_milvus --export milvus_dump.json
  uv run python -m test.inspect_milvus --full-embedding
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import app.config  # noqa: F401  # loads .env
from app.config import RAG_TABLE_PREFIX
from pymilvus import MilvusClient

OUTPUT_FIELDS = [
    "node_id",
    "parent_id",
    "kind",
    "title",
    "text",
    "metadata",
    "child_ids",
    "created_at",
    "embedding",
]

KIND_ORDER = {"document": 0, "section": 1, "paragraph": 2, "sentence": 3, "attachment": 4}


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _embedding_summary(vector: Any, preview: int = 5) -> dict[str, Any]:
    if vector is None:
        return {"dim": 0, "norm": 0.0, "preview": [], "tail": []}

    values = [float(x) for x in vector]
    norm = math.sqrt(sum(v * v for v in values))
    return {
        "dim": len(values),
        "norm": round(norm, 6),
        "preview": [round(v, 6) for v in values[:preview]],
        "tail": [round(v, 6) for v in values[-2:]] if len(values) > preview else [],
    }


def _truncate(text: str, max_len: int) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        kind = str(row.get("kind", ""))
        return (KIND_ORDER.get(kind, 99), str(row.get("node_id", "")))

    return sorted(rows, key=sort_key)


def _row_for_export(row: dict[str, Any], *, full_embedding: bool) -> dict[str, Any]:
    item = {
        "node_id": row.get("node_id"),
        "parent_id": row.get("parent_id"),
        "kind": row.get("kind"),
        "title": row.get("title"),
        "text": row.get("text"),
        "metadata": row.get("metadata") or {},
        "child_ids": row.get("child_ids") or [],
        "created_at": row.get("created_at", 0),
        "embedding_summary": _embedding_summary(row.get("embedding")),
    }
    if full_embedding and row.get("embedding") is not None:
        item["embedding"] = [float(x) for x in row["embedding"]]
    return item


def _query_all_rows(client: MilvusClient, collection_name: str, *, limit: int) -> list[dict[str, Any]]:
    return client.query(
        collection_name=collection_name,
        filter='node_id != ""',
        output_fields=OUTPUT_FIELDS,
        limit=limit,
    )


def _print_collection(
    client: MilvusClient,
    collection_name: str,
    *,
    limit: int,
    text_max: int,
    full_embedding: bool,
) -> list[dict[str, Any]]:
    if not client.has_collection(collection_name):
        print(f"\n[collection] {collection_name} — 不存在，跳过")
        return []

    stats = client.get_collection_stats(collection_name)
    rows = _sort_rows(_query_all_rows(client, collection_name, limit=limit))

    print(f"\n{'=' * 72}")
    print(f"Collection: {collection_name}")
    print(f"Stats: {stats} | rows fetched: {len(rows)}")
    print(f"{'=' * 72}")

    if not rows:
        print("(empty)")
        return rows

    for index, row in enumerate(rows, start=1):
        emb = _embedding_summary(row.get("embedding"))
        metadata = row.get("metadata") or {}
        child_ids = row.get("child_ids") or []

        print(f"\n[{index}] {row.get('node_id')}")
        print(f"  kind       : {row.get('kind')}")
        print(f"  title      : {row.get('title')}")
        print(f"  parent_id  : {row.get('parent_id') or '-'}")
        print(f"  created_at : {row.get('created_at', 0)}")
        print(f"  text       : {_truncate(str(row.get('text', '')), text_max)}")
        print(f"  child_ids  : {child_ids if child_ids else '-'}")
        print(f"  metadata   : {json.dumps(metadata, ensure_ascii=False)}")
        print(
            f"  embedding  : dim={emb['dim']}, norm={emb['norm']}, "
            f"head={emb['preview']}, tail={emb['tail']}"
        )
        if full_embedding and row.get("embedding") is not None:
            print(f"  embedding(full): {[float(x) for x in row['embedding']]}")

    return rows


def _default_collections(prefix: str) -> list[str]:
    return [
        f"{prefix}_vec",
        f"{prefix}_para_full_vec",
        f"{prefix}_images_vec",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect all data stored in Milvus")
    parser.add_argument(
        "--db",
        default=os.getenv("MILVUS_DB", "default"),
        help="Milvus database name",
    )
    parser.add_argument(
        "--collection",
        action="append",
        default=None,
        help="Collection to inspect (repeatable). Default: all RAG collections",
    )
    parser.add_argument(
        "--prefix",
        default=RAG_TABLE_PREFIX,
        help=f"RAG table prefix (default: {RAG_TABLE_PREFIX})",
    )
    parser.add_argument(
        "--all-collections",
        action="store_true",
        help="Inspect every collection in the database, not only RAG ones",
    )
    parser.add_argument("--limit", type=int, default=16384, help="Max rows per collection")
    parser.add_argument("--text-max", type=int, default=300, help="Max chars of text per row")
    parser.add_argument(
        "--full-embedding",
        action="store_true",
        help="Print/export full embedding vectors",
    )
    parser.add_argument(
        "--export",
        default=None,
        help="Export all fetched rows to a UTF-8 JSON file",
    )
    args = parser.parse_args()
    _configure_stdout()

    uri = os.getenv("MILVUS_URI")
    token = os.getenv("MILVUS_TOKEN")
    if not uri:
        raise SystemExit("MILVUS_URI is not set")

    client = MilvusClient(uri=uri, token=token, db_name=args.db)
    try:
        all_in_db = client.list_collections()
        print(f"Database: {args.db}")
        print(f"All collections: {all_in_db}")

        if args.all_collections:
            target_collections = all_in_db
        elif args.collection:
            target_collections = args.collection
        else:
            target_collections = [
                name for name in _default_collections(args.prefix) if name in all_in_db
            ]
            if not target_collections:
                target_collections = [n for n in all_in_db if args.prefix in n]
            if not target_collections and all_in_db:
                target_collections = all_in_db

        export_payload: dict[str, Any] = {
            "database": args.db,
            "collections": {},
        }

        total_rows = 0
        for collection_name in target_collections:
            rows = _print_collection(
                client,
                collection_name,
                limit=args.limit,
                text_max=args.text_max,
                full_embedding=args.full_embedding,
            )
            total_rows += len(rows)
            export_payload["collections"][collection_name] = [
                _row_for_export(row, full_embedding=args.full_embedding) for row in rows
            ]

        print(f"\n{'=' * 72}")
        print(f"Done. collections={len(target_collections)}, total rows={total_rows}")

        if args.export:
            export_payload["total_rows"] = total_rows
            with open(args.export, "w", encoding="utf-8") as handle:
                json.dump(export_payload, handle, ensure_ascii=False, indent=2)
            print(f"Exported to {args.export}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
