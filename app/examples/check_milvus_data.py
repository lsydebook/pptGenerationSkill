"""Inspect Milvus indexed nodes: text, metadata, and embedding vectors."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import app.config  # noqa: F401
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

KIND_ORDER = {"document": 0, "section": 1, "paragraph": 2, "sentence": 3}


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _rows_to_export(rows: list[dict[str, Any]], *, include_full_embedding: bool) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for row in rows:
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
        if include_full_embedding and row.get("embedding") is not None:
            item["embedding"] = [float(x) for x in row["embedding"]]
        exported.append(item)
    return exported


def _truncate(text: str, max_len: int) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _embedding_summary(vector: Any, preview: int = 5) -> dict[str, Any]:
    if vector is None:
        return {"dim": 0, "norm": 0.0, "preview": []}

    values = [float(x) for x in vector]
    norm = math.sqrt(sum(v * v for v in values))
    return {
        "dim": len(values),
        "norm": round(norm, 6),
        "preview": [round(v, 6) for v in values[:preview]],
        "tail": [round(v, 6) for v in values[-2:]] if len(values) > preview else [],
    }


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        kind = str(row.get("kind", ""))
        return (KIND_ORDER.get(kind, 99), str(row.get("node_id", "")))

    return sorted(rows, key=sort_key)


def inspect_collection(
    client: MilvusClient,
    collection_name: str,
    *,
    document_id: str | None,
    limit: int,
    text_max: int,
    show_full_embedding: bool,
    export_path: str | None = None,
) -> None:
    filter_expr = 'node_id != ""'
    if document_id:
        safe_id = document_id.replace('"', '\\"')
        filter_expr = f'node_id == "{safe_id}" or node_id like "{safe_id}:%"'

    rows = client.query(
        collection_name=collection_name,
        filter=filter_expr,
        output_fields=OUTPUT_FIELDS,
        limit=limit,
    )
    rows = _sort_rows(rows)

    stats = client.get_collection_stats(collection_name)
    print(f"\n{'=' * 72}")
    print(f"Collection: {collection_name}")
    print(f"Stats: {stats} | matched rows: {len(rows)}")
    if document_id:
        print(f"Filter document: {document_id!r}")
    print(f"{'=' * 72}")

    if not rows:
        print("(no rows)")
        return

    if export_path:
        payload = {
            "collection": collection_name,
            "document_id": document_id,
            "row_count": len(rows),
            "rows": _rows_to_export(rows, include_full_embedding=show_full_embedding),
        }
        with open(export_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\nExported {len(rows)} rows to {export_path}")

    for index, row in enumerate(rows, start=1):
        embedding = row.get("embedding")
        emb = _embedding_summary(embedding)
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
        if show_full_embedding and embedding is not None:
            print(f"  embedding(full): {[float(x) for x in embedding]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Milvus RAG node records")
    parser.add_argument(
        "--db",
        default=os.getenv("MILVUS_DB", "default"),
        help="Milvus database name (default: MILVUS_DB env)",
    )
    parser.add_argument(
        "--collection",
        default="rag_nodes_vec",
        help="Collection name to inspect",
    )
    parser.add_argument(
        "--document-id",
        default=None,
        help='Only show one document tree, e.g. "新建 文本文档" or "input"',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max rows to fetch",
    )
    parser.add_argument(
        "--text-max",
        type=int,
        default=200,
        help="Max characters of text to print per row",
    )
    parser.add_argument(
        "--full-embedding",
        action="store_true",
        help="Print full embedding vector (very verbose)",
    )
    parser.add_argument(
        "--list-documents",
        action="store_true",
        help="List top-level document node_ids only",
    )
    parser.add_argument(
        "--export",
        default=None,
        help="Export matched rows to a UTF-8 JSON file",
    )
    args = parser.parse_args()
    _configure_stdout()

    uri = os.getenv("MILVUS_URI")
    token = os.getenv("MILVUS_TOKEN")
    if not uri:
        raise SystemExit("MILVUS_URI is not set")

    client = MilvusClient(uri=uri, token=token, db_name=args.db)
    try:
        print(f"Database: {args.db}")
        print(f"Collections: {client.list_collections()}")

        if args.list_documents:
            rows = client.query(
                collection_name=args.collection,
                filter='kind == "document"',
                output_fields=["node_id", "title", "text", "metadata"],
                limit=args.limit,
            )
            print(f"\nDocuments in {args.collection}:")
            for row in rows:
                meta = row.get("metadata") or {}
                source = meta.get("source_filename", "-")
                print(
                    f"  - {row['node_id']!r} | title={row.get('title')!r} | "
                    f"source={source!r} | chars={len(str(row.get('text', '')))}"
                )
            return

        inspect_collection(
            client,
            args.collection,
            document_id=args.document_id,
            limit=args.limit,
            text_max=args.text_max,
            show_full_embedding=args.full_embedding,
            export_path=args.export,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
