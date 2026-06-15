"""Probe Zilliz/Milvus: BM25 + SPARSE_FLOAT_VECTOR + Chinese analyzer."""

from __future__ import annotations

import sys
import time
import traceback

import src.config.env_loader  # noqa: F401
from src.config.indexing_config import MILVUS_DB, MILVUS_TOKEN, MILVUS_URI
from pymilvus import DataType, Function, FunctionType, MilvusClient

SAMPLE_ROWS = [
    {"text": "information retrieval focuses on finding relevant documents."},
    {"text": "青蒿素是抗疟疾药物，屠呦呦因发现青蒿素获得诺贝尔奖。"},
    {"text": "大语言模型正在改变软件开发方式。"},
    {"text": "KohakuRAG采用Tree Retrieval进行层次化文档检索。"},
]

QUERY_CASES = [
    ("en", "information retrieval documents"),
    ("zh", "青蒿素"),
    ("zh", "大语言模型"),
    ("zh", "屠呦呦"),
    ("mixed", "KohakuRAG"),
    ("en", "Tree Retrieval"),
]


def _extract_hits(search_result) -> list[dict]:
    """pymilvus 3.x SearchResult -> list of hit dicts for first query."""
    if not search_result:
        return []
    try:
        group = search_result[0]
    except (TypeError, KeyError, IndexError):
        return []
    return list(group) if group is not None else []


def _preview_hit(hit: dict) -> str:
    entity = hit.get("entity") or {}
    text = entity.get("text", "")
    score = float(hit.get("distance", hit.get("score", 0.0)))
    return f"score={score:.4f} text={text[:100]}"


def main() -> int:
    collection_name = f"bm25_chinese_probe_{int(time.time())}"
    print("=== Zilliz BM25 + Chinese Analyzer probe ===")
    print(f"uri={MILVUS_URI}")
    print(f"db={MILVUS_DB}")
    print(f"collection={collection_name}")

    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN, db_name=MILVUS_DB)

    try:
        print("\n[1/5] create schema (chinese analyzer + BM25 function) ...")
        schema = client.create_schema()
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=4096,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        schema.add_field(
            field_name="sparse",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_function(
            Function(
                name="text_bm25_emb",
                input_field_names=["text"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="sparse",
            index_type="AUTOINDEX",
            metric_type="BM25",
        )

        print("[2/5] create_collection ...")
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

        print("[3/5] insert sample rows ...")
        client.insert(collection_name, SAMPLE_ROWS)
        client.flush(collection_name)

        print("[4/5] BM25 search cases ...")
        search_params = {"params": {"level": 10}}
        failed: list[str] = []

        for tag, query in QUERY_CASES:
            result = client.search(
                collection_name=collection_name,
                data=[query],
                anns_field="sparse",
                output_fields=["text"],
                limit=2,
                search_params=search_params,
            )
            hits = _extract_hits(result)
            print(f"\n  [{tag}] query={query!r} hits={len(hits)}")
            for hit in hits:
                print(f"    - {_preview_hit(hit)}")
            if not hits:
                failed.append(query)

        print("\n[5/5] cleanup drop_collection ...")
        client.drop_collection(collection_name)

        if failed:
            print("\n=== RESULT: FAILED (empty hits) ===")
            print("queries with no hits:", failed)
            return 1

        print("\n=== RESULT: PASSED ===")
        print("BM25 + SPARSE_FLOAT_VECTOR + chinese analyzer work on this cluster.")
        return 0

    except Exception as exc:
        print("\n=== RESULT: FAILED ===")
        print(f"error_type={type(exc).__name__}")
        print(f"error={exc}")
        traceback.print_exc()
        try:
            if client.has_collection(collection_name):
                client.drop_collection(collection_name)
        except Exception as cleanup_exc:  # noqa: BLE001
            print(f"cleanup failed: {cleanup_exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
