"""Milvus vector store helpers — stores both vectors and node metadata."""

from __future__ import annotations

from typing import Any, Sequence

from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType, MilvusClient
from pymilvus.milvus_client import IndexParams

from src.config.indexing_config import MILVUS_BM25_ANALYZER, RAG_VEC_COLLECTION_SUFFIX
from src.parsing.document_types import NodeKind, StoredNode
from src.storage.bm25_scores import normalize_bm25_top_k

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

BM25_SPARSE_FIELD = "sparse"
BM25_FUNCTION_NAME = "text_bm25_emb"
BM25_SEARCH_PARAMS = {"params": {"level": 10}}


def _vector_to_list(vector: Sequence[float]) -> list[float]:
    if hasattr(vector, "tolist"):
        return [float(x) for x in vector.tolist()]
    return [float(x) for x in vector]


class MilvusVectorStore:
    """Manages Milvus collections for vector + metadata storage via MilvusClient."""

    def __init__(
        self,
        *,
        dimensions: int,
        table_prefix: str,
        milvus_uri: str,
        milvus_token: str | None,
        milvus_db: str,
        index_type: str,
        metric: str,
        search_nprobe: int,
        search_ef: int,
        image_collection: str | None,
        create_para_full: bool,
        create_image: bool,
        collection_suffix: str | None = None,
    ) -> None:
        self._dimensions = int(dimensions)
        self._metric = metric.upper()
        self._index_type = index_type.upper()
        self._search_nprobe = search_nprobe
        self._search_ef = search_ef
        self._collection_suffix = collection_suffix or RAG_VEC_COLLECTION_SUFFIX
        self._bm25_analyzer = MILVUS_BM25_ANALYZER

        client_kwargs: dict[str, Any] = {"uri": milvus_uri}
        if milvus_token:
            client_kwargs["token"] = milvus_token
        if milvus_db:
            client_kwargs["db_name"] = milvus_db
        self._client = MilvusClient(**client_kwargs)

        self._table_prefix = table_prefix
        suffix = f"_{self._collection_suffix}" if self._collection_suffix else ""
        self._collection_name = f"{table_prefix}_vec{suffix}"
        self._para_full_name = f"{table_prefix}_para_full_vec{suffix}"
        self._image_name = image_collection or f"{table_prefix}_images_vec{suffix}"

        self._ensure_collection(self._collection_name, with_bm25=True)
        self._para_full_collection_name = self._ensure_collection(
            self._para_full_name,
            create_if_missing=create_para_full,
            with_bm25=False,
        )
        self._image_collection_name = self._ensure_collection(
            self._image_name,
            create_if_missing=create_image,
            with_bm25=False,
        )

    @property
    def client(self) -> MilvusClient:
        return self._client

    @property
    def collection(self) -> str:
        return self._collection_name

    @property
    def para_full_collection(self) -> str | None:
        return self._para_full_collection_name

    @property
    def image_collection(self) -> str | None:
        return self._image_collection_name

    @property
    def bm25_enabled(self) -> bool:
        return True

    def close(self) -> None:
        self._client.close()

    def ensure_para_full_collection(self) -> None:
        if self._para_full_collection_name is None:
            self._para_full_collection_name = self._ensure_collection(
                self._para_full_name,
                with_bm25=False,
            )

    def ensure_image_collection(self) -> None:
        if self._image_collection_name is None:
            self._image_collection_name = self._ensure_collection(
                self._image_name,
                with_bm25=False,
            )

    def _common_fields(self, *, enable_text_analyzer: bool) -> list[FieldSchema]:
        text_params: dict[str, Any] = {"max_length": 65535}
        if enable_text_analyzer:
            text_params["enable_analyzer"] = True
            text_params["analyzer_params"] = {"type": self._bm25_analyzer}

        fields = [
            FieldSchema(
                name="node_id",
                dtype=DataType.VARCHAR,
                max_length=512,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="kind", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="text", dtype=DataType.VARCHAR, **text_params),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="child_ids", dtype=DataType.JSON),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._dimensions,
            ),
        ]
        return fields

    def _build_main_schema(self) -> CollectionSchema:
        fields = self._common_fields(enable_text_analyzer=True)
        fields.append(
            FieldSchema(name=BM25_SPARSE_FIELD, dtype=DataType.SPARSE_FLOAT_VECTOR)
        )
        bm25_function = Function(
            name=BM25_FUNCTION_NAME,
            input_field_names=["text"],
            output_field_names=[BM25_SPARSE_FIELD],
            function_type=FunctionType.BM25,
        )
        return CollectionSchema(
            fields=fields,
            functions=[bm25_function],
            description="document nodes with dense+sparse BM25",
        )

    def _build_dense_schema(self) -> CollectionSchema:
        return CollectionSchema(
            fields=self._common_fields(enable_text_analyzer=False),
            description="document nodes with dense vectors only",
        )

    def _add_dense_index(self, index_params: IndexParams) -> None:
        if self._index_type == "HNSW":
            index_params.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type=self._metric,
                params={"M": 16, "efConstruction": 200},
            )
        elif self._index_type == "IVF_FLAT":
            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type=self._metric,
                params={"nlist": 1024},
            )
        else:
            index_params.add_index(
                field_name="embedding",
                index_type="FLAT",
                metric_type=self._metric,
            )

    def _build_main_index_params(self) -> IndexParams:
        index_params = IndexParams()
        self._add_dense_index(index_params)
        index_params.add_index(
            field_name=BM25_SPARSE_FIELD,
            index_type="AUTOINDEX",
            metric_type="BM25",
        )
        return index_params

    def _build_dense_index_params(self) -> IndexParams:
        index_params = IndexParams()
        self._add_dense_index(index_params)
        return index_params

    def _ensure_collection(
        self,
        name: str,
        *,
        create_if_missing: bool = True,
        with_bm25: bool = False,
    ) -> str | None:
        if self._client.has_collection(name):
            self._client.load_collection(name)
            return name

        if not create_if_missing:
            return None

        schema = self._build_main_schema() if with_bm25 else self._build_dense_schema()
        index_params = (
            self._build_main_index_params()
            if with_bm25
            else self._build_dense_index_params()
        )
        self._client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
        )
        return name

    def _build_search_params(self) -> dict[str, object]:
        if self._index_type == "HNSW":
            return {"metric_type": self._metric, "params": {"ef": self._search_ef}}
        if self._index_type == "IVF_FLAT":
            return {
                "metric_type": self._metric,
                "params": {"nprobe": self._search_nprobe},
            }
        return {"metric_type": self._metric, "params": {}}

    def _expr_for_kinds(self, kinds: set[NodeKind] | None) -> str | None:
        if not kinds:
            return None
        values = [f'"{k.value}"' for k in kinds]
        return f"kind in [{', '.join(values)}]"

    def _build_expr(
        self,
        kinds: set[NodeKind] | None = None,
        *,
        date_start: int | None = None,
        date_end: int | None = None,
        ids: list[str] | None = None,
    ) -> str | None:
        clauses: list[str] = []

        kinds_expr = self._expr_for_kinds(kinds)
        if kinds_expr:
            clauses.append(kinds_expr)

        if date_start is not None and date_end is not None:
            clauses.append(f"created_at >= {date_start} and created_at <= {date_end}")
        elif date_start is not None:
            clauses.append(f"created_at >= {date_start}")
        elif date_end is not None:
            clauses.append(f"created_at <= {date_end}")

        ids_expr = self._expr_for_ids(ids or [])
        if ids_expr:
            clauses.append(ids_expr)

        if not clauses:
            return None
        return " and ".join(clauses)

    def _score_from_distance(self, distance: float) -> float:
        if self._metric == "COSINE":
            return 1.0 - float(distance)
        if self._metric in {"IP", "INNER_PRODUCT"}:
            return float(distance)
        return -float(distance)

    def _expr_for_ids(self, ids: list[str]) -> str | None:
        if not ids:
            return None
        quoted = [f'"{value.replace("\"", "\\\"")}"' for value in ids]
        return f"node_id in [{', '.join(quoted)}]"

    @staticmethod
    def _nodes_to_rows(nodes: Sequence[StoredNode]) -> list[dict[str, Any]]:
        return [
            {
                "node_id": node.node_id,
                "parent_id": node.parent_id or "",
                "kind": node.kind.value,
                "title": node.title,
                "text": node.text,
                "metadata": dict(node.metadata),
                "child_ids": list(node.child_ids),
                "created_at": node.created_at or 0,
                "embedding": _vector_to_list(node.embedding),
            }
            for node in nodes
        ]

    def upsert_nodes(self, collection_name: str | None, nodes: Sequence[StoredNode]) -> None:
        if not collection_name or not nodes:
            return
        self._client.upsert(
            collection_name=collection_name,
            data=self._nodes_to_rows(nodes),
        )

    def upsert_para_full_vectors(self, nodes: Sequence[StoredNode]) -> None:
        if self._para_full_collection_name is None or not nodes:
            return

        para_nodes = [
            node
            for node in nodes
            if node.kind == NodeKind.PARAGRAPH and "full_embedding" in node.metadata
        ]
        if not para_nodes:
            return

        data = []
        for node in para_nodes:
            full_emb_hex = node.metadata["full_embedding"]
            full_emb = bytes.fromhex(full_emb_hex)
            data.append(
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id or "",
                    "kind": node.kind.value,
                    "title": node.title,
                    "text": node.text,
                    "metadata": dict(node.metadata),
                    "child_ids": list(node.child_ids),
                    "created_at": node.created_at or 0,
                    "embedding": _vector_to_list(_bytes_to_float_list(full_emb)),
                }
            )

        self._client.upsert(
            collection_name=self._para_full_collection_name,
            data=data,
        )

    def fetch_node(self, node_id: str) -> dict | None:
        expr = f'node_id == "{node_id.replace("\"", "\\\"")}"'
        results = self._client.query(
            collection_name=self._collection_name,
            filter=expr,
            output_fields=OUTPUT_FIELDS,
            limit=1,
        )
        if not results:
            return None
        return self._row_to_dict(results[0])

    def fetch_nodes(self, node_ids: Sequence[str]) -> dict[str, dict]:
        if not node_ids:
            return {}
        expr = self._expr_for_ids(list(node_ids))
        if not expr:
            return {}

        results = self._client.query(
            collection_name=self._collection_name,
            filter=expr,
            output_fields=OUTPUT_FIELDS,
        )
        return {r["node_id"]: self._row_to_dict(r) for r in results}

    @staticmethod
    def _row_to_dict(row: dict) -> dict:
        return {
            "node_id": row.get("node_id", ""),
            "parent_id": row.get("parent_id") or None,
            "kind": row.get("kind", ""),
            "title": row.get("title", ""),
            "text": row.get("text", ""),
            "metadata": row.get("metadata", {}) or {},
            "child_ids": row.get("child_ids", []) or [],
            "created_at": row.get("created_at", 0) or 0,
            "embedding": row.get("embedding"),
        }

    def fetch_embedding(self, node_id: str) -> list[float]:
        expr = f'node_id == "{node_id.replace("\"", "\\\"")}"'
        results = self._client.query(
            collection_name=self._collection_name,
            filter=expr,
            output_fields=["embedding"],
            limit=1,
        )
        if not results:
            return [0.0] * self._dimensions
        return [float(x) for x in results[0]["embedding"]]

    @staticmethod
    def _hit_node_id(hit: dict[str, Any]) -> str:
        entity = hit.get("entity") or {}
        return str(hit.get("id") or hit.get("node_id") or entity.get("node_id") or "")

    @staticmethod
    def _hit_distance(hit: dict[str, Any]) -> float:
        return float(hit.get("distance", 0.0))

    def search(
        self,
        collection_name: str | None,
        query_vector: Sequence[float],
        k: int,
        kinds: set[NodeKind] | None,
        *,
        date_start: int | None = None,
        date_end: int | None = None,
    ) -> list[tuple[str, float]]:
        if not collection_name:
            return []

        filter_expr = self._build_expr(
            kinds,
            date_start=date_start,
            date_end=date_end,
        )
        results = self._client.search(
            collection_name=collection_name,
            data=[_vector_to_list(query_vector)],
            anns_field="embedding",
            search_params=self._build_search_params(),
            limit=k,
            filter=filter_expr or "",
            output_fields=["node_id", "kind"],
        )
        hits = results[0] if results else []
        return [
            (self._hit_node_id(hit), self._score_from_distance(self._hit_distance(hit)))
            for hit in hits
        ]

    def search_batch(
        self,
        collection_name: str | None,
        query_vectors: Sequence[Sequence[float]],
        k: int,
        kinds: set[NodeKind] | None,
        *,
        date_start: int | None = None,
        date_end: int | None = None,
    ) -> list[list[tuple[str, float]]]:
        if not collection_name or not query_vectors:
            return [[] for _ in query_vectors]

        filter_expr = self._build_expr(
            kinds,
            date_start=date_start,
            date_end=date_end,
        )
        results = self._client.search(
            collection_name=collection_name,
            data=[_vector_to_list(vector) for vector in query_vectors],
            anns_field="embedding",
            search_params=self._build_search_params(),
            limit=k,
            filter=filter_expr or "",
            output_fields=["node_id", "kind"],
        )
        batch_hits: list[list[tuple[str, float]]] = []
        for hits in results:
            batch_hits.append(
                [
                    (
                        self._hit_node_id(hit),
                        self._score_from_distance(self._hit_distance(hit)),
                    )
                    for hit in hits
                ]
            )
        return batch_hits

    def search_bm25_batch(
        self,
        query_texts: Sequence[str],
        k: int,
        kinds: set[NodeKind] | None,
        *,
        date_start: int | None = None,
        date_end: int | None = None,
    ) -> list[list[tuple[str, float]]]:
        if k <= 0 or not query_texts:
            return [[] for _ in query_texts]

        filter_expr = self._build_expr(
            kinds,
            date_start=date_start,
            date_end=date_end,
        )
        batch_data: list[str] = []
        batch_indices: list[int] = []
        for idx, query_text in enumerate(query_texts):
            query = (query_text or "").strip()
            if query:
                batch_data.append(query)
                batch_indices.append(idx)

        per_query: list[list[tuple[str, float]]] = [[] for _ in query_texts]
        if not batch_data:
            return per_query

        results = self._client.search(
            collection_name=self._collection_name,
            data=batch_data,
            anns_field=BM25_SPARSE_FIELD,
            search_params=BM25_SEARCH_PARAMS,
            limit=k,
            filter=filter_expr or "",
            output_fields=["node_id", "kind"],
        )
        for result_idx, hits in enumerate(results):
            orig_idx = batch_indices[result_idx]
            raw_hits = [
                (self._hit_node_id(hit), self._hit_distance(hit))
                for hit in hits
                if self._hit_node_id(hit)
            ]
            per_query[orig_idx] = normalize_bm25_top_k(raw_hits)
        return per_query

    def search_bm25(
        self,
        query_text: str,
        k: int,
        kinds: set[NodeKind] | None,
        *,
        date_start: int | None = None,
        date_end: int | None = None,
    ) -> list[tuple[str, float]]:
        """Zilliz BM25 full-text search on sparse field (chinese analyzer)."""
        query = (query_text or "").strip()
        if not query or k <= 0:
            return []

        filter_expr = self._build_expr(
            kinds,
            date_start=date_start,
            date_end=date_end,
        )
        results = self._client.search(
            collection_name=self._collection_name,
            data=[query],
            anns_field=BM25_SPARSE_FIELD,
            search_params=BM25_SEARCH_PARAMS,
            limit=k,
            filter=filter_expr or "",
            output_fields=["node_id", "kind"],
        )
        hits = results[0] if results else []
        raw_hits = [
            (self._hit_node_id(hit), self._hit_distance(hit))
            for hit in hits
            if self._hit_node_id(hit)
        ]
        return normalize_bm25_top_k(raw_hits)

    def count_searchable_nodes(
        self,
        kinds: set[NodeKind] | None = None,
    ) -> int:
        filter_expr = self._expr_for_kinds(kinds)
        results = self._client.query(
            collection_name=self._collection_name,
            filter=filter_expr or "",
            output_fields=["node_id"],
        )
        return len(results)

    def has_searchable_nodes(
        self,
        kinds: set[NodeKind] | None = None,
    ) -> bool:
        filter_expr = self._expr_for_kinds(kinds)
        results = self._client.query(
            collection_name=self._collection_name,
            filter=filter_expr or "",
            output_fields=["node_id"],
            limit=1,
        )
        return bool(results)

    def search_with_details(
        self,
        collection_name: str | None,
        query_vector: Sequence[float],
        k: int,
        kinds: set[NodeKind] | None,
        *,
        date_start: int | None = None,
        date_end: int | None = None,
    ) -> list[dict]:
        if not collection_name:
            return []

        filter_expr = self._build_expr(
            kinds,
            date_start=date_start,
            date_end=date_end,
        )
        results = self._client.search(
            collection_name=collection_name,
            data=[_vector_to_list(query_vector)],
            anns_field="embedding",
            search_params=self._build_search_params(),
            limit=k,
            filter=filter_expr or "",
            output_fields=OUTPUT_FIELDS,
        )
        hits = results[0] if results else []
        out: list[dict] = []
        for hit in hits:
            entity = hit.get("entity") or {}
            row = self._row_to_dict({**entity, **hit})
            row["score"] = self._score_from_distance(self._hit_distance(hit))
            out.append(row)
        return out


def _bytes_to_float_list(data: bytes) -> list[float]:
    import struct

    if not data:
        return []

    count = len(data) // 4
    return list(struct.unpack("<" + "f" * count, data))
