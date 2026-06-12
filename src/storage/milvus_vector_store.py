"""Milvus vector store helpers — stores both vectors and node metadata."""

from __future__ import annotations

from typing import Any, Sequence

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from pymilvus.milvus_client import IndexParams

from src.parsing.document_types import NodeKind, StoredNode

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
    ) -> None:
        self._dimensions = int(dimensions)
        self._metric = metric.upper()
        self._index_type = index_type.upper()
        self._search_nprobe = search_nprobe
        self._search_ef = search_ef

        client_kwargs: dict[str, Any] = {"uri": milvus_uri}
        if milvus_token:
            client_kwargs["token"] = milvus_token
        if milvus_db:
            client_kwargs["db_name"] = milvus_db
        self._client = MilvusClient(**client_kwargs)

        self._table_prefix = table_prefix
        self._collection_name = f"{table_prefix}_vec"
        self._para_full_name = f"{table_prefix}_para_full_vec"
        self._image_name = image_collection or f"{table_prefix}_images_vec"

        self._ensure_collection(self._collection_name)
        self._para_full_collection_name = self._ensure_collection(
            self._para_full_name,
            create_if_missing=create_para_full,
        )
        self._image_collection_name = self._ensure_collection(
            self._image_name,
            create_if_missing=create_image,
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

    def close(self) -> None:
        self._client.close()

    def ensure_para_full_collection(self) -> None:
        if self._para_full_collection_name is None:
            self._para_full_collection_name = self._ensure_collection(self._para_full_name)

    def ensure_image_collection(self) -> None:
        if self._image_collection_name is None:
            self._image_collection_name = self._ensure_collection(self._image_name)

    def _build_schema(self) -> CollectionSchema:
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
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="child_ids", dtype=DataType.JSON),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._dimensions,
            ),
        ]
        return CollectionSchema(
            fields,
            description="document nodes with vectors and metadata",
        )

    def _build_index_params(self) -> IndexParams:
        index_params = IndexParams()
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
        return index_params

    def _ensure_collection(self, name: str, *, create_if_missing: bool = True) -> str | None:
        if self._client.has_collection(name):
            self._client.load_collection(name)
            return name

        if not create_if_missing:
            return None

        self._client.create_collection(
            collection_name=name,
            schema=self._build_schema(),
            index_params=self._build_index_params(),
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

    def fetch_all_searchable_nodes(
        self,
        *,
        kinds: set[NodeKind] | None = None,
        batch_size: int = 500,
    ) -> list[dict]:
        """Fetch node_id/kind/text for BM25 indexing."""
        filter_expr = self._expr_for_kinds(kinds)
        rows: list[dict] = []
        offset = 0
        while True:
            results = self._client.query(
                collection_name=self._collection_name,
                filter=filter_expr or "",
                output_fields=["node_id", "kind", "text"],
                limit=batch_size,
                offset=offset,
            )
            if not results:
                break
            rows.extend(self._row_to_dict(r) for r in results)
            if len(results) < batch_size:
                break
            offset += batch_size
        return rows

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
