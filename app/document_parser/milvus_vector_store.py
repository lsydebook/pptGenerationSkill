"""Milvus vector store helpers — stores both vectors and node metadata."""

from typing import Any, Sequence

from .types import NodeKind, StoredNode


def _vector_to_list(vector: Sequence[float]) -> list[float]:
    if hasattr(vector, "tolist"):
        return [float(x) for x in vector.tolist()]
    return [float(x) for x in vector]


class MilvusVectorStore:
    """Manages Milvus collections for vector + metadata storage.

    Each node stores:
      - node_id (VARCHAR, primary key)
      - parent_id (VARCHAR, nullable)
      - kind (VARCHAR)
      - title (VARCHAR)
      - text (VARCHAR)
      - metadata (JSON)
      - child_ids (JSON)
      - created_at (INT64, Unix timestamp)
      - embedding (FLOAT_VECTOR)
    """

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

        self._milvus_alias = f"milvus_{id(self)}"
        self._connect(milvus_uri, milvus_token, milvus_db)

        self._collection_name = f"{table_prefix}_vec"
        self._para_full_collection_name = f"{table_prefix}_para_full_vec"
        self._image_collection_name = image_collection or f"{table_prefix}_images_vec"

        self.collection = self._ensure_collection(self._collection_name)
        self.para_full_collection = self._ensure_collection(
            self._para_full_collection_name, create_if_missing=create_para_full
        )
        self.image_collection = self._ensure_collection(
            self._image_collection_name, create_if_missing=create_image
        )

    def _connect(self, uri: str, token: str | None, db_name: str) -> None:
        from pymilvus import connections

        kwargs: dict[str, object] = {"alias": self._milvus_alias, "uri": uri}
        if token:
            kwargs["token"] = token
        if db_name:
            kwargs["db_name"] = db_name
        connections.connect(**kwargs)

    def ensure_para_full_collection(self) -> None:
        if self.para_full_collection is None:
            self.para_full_collection = self._ensure_collection(
                self._para_full_collection_name
            )

    def ensure_image_collection(self) -> None:
        if self.image_collection is None:
            self.image_collection = self._ensure_collection(self._image_collection_name)

    def _ensure_collection(self, name: str, *, create_if_missing: bool = True):
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

        if utility.has_collection(name, using=self._milvus_alias):
            collection = Collection(name, using=self._milvus_alias)
            collection.load()
            return collection

        if not create_if_missing:
            return None

        # Extended schema — stores all node metadata inline
        fields = [
            FieldSchema(
                name="node_id",
                dtype=DataType.VARCHAR,
                max_length=512,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name="parent_id",
                dtype=DataType.VARCHAR,
                max_length=512,
            ),
            FieldSchema(
                name="kind",
                dtype=DataType.VARCHAR,
                max_length=32,
            ),
            FieldSchema(
                name="title",
                dtype=DataType.VARCHAR,
                max_length=4096,
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
            ),
            FieldSchema(
                name="child_ids",
                dtype=DataType.JSON,
            ),
            FieldSchema(
                name="created_at",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._dimensions,
            ),
        ]
        schema = CollectionSchema(fields, description="document nodes with vectors and metadata")
        collection = Collection(name, schema, using=self._milvus_alias)

        index_params = self._build_index_params()
        collection.create_index("embedding", index_params)
        collection.load()
        return collection

    def _build_index_params(self) -> dict[str, object]:
        if self._index_type == "HNSW":
            return {
                "index_type": "HNSW",
                "metric_type": self._metric,
                "params": {"M": 16, "efConstruction": 200},
            }
        if self._index_type == "IVF_FLAT":
            return {
                "index_type": "IVF_FLAT",
                "metric_type": self._metric,
                "params": {"nlist": 1024},
            }
        return {
            "index_type": "FLAT",
            "metric_type": self._metric,
            "params": {},
        }

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
        """Combine multiple filter clauses into one Milvus expression."""
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

    def upsert_nodes(self, collection, nodes: Sequence[StoredNode]) -> None:
        """Upsert StoredNode records into a Milvus collection."""
        if collection is None or not nodes:
            return

        data = [
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

        if hasattr(collection, "upsert"):
            collection.upsert(data)
        else:
            ids = [d["node_id"] for d in data]
            expr = self._expr_for_ids(ids)
            if expr:
                collection.delete(expr)
            collection.insert(data)
        collection.flush()

    def upsert_para_full_vectors(self, nodes: Sequence[StoredNode]) -> None:
        """Upsert paragraph-level full embeddings into the para_full collection."""
        if self.para_full_collection is None or not nodes:
            return

        para_nodes = [
            node
            for node in nodes
            if node.kind == NodeKind.PARAGRAPH and "full_embedding" in node.metadata
        ]
        if not para_nodes:
            return

        if self.para_full_collection is None:
            self.ensure_para_full_collection()
        if self.para_full_collection is None:
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

        if hasattr(self.para_full_collection, "upsert"):
            self.para_full_collection.upsert(data)
        else:
            ids = [d["node_id"] for d in data]
            expr = self._expr_for_ids(ids)
            if expr:
                self.para_full_collection.delete(expr)
            self.para_full_collection.insert(data)
        self.para_full_collection.flush()

    def fetch_node(self, node_id: str) -> dict | None:
        """Fetch a single node's full record from the main collection."""
        if self.collection is None:
            return None

        expr = f'node_id == "{node_id.replace("\"", "\\\"")}"'
        results = self.collection.query(
            expr=expr,
            output_fields=["node_id", "parent_id", "kind", "title", "text", "metadata", "child_ids", "created_at", "embedding"],
        )
        if not results:
            return None
        return self._row_to_dict(results[0])

    def fetch_nodes(self, node_ids: Sequence[str]) -> dict[str, dict]:
        """Fetch multiple nodes by ID; returns {node_id: row_dict}."""
        if not node_ids:
            return {}
        expr = self._expr_for_ids(list(node_ids))
        if not expr:
            return {}

        results = self.collection.query(
            expr=expr,
            output_fields=["node_id", "parent_id", "kind", "title", "text", "metadata", "child_ids", "created_at", "embedding"],
        )
        return {r["node_id"]: self._row_to_dict(r) for r in results}

    @staticmethod
    def _row_to_dict(row: dict) -> dict:
        """Normalise a Milvus query result row to a flat dict."""
        return {
            "node_id": row.get("node_id", ""),
            "parent_id": row.get("parent_id") or None,
            "kind": row.get("kind", ""),
            "title": row.get("title", ""),
            "text": row.get("text", ""),
            "metadata": row.get("metadata", {}) or {},
            "child_ids": row.get("child_ids", []) or [],
            "created_at": row.get("created_at", 0) or 0,
        }

    def fetch_embedding(self, node_id: str) -> list[float]:
        if self.collection is None:
            return [0.0] * self._dimensions

        expr = f'node_id == "{node_id.replace("\"", "\\\"")}"'
        results = self.collection.query(expr=expr, output_fields=["embedding"])
        if not results:
            return [0.0] * self._dimensions
        return [float(x) for x in results[0]["embedding"]]

    def search(
        self,
        collection,
        query_vector: Sequence[float],
        k: int,
        kinds: set[NodeKind] | None,
        *,
        date_start: int | None = None,
        date_end: int | None = None,
    ) -> list[tuple[str, float]]:
        if collection is None:
            return []

        expr = self._build_expr(kinds, date_start=date_start, date_end=date_end)
        results = collection.search(
            data=[_vector_to_list(query_vector)],
            anns_field="embedding",
            param=self._build_search_params(),
            limit=k,
            expr=expr,
            output_fields=["node_id", "kind"],
        )
        hits = results[0] if results else []
        return [(hit.id, self._score_from_distance(hit.distance)) for hit in hits]

    def search_with_details(
        self,
        collection,
        query_vector: Sequence[float],
        k: int,
        kinds: set[NodeKind] | None,
        *,
        date_start: int | None = None,
        date_end: int | None = None,
    ) -> list[dict]:
        """Search and return full row dicts (including text/metadata)."""
        if collection is None:
            return []

        expr = self._build_expr(kinds, date_start=date_start, date_end=date_end)
        results = collection.search(
            data=[_vector_to_list(query_vector)],
            anns_field="embedding",
            param=self._build_search_params(),
            limit=k,
            expr=expr,
            output_fields=[
                "node_id", "parent_id", "kind", "title", "text",
                "metadata", "child_ids", "created_at", "embedding",
            ],
        )
        hits = results[0] if results else []
        out = []
        for hit in hits:
            row = self._row_to_dict(hit.entity.to_dict() if hasattr(hit.entity, "to_dict") else {})
            row["score"] = self._score_from_distance(hit.distance)
            out.append(row)
        return out


def _bytes_to_float_list(data: bytes) -> list[float]:
    import struct

    if not data:
        return []

    count = len(data) // 4
    return list(struct.unpack("<" + "f" * count, data))
