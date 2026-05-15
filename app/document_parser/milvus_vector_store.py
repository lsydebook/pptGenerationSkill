"""Milvus vector store helpers."""

from typing import Sequence

from .types import NodeKind


def _vector_to_list(vector: Sequence[float]) -> list[float]:
    if hasattr(vector, "tolist"):
        return [float(x) for x in vector.tolist()]
    return [float(x) for x in vector]


class MilvusVectorStore:
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

        fields = [
            FieldSchema(
                name="node_id",
                dtype=DataType.VARCHAR,
                max_length=512,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name="kind",
                dtype=DataType.VARCHAR,
                max_length=32,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._dimensions,
            ),
        ]
        schema = CollectionSchema(fields, description="document vectors")
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

    def upsert_vectors(self, collection, data: list[dict]) -> None:
        if collection is None or not data:
            return

        if hasattr(collection, "upsert"):
            collection.upsert(data)
        else:
            ids = [item["node_id"] for item in data]
            expr = self._expr_for_ids(ids)
            if expr:
                collection.delete(expr)
            collection.insert(data)
        collection.flush()

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
    ) -> list[tuple[str, float]]:
        if collection is None:
            return []

        expr = self._expr_for_kinds(kinds)
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
