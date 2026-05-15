"""Milvus + PostgreSQL backed node store."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Sequence

from .milvus_vector_store import MilvusVectorStore
from .postgres_metadata_store import PostgresMetadataStore
from .types import NodeKind, RetrievalMatch, StoredNode

ParagraphSearchMode = Literal["averaged", "full", "both"]


def _load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}

    env_vars: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_vars[key.strip()] = value.strip().strip("\"").strip("'")
    return env_vars


def _vector_to_list(vector: Sequence[float]) -> list[float]:
    if hasattr(vector, "tolist"):
        return [float(x) for x in vector.tolist()]
    return [float(x) for x in vector]


class MilvusPostgresNodeStore:
    """Vector storage in Milvus, metadata in PostgreSQL."""

    def __init__(
        self,
        *,
        dimensions: int,
        table_prefix: str = "rag_nodes",
        paragraph_search_mode: ParagraphSearchMode = "averaged",
        milvus_uri: str | None = None,
        milvus_token: str | None = None,
        milvus_db: str | None = None,
        pg_dsn: str | None = None,
        index_type: str = "HNSW",
        metric: str = "COSINE",
        search_nprobe: int = 16,
        search_ef: int = 64,
        image_collection: str | None = None,
    ) -> None:
        self._dimensions = int(dimensions)
        self._table_prefix = table_prefix
        self._paragraph_search_mode = paragraph_search_mode
        self._metric = metric.upper()
        self._index_type = index_type.upper()
        self._search_nprobe = search_nprobe
        self._search_ef = search_ef
        self._image_collection_name = image_collection

        self._dotenv = _load_dotenv()
        self._milvus_uri = milvus_uri or self._get_env("MILVUS_URI")
        self._milvus_token = milvus_token or self._get_env("MILVUS_TOKEN")
        self._milvus_db = milvus_db or self._get_env("MILVUS_DB") or "default"
        self._pg_dsn = (
            pg_dsn
            or self._get_env("PG_DSN")
            or self._get_env("POSTGRES_DSN")
            or self._get_env("DATABASE_URL")
        )

        if not self._milvus_uri:
            raise ValueError("MILVUS_URI is required")
        if not self._pg_dsn:
            raise ValueError("PG_DSN/POSTGRES_DSN/DATABASE_URL is required")

        self._executor = ThreadPoolExecutor(max_workers=1)

        self._vector_store = MilvusVectorStore(
            dimensions=self._dimensions,
            table_prefix=table_prefix,
            milvus_uri=self._milvus_uri,
            milvus_token=self._milvus_token,
            milvus_db=self._milvus_db,
            index_type=self._index_type,
            metric=self._metric,
            search_nprobe=self._search_nprobe,
            search_ef=self._search_ef,
            image_collection=self._image_collection_name,
            create_para_full=False,
            create_image=False,
        )
        self._metadata_store = PostgresMetadataStore(
            pg_dsn=self._pg_dsn,
            table_prefix=table_prefix,
        )

    def __del__(self) -> None:
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
        try:
            if hasattr(self, "_metadata_store"):
                self._metadata_store.close()
        except Exception:
            pass

    def _get_env(self, key: str) -> str | None:
        return os.getenv(key) or self._dotenv.get(key)

    def _sync_upsert_nodes(self, nodes: Sequence[StoredNode]) -> None:
        if not nodes:
            return
        self._metadata_store.upsert_nodes(nodes)

        main_data = [
            {
                "node_id": node.node_id,
                "kind": node.kind.value,
                "embedding": _vector_to_list(node.embedding),
            }
            for node in nodes
        ]
        self._vector_store.upsert_vectors(self._vector_store.collection, main_data)

        para_nodes = [
            node
            for node in nodes
            if node.kind == NodeKind.PARAGRAPH and "full_embedding" in node.metadata
        ]
        if para_nodes:
            if self._vector_store.para_full_collection is None:
                self._vector_store.ensure_para_full_collection()

            para_data = []
            for node in para_nodes:
                full_emb_hex = node.metadata["full_embedding"]
                full_emb = bytes.fromhex(full_emb_hex)
                full_emb_values = _vector_to_list(
                    _bytes_to_float_list(full_emb)
                )
                para_data.append(
                    {
                        "node_id": node.node_id,
                        "kind": node.kind.value,
                        "embedding": full_emb_values,
                    }
                )
            self._vector_store.upsert_vectors(
                self._vector_store.para_full_collection, para_data
            )

    async def upsert_nodes(self, nodes: Sequence[StoredNode]) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._sync_upsert_nodes, nodes)

    def _fetch_rows(self, node_ids: Sequence[str]) -> dict[str, dict]:
        return self._metadata_store.fetch_rows(node_ids)

    def _fetch_embedding(self, node_id: str) -> list[float]:
        return self._vector_store.fetch_embedding(node_id)

    def _row_to_node(self, row: dict, embedding: list[float] | None = None) -> StoredNode:
        vector = embedding if embedding is not None else [0.0] * self._dimensions
        return StoredNode(
            node_id=row["node_id"],
            parent_id=row.get("parent_id"),
            kind=NodeKind(row["kind"]),
            title=row["title"],
            text=row["text"],
            metadata=row.get("metadata", {}),
            embedding=vector,
            child_ids=list(row.get("child_ids", [])),
        )

    async def get_node(self, node_id: str) -> StoredNode:
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(self._executor, self._fetch_rows, [node_id])
        if node_id not in rows:
            raise KeyError(node_id)
        embedding = await loop.run_in_executor(
            self._executor, self._fetch_embedding, node_id
        )
        return self._row_to_node(rows[node_id], embedding)

    def _sync_search_collection(
        self,
        collection,
        query_vector: Sequence[float],
        k: int,
        kinds: set[NodeKind] | None,
    ) -> list[tuple[str, float]]:
        if collection is None:
            return []

        return self._vector_store.search(collection, query_vector, k, kinds)

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int = 5,
        kinds: set[NodeKind] | None = None,
        paragraph_search_mode: ParagraphSearchMode | None = None,
    ) -> list[RetrievalMatch]:
        mode = paragraph_search_mode or self._paragraph_search_mode

        loop = asyncio.get_event_loop()
        main_hits = await loop.run_in_executor(
            self._executor,
            self._sync_search_collection,
            self._vector_store.collection,
            query_vector,
            k,
            kinds,
        )

        para_hits: list[tuple[str, float]] = []
        if mode in {"full", "both"} and self._vector_store.para_full_collection is not None:
            para_hits = await loop.run_in_executor(
                self._executor,
                self._sync_search_collection,
                self._vector_store.para_full_collection,
                query_vector,
                k,
                kinds,
            )

        all_hits = main_hits + para_hits
        if not all_hits:
            return []

        node_ids = [hit[0] for hit in all_hits]
        rows = await loop.run_in_executor(self._executor, self._fetch_rows, node_ids)

        matches: list[RetrievalMatch] = []
        seen: set[str] = set()
        for node_id, score in all_hits:
            row = rows.get(node_id)
            if row is None:
                continue
            if mode == "both" and node_id in seen:
                continue
            seen.add(node_id)
            matches.append(RetrievalMatch(node=self._row_to_node(row), score=score))

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:k]

    async def search_images(
        self,
        query_vector: Sequence[float],
        *,
        k: int = 5,
    ) -> list[RetrievalMatch]:
        if self._vector_store.image_collection is None:
            return []

        loop = asyncio.get_event_loop()
        hits = await loop.run_in_executor(
            self._executor,
            self._sync_search_collection,
            self._vector_store.image_collection,
            query_vector,
            k,
            None,
        )
        if not hits:
            return []

        node_ids = [hit[0] for hit in hits]
        rows = await loop.run_in_executor(self._executor, self._fetch_rows, node_ids)
        matches: list[RetrievalMatch] = []
        for node_id, score in hits:
            row = rows.get(node_id)
            if row is None:
                continue
            matches.append(RetrievalMatch(node=self._row_to_node(row), score=score))
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:k]

    async def get_context(
        self,
        node_id: str,
        *,
        parent_depth: int = 1,
        child_depth: int = 0,
    ) -> list[StoredNode]:
        node = await self.get_node(node_id)
        context: list[StoredNode] = [node]

        if parent_depth > 0:
            parent = node
            for _ in range(parent_depth):
                if parent.parent_id is None:
                    break
                parent = await self.get_node(parent.parent_id)
                context.append(parent)

        if child_depth > 0:
            await self._collect_children(
                node, depth=child_depth, accumulator=context, seen=set()
            )

        unique: list[StoredNode] = []
        seen_ids: set[str] = set()
        for item in context:
            if item.node_id in seen_ids:
                continue
            seen_ids.add(item.node_id)
            unique.append(item)

        return unique

    async def _collect_children(
        self,
        node: StoredNode,
        *,
        depth: int,
        accumulator: list[StoredNode],
        seen: set[str],
    ) -> None:
        if depth <= 0:
            return

        for child_id in node.child_ids:
            if child_id in seen:
                continue

            seen.add(child_id)
            child = await self.get_node(child_id)
            accumulator.append(child)
            await self._collect_children(
                child,
                depth=depth - 1,
                accumulator=accumulator,
                seen=seen,
            )

    def has_image_index(self) -> bool:
        return self._vector_store.image_collection is not None

    def has_full_paragraph_index(self) -> bool:
        return self._vector_store.para_full_collection is not None

    def set_paragraph_search_mode(self, mode: ParagraphSearchMode) -> None:
        self._paragraph_search_mode = mode


def _bytes_to_float_list(data: bytes) -> list[float]:
    import struct

    if not data:
        return []

    count = len(data) // 4
    return list(struct.unpack("<" + "f" * count, data))
