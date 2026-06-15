"""Milvus-only node store.

All node metadata (text, title, parent_id, child_ids, metadata JSON) is stored
inline in Milvus alongside the embedding vector.  The class retains the same
public interface so callers (RAGIndexer, rag_service) require no changes.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Literal, Sequence

import src.config.env_loader  # noqa: F401  # 加载 .env
from src.config.logging_config import get_logger
from src.parsing.document_types import NodeKind, RetrievalMatch, StoredNode

from .milvus_vector_store import MilvusVectorStore

logger = get_logger(__name__)

ParagraphSearchMode = Literal["averaged", "full", "both"]


class MilvusPostgresNodeStore:
    """Vector + metadata storage entirely in Milvus (no PostgreSQL dependency)."""

    def __init__(
        self,
        *,
        dimensions: int,
        table_prefix: str = "rag_nodes",
        paragraph_search_mode: ParagraphSearchMode = "averaged",
        milvus_uri: str | None = None,
        milvus_token: str | None = None,
        milvus_db: str | None = None,
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

        self._milvus_uri = milvus_uri or os.getenv("MILVUS_URI")
        self._milvus_token = milvus_token or os.getenv("MILVUS_TOKEN")
        self._milvus_db = milvus_db or os.getenv("MILVUS_DB") or "default"

        if not self._milvus_uri:
            raise ValueError("MILVUS_URI is required")

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
            create_para_full=True,
            create_image=False,
        )

    def __del__(self) -> None:
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
        if hasattr(self, "_vector_store"):
            self._vector_store.close()

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def _sync_upsert_nodes(self, nodes: Sequence[StoredNode]) -> None:
        if not nodes:
            return
        kind_counts: dict[str, int] = {}
        for node in nodes:
            key = node.kind.value
            kind_counts[key] = kind_counts.get(key, 0) + 1
        logger.info(
            "milvus upsert_nodes count=%s kinds=%s collection=%s",
            len(nodes),
            kind_counts,
            getattr(self._vector_store, "collection", self._table_prefix),
        )
        # All metadata + vectors go into the main collection
        self._vector_store.upsert_nodes(self._vector_store.collection, nodes)

        # Paragraph full-embedding collection (if any)
        self._vector_store.upsert_para_full_vectors(nodes)
        logger.info("milvus upsert_nodes done count=%s", len(nodes))

    async def upsert_nodes(self, nodes: Sequence[StoredNode]) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._sync_upsert_nodes, nodes)

    # ------------------------------------------------------------------
    # Read path — single node
    # ------------------------------------------------------------------

    def _row_to_node(self, row: dict) -> StoredNode:
        # If the row has an "embedding" key (from query), use it; otherwise
        # fetch it separately or default to a zero vector.
        embedding = row.get("embedding")
        if embedding is not None:
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            vector = [float(x) for x in embedding]
        else:
            vector = [0.0] * self._dimensions

        return StoredNode(
            node_id=row["node_id"],
            parent_id=row.get("parent_id"),
            kind=NodeKind(row["kind"]),
            title=row["title"],
            text=row["text"],
            metadata=row.get("metadata", {}),
            embedding=vector,
            child_ids=list(row.get("child_ids", [])),
            created_at=row.get("created_at", 0) or 0,
        )

    async def get_node(self, node_id: str) -> StoredNode:
        loop = asyncio.get_event_loop()
        row = await loop.run_in_executor(
            self._executor, self._vector_store.fetch_node, node_id
        )
        if row is None:
            raise KeyError(node_id)
        return self._row_to_node(row)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _sync_search_collection(
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
        return self._vector_store.search(
            collection, query_vector, k, kinds,
            date_start=date_start, date_end=date_end,
        )

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int = 5,
        kinds: set[NodeKind] | None = None,
        paragraph_search_mode: ParagraphSearchMode | None = None,
        date_range: tuple[int, int] | None = None,
    ) -> list[RetrievalMatch]:
        mode = paragraph_search_mode or self._paragraph_search_mode
        date_start, date_end = date_range if date_range else (None, None)

        loop = asyncio.get_event_loop()
        main_hits = await loop.run_in_executor(
            self._executor,
            partial(
                self._sync_search_collection,
                self._vector_store.collection,
                query_vector,
                k,
                kinds,
                date_start=date_start,
                date_end=date_end,
            ),
        )

        para_hits: list[tuple[str, float]] = []
        if mode in {"full", "both"} and self._vector_store.para_full_collection is not None:
            para_hits = await loop.run_in_executor(
                self._executor,
                partial(
                    self._sync_search_collection,
                    self._vector_store.para_full_collection,
                    query_vector,
                    k,
                    kinds,
                    date_start=date_start,
                    date_end=date_end,
                ),
            )

        all_hits = main_hits + para_hits
        if not all_hits:
            return []

        node_ids = [hit[0] for hit in all_hits]
        rows = await loop.run_in_executor(
            self._executor, self._vector_store.fetch_nodes, node_ids
        )

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
        rows = await loop.run_in_executor(
            self._executor, self._vector_store.fetch_nodes, node_ids
        )
        matches: list[RetrievalMatch] = []
        for node_id, score in hits:
            row = rows.get(node_id)
            if row is None:
                continue
            matches.append(RetrievalMatch(node=self._row_to_node(row), score=score))
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:k]

    # ------------------------------------------------------------------
    # Context (parent/child traversal)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def has_image_index(self) -> bool:
        return self._vector_store.image_collection is not None

    def has_full_paragraph_index(self) -> bool:
        return self._vector_store.para_full_collection is not None

    def set_paragraph_search_mode(self, mode: ParagraphSearchMode) -> None:
        self._paragraph_search_mode = mode

    # ------------------------------------------------------------------
    # BM25 hybrid search (Zilliz sparse + chinese analyzer)
    # ------------------------------------------------------------------

    def _searchable_kinds(self) -> set[NodeKind]:
        return {NodeKind.SENTENCE, NodeKind.PARAGRAPH}

    def has_bm25_index(self) -> bool:
        if not self._vector_store.bm25_enabled:
            return False
        return self._vector_store.has_searchable_nodes(kinds=self._searchable_kinds())

    def bm25_index_size(self) -> int:
        if not self._vector_store.bm25_enabled:
            return 0
        return self._vector_store.count_searchable_nodes(kinds=self._searchable_kinds())

    def _sync_search_bm25(
        self,
        query: str,
        k: int,
        kinds: set[NodeKind] | None,
    ) -> list[tuple[str, float]]:
        return self._vector_store.search_bm25(query, k, kinds)

    async def search_bm25(
        self,
        query: str,
        *,
        k: int = 5,
        kinds: set[NodeKind] | None = None,
    ) -> list[RetrievalMatch]:
        loop = asyncio.get_event_loop()
        hits = await loop.run_in_executor(
            self._executor,
            self._sync_search_bm25,
            query,
            k,
            kinds,
        )
        if not hits:
            return []

        node_ids = [hit[0] for hit in hits]
        rows = await loop.run_in_executor(
            self._executor, self._vector_store.fetch_nodes, node_ids
        )

        matches: list[RetrievalMatch] = []
        for node_id, score in hits:
            row = rows.get(node_id)
            if row is None:
                continue
            matches.append(RetrievalMatch(node=self._row_to_node(row), score=score))
        return matches
