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
from src.config.indexing_config import MILVUS_IO_WORKERS
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

        self._executor = ThreadPoolExecutor(max_workers=max(1, MILVUS_IO_WORKERS))

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

    async def get_nodes(self, node_ids: Sequence[str]) -> dict[str, StoredNode]:
        ids = [node_id for node_id in node_ids if node_id]
        if not ids:
            return {}
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(
            self._executor, self._vector_store.fetch_nodes, ids
        )
        return {
            node_id: self._row_to_node(row)
            for node_id, row in rows.items()
        }

    async def get_node(self, node_id: str) -> StoredNode:
        nodes = await self.get_nodes([node_id])
        node = nodes.get(node_id)
        if node is None:
            raise KeyError(node_id)
        return node

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

    def _hits_to_matches(
        self,
        hits: Sequence[tuple[str, float]],
        rows: dict[str, dict],
        *,
        paragraph_mode: ParagraphSearchMode | None = None,
        limit: int | None = None,
    ) -> list[RetrievalMatch]:
        mode = paragraph_mode or self._paragraph_search_mode
        matches: list[RetrievalMatch] = []
        seen: set[str] = set()
        for node_id, score in hits:
            row = rows.get(node_id)
            if row is None:
                continue
            if mode == "both" and node_id in seen:
                continue
            seen.add(node_id)
            matches.append(RetrievalMatch(node=self._row_to_node(row), score=score))
        matches.sort(key=lambda item: item.score, reverse=True)
        if limit is not None:
            return matches[:limit]
        return matches

    def _sync_search_dense_batch(
        self,
        query_vectors: Sequence[Sequence[float]],
        k: int,
        kinds: set[NodeKind] | None,
        *,
        date_start: int | None = None,
        date_end: int | None = None,
    ) -> list[list[RetrievalMatch]]:
        if not query_vectors:
            return []

        mode = self._paragraph_search_mode
        main_batches = self._vector_store.search_batch(
            self._vector_store.collection,
            query_vectors,
            k,
            kinds,
            date_start=date_start,
            date_end=date_end,
        )
        if mode in {"full", "both"} and self._vector_store.para_full_collection is not None:
            para_batches = self._vector_store.search_batch(
                self._vector_store.para_full_collection,
                query_vectors,
                k,
                kinds,
                date_start=date_start,
                date_end=date_end,
            )
            main_batches = [
                main_hits + para_hits
                for main_hits, para_hits in zip(main_batches, para_batches, strict=True)
            ]

        all_node_ids = {
            node_id
            for batch_hits in main_batches
            for node_id, _ in batch_hits
        }
        rows = self._vector_store.fetch_nodes(list(all_node_ids))
        return [
            self._hits_to_matches(batch_hits, rows, limit=k)
            for batch_hits in main_batches
        ]

    async def search_dense_batch(
        self,
        query_vectors: Sequence[Sequence[float]],
        *,
        k: int = 5,
        kinds: set[NodeKind] | None = None,
        date_range: tuple[int, int] | None = None,
    ) -> list[list[RetrievalMatch]]:
        date_start, date_end = date_range if date_range else (None, None)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(
                self._sync_search_dense_batch,
                query_vectors,
                k,
                kinds,
                date_start=date_start,
                date_end=date_end,
            ),
        )

    def _sync_search_bm25_batch(
        self,
        query_texts: Sequence[str],
        k: int,
        kinds: set[NodeKind] | None,
    ) -> list[list[RetrievalMatch]]:
        if k <= 0 or not query_texts:
            return [[] for _ in query_texts]

        batch_hits = self._vector_store.search_bm25_batch(query_texts, k, kinds)
        all_node_ids = {
            node_id
            for hits in batch_hits
            for node_id, _ in hits
        }
        if not all_node_ids:
            return [[] for _ in query_texts]

        rows = self._vector_store.fetch_nodes(list(all_node_ids))
        return [self._hits_to_matches(hits, rows) for hits in batch_hits]

    async def search_bm25_batch(
        self,
        query_texts: Sequence[str],
        *,
        k: int = 5,
        kinds: set[NodeKind] | None = None,
    ) -> list[list[RetrievalMatch]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(self._sync_search_bm25_batch, query_texts, k, kinds),
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
        contexts = await self.get_context_batch([(node_id, parent_depth, child_depth)])
        return contexts[0] if contexts else []

    def _sync_get_context_batch(
        self,
        specs: Sequence[tuple[str, int, int]],
    ) -> list[list[StoredNode]]:
        if not specs:
            return []

        rows: dict[str, dict] = {}
        root_ids = [node_id for node_id, _, _ in specs if node_id]
        rows.update(self._vector_store.fetch_nodes(root_ids))

        parent_ids: set[str] = set()
        for node_id, parent_depth, _ in specs:
            if parent_depth <= 0:
                continue
            row = rows.get(node_id)
            if row and row.get("parent_id"):
                parent_ids.add(str(row["parent_id"]))
        if parent_ids:
            rows.update(self._vector_store.fetch_nodes(list(parent_ids)))

        max_child_depth = max((child_depth for _, _, child_depth in specs), default=0)
        if max_child_depth > 0:
            frontier = {
                child_id
                for node_id, _, child_depth in specs
                if child_depth > 0
                for child_id in (rows.get(node_id) or {}).get("child_ids", [])
                if child_id
            }
            for _ in range(max_child_depth):
                missing = [node_id for node_id in frontier if node_id not in rows]
                if not missing:
                    break
                rows.update(self._vector_store.fetch_nodes(missing))
                next_frontier: set[str] = set()
                for node_id in missing:
                    row = rows.get(node_id)
                    if not row:
                        continue
                    for child_id in row.get("child_ids", []) or []:
                        if child_id and child_id not in rows:
                            next_frontier.add(child_id)
                frontier = next_frontier

        results: list[list[StoredNode]] = []
        for node_id, parent_depth, child_depth in specs:
            context: list[StoredNode] = []
            row = rows.get(node_id)
            if row is not None:
                context.append(self._row_to_node(row))

            if parent_depth > 0 and row is not None:
                parent_id = row.get("parent_id")
                parent_row = rows.get(parent_id) if parent_id else None
                if parent_row is not None:
                    context.append(self._row_to_node(parent_row))

            if child_depth > 0 and row is not None:
                self._append_child_context_sync(
                    self._row_to_node(row),
                    depth=child_depth,
                    rows=rows,
                    accumulator=context,
                    seen={node.node_id for node in context},
                )

            unique: list[StoredNode] = []
            seen_ids: set[str] = set()
            for item in context:
                if item.node_id in seen_ids:
                    continue
                seen_ids.add(item.node_id)
                unique.append(item)
            results.append(unique)
        return results

    def _append_child_context_sync(
        self,
        node: StoredNode,
        *,
        depth: int,
        rows: dict[str, dict],
        accumulator: list[StoredNode],
        seen: set[str],
    ) -> None:
        if depth <= 0:
            return
        for child_id in node.child_ids:
            if child_id in seen:
                continue
            row = rows.get(child_id)
            if row is None:
                continue
            seen.add(child_id)
            child = self._row_to_node(row)
            accumulator.append(child)
            self._append_child_context_sync(
                child,
                depth=depth - 1,
                rows=rows,
                accumulator=accumulator,
                seen=seen,
            )

    async def get_context_batch(
        self,
        specs: Sequence[tuple[str, int, int]],
    ) -> list[list[StoredNode]]:
        if not specs:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_get_context_batch,
            list(specs),
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
