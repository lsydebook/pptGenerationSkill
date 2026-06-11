"""Shared RAG stack: Embedder + Milvus + indexing pipeline."""

from __future__ import annotations

import asyncio

from .config import (
    DASHSCOPE_API_KEY,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    PARAGRAPH_MODE,
    MILVUS_INDEX_TYPE,
    MILVUS_METRIC,
    RAG_TABLE_PREFIX,
)
from .document_parser.datastore_milvus_pg import MilvusPostgresNodeStore
from .document_parser.indexing_pipeline import RAGIndexer
from .document_parser.embedder import Embedder

_embedder: Embedder | None = None
_datastore: MilvusPostgresNodeStore | None = None
_indexer: RAGIndexer | None = None


def _init_sync() -> None:
    global _embedder, _datastore, _indexer

    print(f"[RAG] Using DashScope embedding model: {EMBEDDING_MODEL}")

    _embedder = Embedder(
        model_name=EMBEDDING_MODEL,
        api_key=DASHSCOPE_API_KEY,
        truncate_dim=EMBEDDING_DIM,
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    _datastore = MilvusPostgresNodeStore(
        dimensions=_embedder.dimension,
        table_prefix=RAG_TABLE_PREFIX,
        paragraph_search_mode=PARAGRAPH_MODE,
        index_type=MILVUS_INDEX_TYPE,
        metric=MILVUS_METRIC,
    )
    _indexer = RAGIndexer(
        embedding_model=_embedder,
        datastore=_datastore,
        paragraph_embedding_mode=PARAGRAPH_MODE,
    )
    print("[RAG] Embedder, Milvus ready")


async def init_rag() -> None:
    if _indexer is not None:
        return
    await asyncio.to_thread(_init_sync)


def shutdown_rag() -> None:
    global _embedder, _datastore, _indexer
    _embedder = None
    _datastore = None
    _indexer = None


def get_rag_indexer() -> RAGIndexer:
    if _indexer is None:
        raise RuntimeError("RAG services not initialized; server lifespan may have failed")
    return _indexer


def get_embedder() -> Embedder:
    if _embedder is None:
        raise RuntimeError("RAG embedder not initialized")
    return _embedder


def get_datastore() -> MilvusPostgresNodeStore:
    if _datastore is None:
        raise RuntimeError("RAG datastore not initialized")
    return _datastore
