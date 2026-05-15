"""Shared RAG stack: Jina embedder + Milvus/PostgreSQL + indexing pipeline."""

from __future__ import annotations

import asyncio

from .config import (
    JINA_EMBEDDING_DIM,
    JINA_EMBEDDING_TASK,
    JINA_PARAGRAPH_MODE,
    MILVUS_INDEX_TYPE,
    MILVUS_METRIC,
    RAG_TABLE_PREFIX,
    resolve_jina_model_path,
)
from .document_parser.datastore_milvus_pg import MilvusPostgresNodeStore
from .document_parser.indexing_pipeline import RAGIndexer
from .document_parser.jina_embedder import JinaV4Embedder

_embedder: JinaV4Embedder | None = None
_datastore: MilvusPostgresNodeStore | None = None
_indexer: RAGIndexer | None = None


def _init_sync() -> None:
    global _embedder, _datastore, _indexer

    model_path = resolve_jina_model_path()
    print(f"[RAG] Loading Jina model from: {model_path}")

    _embedder = JinaV4Embedder(
        model_name=model_path,
        task=JINA_EMBEDDING_TASK,
        truncate_dim=JINA_EMBEDDING_DIM,
    )
    _datastore = MilvusPostgresNodeStore(
        dimensions=_embedder.dimension,
        table_prefix=RAG_TABLE_PREFIX,
        paragraph_search_mode=JINA_PARAGRAPH_MODE,
        index_type=MILVUS_INDEX_TYPE,
        metric=MILVUS_METRIC,
    )
    _indexer = RAGIndexer(
        embedding_model=_embedder,
        datastore=_datastore,
        paragraph_embedding_mode=JINA_PARAGRAPH_MODE,
    )
    print("[RAG] Embedder, Milvus, and PostgreSQL ready")


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


def get_embedder() -> JinaV4Embedder:
    if _embedder is None:
        raise RuntimeError("RAG embedder not initialized")
    return _embedder


def get_datastore() -> MilvusPostgresNodeStore:
    if _datastore is None:
        raise RuntimeError("RAG datastore not initialized")
    return _datastore
