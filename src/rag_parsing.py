"""RAG 入库主流程：解析 → 向量化 → 写入 Milvus。"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.embedding import EmbeddingModel
from src.config.indexing_config import (
    MAX_FILE_SIZE_MB,
    MILVUS_DB,
    MILVUS_INDEX_TYPE,
    MILVUS_METRIC,
    MILVUS_TOKEN,
    MILVUS_URI,
    PARAGRAPH_MODE,
    RAG_TABLE_PREFIX,
    RAG_VEC_COLLECTION_SUFFIX,
    SUPPORTED_EXTS,
)
from src.config.llm_config import EMBEDDING_MODEL
from src.config.logging_config import get_logger

logger = get_logger(__name__)
from src.parsing.document_indexer import DocumentIndexer, documents_from_payload
from src.parsing.document_payload_builder import parse_document_path
from src.parsing.document_types import DocumentPayload
from src.storage.milvus_rag_node_store import MilvusPostgresNodeStore

_embedder: EmbeddingModel | None = None
_datastore: MilvusPostgresNodeStore | None = None
_indexer: DocumentIndexer | None = None


class IndexingError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


ParseError = IndexingError


@dataclass
class IndexingRequest:
    file_data: bytes | None = None
    filename: str | None = None
    content_type: str | None = None
    text: str | None = None
    note: str | None = None


@dataclass
class IndexingResult:
    filename: str
    content_type: str | None
    documents: list[dict[str, Any]] = field(default_factory=list)
    indexing: list[dict[str, Any]] = field(default_factory=list)


async def init_parsing() -> None:
    global _embedder, _datastore, _indexer
    if _indexer is not None:
        return

    logger.info(
        "init parsing start embedding_model=%s paragraph_mode=%s milvus_prefix=%s",
        EMBEDDING_MODEL,
        PARAGRAPH_MODE,
        RAG_TABLE_PREFIX,
    )
    _embedder = EmbeddingModel()
    _datastore = MilvusPostgresNodeStore(
        dimensions=_embedder.dimension,
        table_prefix=RAG_TABLE_PREFIX,
        paragraph_search_mode=PARAGRAPH_MODE,
        milvus_uri=MILVUS_URI,
        milvus_token=MILVUS_TOKEN,
        milvus_db=MILVUS_DB,
        index_type=MILVUS_INDEX_TYPE,
        metric=MILVUS_METRIC,
    )
    _indexer = DocumentIndexer(
        embedding_model=_embedder,
        datastore=_datastore,
        paragraph_embedding_mode=PARAGRAPH_MODE,
    )
    logger.info(
        "init parsing done milvus_prefix=%s collection_suffix=%s bm25=zilliz",
        RAG_TABLE_PREFIX,
        RAG_VEC_COLLECTION_SUFFIX,
    )


def shutdown_parsing() -> None:
    global _embedder, _datastore, _indexer
    _embedder = None
    _datastore = None
    _indexer = None


def get_embedder() -> EmbeddingModel:
    if _embedder is None:
        raise RuntimeError("Parsing pipeline not initialized")
    return _embedder


def get_datastore() -> MilvusPostgresNodeStore:
    if _datastore is None:
        raise RuntimeError("Datastore not initialized")
    return _datastore


def get_indexer() -> DocumentIndexer:
    if _indexer is None:
        raise RuntimeError("Indexer not initialized")
    return _indexer


def validate_indexing_request(request: IndexingRequest) -> str | None:
    """同步校验请求；通过则返回用于展示的 filename_hint。"""
    text_value = (request.text or "").strip()
    has_file = request.file_data is not None

    if not has_file and not text_value:
        raise IndexingError("missing file or text", status_code=400)

    filename_hint = "input.txt"

    if has_file:
        if not request.filename:
            raise IndexingError("missing filename", status_code=400)
        ext = os.path.splitext(request.filename)[1].lower()
        if ext not in SUPPORTED_EXTS:
            raise IndexingError(f"unsupported file extension: {ext}", status_code=415)
        if not request.file_data:
            raise IndexingError("empty file", status_code=400)
        size_mb = len(request.file_data) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise IndexingError(f"file too large: {size_mb:.2f} MB", status_code=413)
        filename_hint = request.filename

    if text_value:
        data = text_value.encode("utf-8")
        size_mb = len(data) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise IndexingError(f"text too large: {size_mb:.2f} MB", status_code=413)
        if has_file:
            filename_hint = "mixed-input"

    return filename_hint


async def run_indexing(request: IndexingRequest) -> IndexingResult:
    text_value = (request.text or "").strip()
    has_file = request.file_data is not None
    logger.info(
        "run_indexing start has_file=%s filename=%s text_len=%s",
        has_file,
        request.filename,
        len(text_value),
    )

    validate_indexing_request(request)
    text_value = (request.text or "").strip()
    has_file = request.file_data is not None

    all_documents: list[dict[str, Any]] = []
    indexing_summaries: list[dict[str, Any]] = []
    response_filename = "input.txt"
    response_content_type = "text/plain"

    if has_file:
        response_filename = request.filename or response_filename
        response_content_type = request.content_type
        documents, summary = await _parse_and_index_upload(
            data=request.file_data,
            filename=request.filename,
            content_type=request.content_type,
            note=request.note,
        )
        all_documents.extend(documents)
        indexing_summaries.append(summary)

    if text_value:
        data = text_value.encode("utf-8")
        documents, summary = await _parse_and_index_upload(
            data=data,
            filename="input.txt",
            content_type="text/plain",
            note=request.note,
        )
        all_documents.extend(documents)
        indexing_summaries.append(summary)

    if has_file and text_value:
        response_filename = "mixed-input"
        response_content_type = "multipart/form-data"

    logger.info(
        "run_indexing done filename=%s documents=%s indexed_batches=%s",
        response_filename,
        len(all_documents),
        len(indexing_summaries),
    )
    return IndexingResult(
        filename=response_filename,
        content_type=response_content_type,
        documents=all_documents,
        indexing=indexing_summaries,
    )


def _safe_filename(filename: str) -> str:
    return os.path.basename(filename).strip() or "upload"


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def document_id_for_bytes(data: bytes, filename: str) -> str:
    """与入库相同的 id：文件名主干 + 内容 sha256 前 12 位。"""
    safe_name = _safe_filename(filename)
    stem = os.path.splitext(safe_name)[0] or safe_name
    return f"{stem}-{_content_hash(data)}"


def planned_document_ids(request: IndexingRequest) -> list[str]:
    """本次请求将会写入的 document_id 列表（去重、保序）。"""
    ids: list[str] = []
    if request.file_data is not None:
        ids.append(document_id_for_bytes(request.file_data, request.filename or "upload"))
    text_value = (request.text or "").strip()
    if text_value:
        ids.append(document_id_for_bytes(text_value.encode("utf-8"), "input.txt"))
    seen: set[str] = set()
    unique: list[str] = []
    for doc_id in ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        unique.append(doc_id)
    return unique


async def existing_document_ids(doc_ids: list[str]) -> list[str]:
    """返回已在 Milvus 中存在的 document_id（根节点 id 即 document_id）。"""
    if not doc_ids:
        return []
    found = await get_datastore().get_nodes(doc_ids)
    return [doc_id for doc_id in doc_ids if doc_id in found]


def _report_metadata(
    *,
    content_type: str | None,
    note: str | None,
) -> dict[str, Any]:
    payload_metadata: dict[str, Any] = {}
    if content_type:
        payload_metadata["content_type"] = content_type
    if note:
        payload_metadata["note"] = note
    return payload_metadata


def _bytes_to_payload(
    data: bytes,
    filename: str,
    content_type: str | None,
    note: str | None,
) -> DocumentPayload:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise IndexingError(f"unsupported file extension: {ext}", status_code=415)

    safe_name = _safe_filename(filename)
    stem = os.path.splitext(safe_name)[0] or safe_name
    doc_id = document_id_for_bytes(data, filename)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, safe_name)
        with open(path, "wb") as handle:
            handle.write(data)

        payload_metadata = _report_metadata(
            content_type=content_type,
            note=note,
        )
        payload = parse_document_path(
            Path(path),
            doc_id=doc_id,
            title=stem,
            metadata=payload_metadata,
        )
        payload.metadata.setdefault("source_filename", filename)
        payload.metadata.setdefault("content_sha256_12", _content_hash(data))
        return payload


async def _parse_and_index_upload(
    data: bytes,
    filename: str,
    content_type: str | None,
    note: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        ext = os.path.splitext(filename)[1].lower()
        logger.info(
            "parse_and_index step 1/3 parse start filename=%s ext=%s size_bytes=%s",
            filename,
            ext,
            len(data),
        )
        payload = await asyncio.to_thread(
            _bytes_to_payload,
            data,
            filename,
            content_type,
            note,
        )
        logger.info(
            "parse_and_index step 2/3 parse done doc_id=%s title=%s text_len=%s",
            payload.document_id,
            payload.title,
            len(payload.text or ""),
        )
        stored_nodes = await get_indexer().index_document(payload)
        documents = await asyncio.to_thread(documents_from_payload, payload)
        for doc in documents:
            doc_metadata = doc.setdefault("metadata", {})
            doc_metadata.setdefault("source_filename", filename)
            if content_type:
                doc_metadata.setdefault("content_type", content_type)
            if note:
                doc_metadata.setdefault("note", note)
        kind_breakdown: dict[str, int] = {}
        for node in stored_nodes:
            key = node.kind.value
            kind_breakdown[key] = kind_breakdown.get(key, 0) + 1
        logger.info(
            "parse_and_index step 3/3 done doc_id=%s nodes=%s kinds=%s bm25_searchable=%s",
            payload.document_id,
            len(stored_nodes),
            kind_breakdown,
            get_datastore().bm25_index_size(),
        )
        return documents, {
            "document_id": payload.document_id,
            "nodes_indexed": len(stored_nodes),
        }
    except IndexingError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("parse_and_index_upload failed filename=%s", filename)
        raise IndexingError(str(exc), status_code=500) from exc
