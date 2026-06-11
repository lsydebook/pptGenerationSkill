"""RAG indexing pipeline: 输入 → 解析 → 向量化 → 入库.

流程:
  1. 文件/文本 → parsing 模块生成 DocumentPayload
  2. 构建文档树 (section / paragraph / sentence)
  3. llm 模块向量化
  4. storage 模块写入 Milvus
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from app.config import (
    DASHSCOPE_API_KEY,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    MAX_FILE_SIZE_MB,
    MILVUS_INDEX_TYPE,
    MILVUS_METRIC,
    PARAGRAPH_MODE,
    PARSE_CONCURRENCY,
    RAG_TABLE_PREFIX,
    SUPPORTED_EXTS,
)
from app.llm.dashscope_embedder import Embedder
from app.parsing.document_payload_builder import parse_document_path, text_to_payload
from app.parsing.document_types import (
    DocumentPayload,
    NodeKind,
    ParagraphPayload,
    SectionPayload,
    SentencePayload,
    StoredNode,
    TreeNode,
)
from app.parsing.text_splitter import split_paragraphs, split_sentences
from app.storage.milvus_rag_node_store import MilvusPostgresNodeStore

ParagraphEmbeddingMode = Literal["averaged", "full", "both"]

# ---------------------------------------------------------------------------
# Runtime singletons (启动时初始化 Embedder + Milvus)
# ---------------------------------------------------------------------------

_embedder: Embedder | None = None
_datastore: MilvusPostgresNodeStore | None = None
_indexer: RAGIndexer | None = None
_PARSE_SEMAPHORE = asyncio.Semaphore(PARSE_CONCURRENCY)


class IndexingError(RuntimeError):
    """RAG 入库流程错误；status_code 供 API 层映射为 HTTP 响应。"""

    def __init__(self, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


# 兼容旧名称
ParseError = IndexingError


@dataclass
class IndexingRequest:
    """一次入库请求的输入（与 HTTP 传输层解耦）。"""

    file_data: bytes | None = None
    filename: str | None = None
    content_type: str | None = None
    text: str | None = None
    note: str | None = None


@dataclass
class IndexingResult:
    """一次入库请求的输出。"""

    filename: str
    content_type: str | None
    documents: list[dict[str, Any]] = field(default_factory=list)
    indexing: list[dict[str, Any]] = field(default_factory=list)


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
        raise RuntimeError("RAG pipeline not initialized; server lifespan may have failed")
    return _indexer


def get_embedder() -> Embedder:
    if _embedder is None:
        raise RuntimeError("RAG embedder not initialized")
    return _embedder


def get_datastore() -> MilvusPostgresNodeStore:
    if _datastore is None:
        raise RuntimeError("RAG datastore not initialized")
    return _datastore


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


# ---------------------------------------------------------------------------
# 入库主入口
# ---------------------------------------------------------------------------


async def run_indexing(request: IndexingRequest) -> IndexingResult:
    """RAG 入库主函数：校验输入 → 解析 → 向量化 → 写入 Milvus。"""
    text_value = (request.text or "").strip()
    has_file = request.file_data is not None

    if not has_file and not text_value:
        raise IndexingError("missing file or text", status_code=400)

    all_documents: list[dict[str, Any]] = []
    indexing_summaries: list[dict[str, Any]] = []
    response_filename = "input.txt"
    response_content_type = "text/plain"

    if has_file:
        if not request.filename:
            raise IndexingError("missing filename", status_code=400)

        ext = os.path.splitext(request.filename)[1].lower()
        if ext not in SUPPORTED_EXTS:
            raise IndexingError(
                f"unsupported file extension: {ext}",
                status_code=415,
            )

        if not request.file_data:
            raise IndexingError("empty file", status_code=400)

        size_mb = len(request.file_data) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise IndexingError(
                f"file too large: {size_mb:.2f} MB",
                status_code=413,
            )

        response_filename = request.filename
        response_content_type = request.content_type
        documents, summary = await parse_and_index_upload(
            data=request.file_data,
            filename=request.filename,
            content_type=request.content_type,
            note=request.note,
        )
        all_documents.extend(documents)
        indexing_summaries.append(summary)

    if text_value:
        data = text_value.encode("utf-8")
        size_mb = len(data) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise IndexingError(
                f"text too large: {size_mb:.2f} MB",
                status_code=413,
            )

        documents, summary = await parse_and_index_upload(
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

    return IndexingResult(
        filename=response_filename,
        content_type=response_content_type,
        documents=all_documents,
        indexing=indexing_summaries,
    )


def _safe_filename(filename: str) -> str:
    name = os.path.basename(filename).strip()
    return name or "upload"


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

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, safe_name)
        with open(path, "wb") as handle:
            handle.write(data)

        doc_id = os.path.splitext(safe_name)[0] or safe_name
        payload_metadata: dict[str, Any] = {}
        if content_type:
            payload_metadata["content_type"] = content_type
        if note:
            payload_metadata["note"] = note

        payload = parse_document_path(
            Path(path),
            doc_id=doc_id,
            title=doc_id,
            metadata=payload_metadata,
        )
        payload.metadata.setdefault("source_filename", filename)
        return payload


def _documents_from_payload(payload: DocumentPayload) -> list[dict[str, Any]]:
    tree_indexer = DocumentIndexer()
    root = tree_indexer.build_tree(payload)
    docs: list[dict[str, Any]] = []
    for node in tree_indexer.flatten(root):
        if node.kind == NodeKind.DOCUMENT:
            continue
        metadata = dict(node.metadata)
        metadata["node_id"] = node.node_id
        metadata["node_kind"] = node.kind.value
        metadata["node_title"] = node.title
        if node.parent_id is not None:
            metadata["parent_id"] = node.parent_id
        docs.append({"text": node.text, "metadata": metadata})
    return docs


async def parse_and_index_upload(
    data: bytes,
    filename: str,
    content_type: str | None,
    note: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """解析上传内容并向量化入库。"""
    async with _PARSE_SEMAPHORE:
        try:
            payload = await asyncio.to_thread(
                _bytes_to_payload, data, filename, content_type, note
            )
            stored_nodes = await get_rag_indexer().index_document(payload)
            documents = await asyncio.to_thread(_documents_from_payload, payload)
            for doc in documents:
                doc_metadata = doc.setdefault("metadata", {})
                doc_metadata.setdefault("source_filename", filename)
                if content_type:
                    doc_metadata.setdefault("content_type", content_type)
                if note:
                    doc_metadata.setdefault("note", note)

            indexing = {
                "document_id": payload.document_id,
                "nodes_indexed": len(stored_nodes),
            }
            return documents, indexing
        except IndexingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise IndexingError(str(exc), status_code=500) from exc


parse_upload = parse_and_index_upload


# ---------------------------------------------------------------------------
# 文档树构建
# ---------------------------------------------------------------------------


def _sentence_payloads_from_text(text: str) -> list[SentencePayload]:
    return [SentencePayload(text=s) for s in split_sentences(text)]


def _paragraph_payload_from_text(text: str) -> ParagraphPayload:
    return ParagraphPayload(
        text=text,
        sentences=_sentence_payloads_from_text(text),
    )


def _sections_from_text(document: DocumentPayload) -> list[SectionPayload]:
    paragraphs = [
        _paragraph_payload_from_text(paragraph)
        for paragraph in split_paragraphs(document.text)
    ]
    return [
        SectionPayload(
            title=document.title,
            paragraphs=paragraphs,
            metadata={"autogenerated": True},
        )
    ]


class DocumentIndexer:
    def build_tree(self, document: DocumentPayload) -> TreeNode:
        counters = {"section": 0, "paragraph": 0, "sentence": 0}
        sections = document.sections or _sections_from_text(document)

        document_text = document.text or "\n\n".join(
            paragraph.text
            for section in sections
            for paragraph in section.paragraphs
        )

        root_metadata = dict(document.metadata)
        root_metadata.setdefault("document_id", document.document_id)
        root_metadata.setdefault("document_title", document.title)
        root = TreeNode(
            node_id=document.document_id,
            parent_id=None,
            kind=NodeKind.DOCUMENT,
            title=document.title,
            text=document_text,
            metadata=root_metadata,
        )

        for section in sections:
            counters["section"] += 1
            section_id = f"{document.document_id}:sec{counters['section']}"
            section_meta = dict(section.metadata)
            section_meta.update(
                {
                    "document_id": document.document_id,
                    "document_title": document.title,
                    "section_index": counters["section"],
                }
            )
            section_node = TreeNode(
                node_id=section_id,
                parent_id=root.node_id,
                kind=NodeKind.SECTION,
                title=section.title,
                text=section.title,
                metadata=section_meta,
            )
            root.children.append(section_node)

            for paragraph in section.paragraphs:
                counters["paragraph"] += 1
                paragraph_id = f"{section_id}:p{counters['paragraph']}"
                paragraph_meta = dict(paragraph.metadata)
                paragraph_meta.update(
                    {
                        "document_id": document.document_id,
                        "document_title": document.title,
                        "section_id": section_id,
                        "section_index": counters["section"],
                        "paragraph_index": counters["paragraph"],
                    }
                )
                paragraph_node = TreeNode(
                    node_id=paragraph_id,
                    parent_id=section_id,
                    kind=NodeKind.PARAGRAPH,
                    title=section.title,
                    text=paragraph.text,
                    metadata=paragraph_meta,
                )
                section_node.children.append(paragraph_node)

                sentences = paragraph.sentences or _sentence_payloads_from_text(
                    paragraph.text
                )
                for sentence in sentences:
                    counters["sentence"] += 1
                    sentence_id = f"{paragraph_id}:s{counters['sentence']}"
                    sentence_meta = dict(sentence.metadata)
                    sentence_meta.update(
                        {
                            "document_id": document.document_id,
                            "document_title": document.title,
                            "section_id": section_id,
                            "paragraph_id": paragraph_id,
                            "sentence_index": counters["sentence"],
                        }
                    )
                    sentence_node = TreeNode(
                        node_id=sentence_id,
                        parent_id=paragraph_id,
                        kind=NodeKind.SENTENCE,
                        title=section.title,
                        text=sentence.text,
                        metadata=sentence_meta,
                    )
                    paragraph_node.children.append(sentence_node)

        return root

    def flatten(self, node: TreeNode) -> Iterable[TreeNode]:
        yield node
        for child in node.children:
            yield from self.flatten(child)


# ---------------------------------------------------------------------------
# 向量化 + 入库
# ---------------------------------------------------------------------------


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def average_embeddings(child_vectors: Sequence[np.ndarray]) -> np.ndarray:
    if not child_vectors:
        raise ValueError("average_embeddings requires at least one child vector.")
    stacked = np.vstack(child_vectors)
    return _normalize_vectors(np.mean(stacked, axis=0, keepdims=True))[0]


class RAGIndexer:
    def __init__(
        self,
        embedding_model: Embedder,
        datastore: MilvusPostgresNodeStore,
        paragraph_embedding_mode: ParagraphEmbeddingMode = "averaged",
    ) -> None:
        self._embedding_model = embedding_model
        self._datastore = datastore
        self._paragraph_embedding_mode = paragraph_embedding_mode
        self._tree_indexer = DocumentIndexer()

    async def index_text(
        self,
        text: str,
        *,
        document_id: str | None = None,
        title: str = "Untitled",
        metadata: dict | None = None,
    ) -> list[StoredNode]:
        payload = text_to_payload(
            document_id=document_id or str(uuid.uuid4()),
            title=title,
            text=text,
            metadata=dict(metadata or {}),
        )
        return await self.index_document(payload)

    async def index_document(self, document: DocumentPayload) -> list[StoredNode]:
        print(f"\n[Indexing] Starting: {document.title}")

        root = self._tree_indexer.build_tree(document)

        print("[Indexing] Computing embeddings...")
        await self._embed_tree(root)

        now_ts = int(datetime.now(timezone.utc).timestamp())
        stored_nodes = [
            self._to_stored(node, created_at=now_ts)
            for node in self._tree_indexer.flatten(root)
        ]

        print(f"[Indexing] Persisting {len(stored_nodes)} nodes...")
        await self._datastore.upsert_nodes(stored_nodes)

        print(f"[Indexing] Complete: {len(stored_nodes)} nodes\n")
        return stored_nodes

    async def _embed_tree(self, root: TreeNode) -> None:
        leaves = [node for node in self._tree_indexer.flatten(root) if not node.children]
        if leaves:
            print(f"  [Embedding] Encoding {len(leaves)} leaf nodes...")
            embeddings = await self._embedding_model.embed_text(
                [leaf.text for leaf in leaves]
            )
            for leaf, vector in zip(leaves, embeddings, strict=True):
                leaf.embedding = vector

        paragraphs = [
            node
            for node in self._tree_indexer.flatten(root)
            if node.kind == NodeKind.PARAGRAPH
        ]

        para_full_map: dict[str, np.ndarray] = {}
        if self._paragraph_embedding_mode in ("full", "both") and paragraphs:
            print(f"  [Embedding] Encoding {len(paragraphs)} paragraph embeddings...")
            para_embeddings = await self._embedding_model.embed_text(
                [p.text for p in paragraphs]
            )
            para_full_map = {
                p.node_id: vec
                for p, vec in zip(paragraphs, para_embeddings, strict=True)
            }

        self._propagate_embeddings(root, para_full_map)

    def _propagate_embeddings(
        self,
        node: TreeNode,
        para_full_map: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if para_full_map is None:
            para_full_map = {}

        if node.embedding is not None:
            return node.embedding

        child_vectors = [
            self._propagate_embeddings(child, para_full_map) for child in node.children
        ]
        if not child_vectors:
            raise ValueError(f"Node {node.node_id} is missing an embedding.")

        averaged_embedding = average_embeddings(child_vectors)

        if node.kind == NodeKind.PARAGRAPH and node.node_id in para_full_map:
            full_embedding = para_full_map[node.node_id]
            if self._paragraph_embedding_mode == "full":
                node.embedding = full_embedding
            elif self._paragraph_embedding_mode == "both":
                node.embedding = averaged_embedding
                node.metadata["full_embedding"] = full_embedding.tobytes().hex()
            else:
                node.embedding = averaged_embedding
        else:
            node.embedding = averaged_embedding

        return node.embedding

    def _to_stored(self, node: TreeNode, *, created_at: int = 0) -> StoredNode:
        if node.embedding is None:
            raise ValueError(f"Node {node.node_id} is missing an embedding.")

        embedding = node.embedding
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        return StoredNode(
            node_id=node.node_id,
            parent_id=node.parent_id,
            kind=node.kind,
            title=node.title,
            text=node.text,
            metadata=node.metadata,
            embedding=embedding,
            child_ids=[child.node_id for child in node.children],
            created_at=created_at,
        )
