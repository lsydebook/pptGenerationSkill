import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

from .config import PARSE_CONCURRENCY, SUPPORTED_EXTS
from .document_parser.indexer import DocumentIndexer
from .document_parser.parsers import parse_document_path
from .document_parser.types import DocumentPayload, NodeKind
from .rag_service import get_rag_indexer


class ParseError(RuntimeError):
    pass


_SEMAPHORE = asyncio.Semaphore(PARSE_CONCURRENCY)


def _safe_filename(filename: str) -> str:
    name = os.path.basename(filename).strip()
    return name or "upload"


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


def _bytes_to_payload(
    data: bytes,
    filename: str,
    content_type: str | None,
    note: str | None,
) -> DocumentPayload:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ParseError(f"unsupported file extension: {ext}")

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


async def parse_and_index_upload(
    data: bytes,
    filename: str,
    content_type: str | None,
    note: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse document bytes, embed, and store in Milvus."""
    async with _SEMAPHORE:
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
        except Exception as exc:  # noqa: BLE001
            raise ParseError(str(exc)) from exc


# Backward-compatible alias
parse_upload = parse_and_index_upload
