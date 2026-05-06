import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

from .config import PARSE_CONCURRENCY, SUPPORTED_EXTS
from .kohakurag.indexer import DocumentIndexer
from .kohakurag.parsers import parse_document_path
from .kohakurag.types import DocumentPayload, NodeKind


class ParseError(RuntimeError):
    pass


_SEMAPHORE = asyncio.Semaphore(PARSE_CONCURRENCY)
def _safe_filename(filename: str) -> str:
    name = os.path.basename(filename).strip()
    return name or "upload"


def _documents_from_payload(payload: DocumentPayload) -> list[dict[str, Any]]:
    indexer = DocumentIndexer()
    root = indexer.build_tree(payload)
    docs: list[dict[str, Any]] = []
    for node in indexer.flatten(root):
        if node.kind == NodeKind.DOCUMENT:
            continue
        metadata = dict(node.metadata)
        metadata["node_id"] = node.node_id
        metadata["node_kind"] = node.kind.value
        metadata["node_title"] = node.title
        if node.parent_id is not None:
            metadata["parent_id"] = node.parent_id
        docs.append({"text": node.text, "metadata": metadata, "assets": []})
    return docs


def _parse_file_bytes(
    data: bytes,
    filename: str,
    content_type: str | None,
    note: str | None,
) -> list[dict[str, Any]]:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ParseError(f"unsupported file extension: {ext}")

    safe_name = _safe_filename(filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, safe_name)
        with open(path, "wb") as handle:
            handle.write(data)

        doc_id = os.path.splitext(safe_name)[0] or safe_name
        payload = parse_document_path(
            Path(path),
            doc_id=doc_id,
            title=doc_id,
            metadata={},
        )
        docs = _documents_from_payload(payload)
        for doc in docs:
            doc_metadata = doc.setdefault("metadata", {})
            doc_metadata.setdefault("source_filename", filename)
            if content_type:
                doc_metadata.setdefault("content_type", content_type)
            if note:
                doc_metadata.setdefault("note", note)
            doc.setdefault("assets", [])
        return docs


async def parse_upload(
    data: bytes,
    filename: str,
    content_type: str | None,
    note: str | None,
) -> list[dict[str, Any]]:
    async with _SEMAPHORE:
        try:
            return await asyncio.to_thread(
                _parse_file_bytes, data, filename, content_type, note
            )
        except Exception as exc:  # noqa: BLE001
            raise ParseError(str(exc)) from exc
