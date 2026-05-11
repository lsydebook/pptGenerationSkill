import asyncio
import hashlib
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Any

from .config import IMAGE_UPLOAD_DIR, PARSE_CONCURRENCY, SUPPORTED_EXTS
from .document_parser.indexer import DocumentIndexer
from .document_parser.parsers import parse_document_path
from .document_parser.types import DocumentPayload, NodeKind


class ParseError(RuntimeError):
    pass


_SEMAPHORE = asyncio.Semaphore(PARSE_CONCURRENCY)

def _safe_filename(filename: str) -> str:
    name = os.path.basename(filename).strip()
    return name or "upload"


def _resolve_image_output_dir() -> str:
    output_dir = os.path.abspath(IMAGE_UPLOAD_DIR)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_image_uploads(
    images: list[tuple[str, bytes, str | None]],
    *,
    note: str | None = None,
) -> list[dict[str, Any]]:
    if not images:
        return []

    output_dir = _resolve_image_output_dir()
    assets: list[dict[str, Any]] = []
    for filename, data, content_type in images:
        safe_name = _safe_filename(filename)
        base, ext = os.path.splitext(safe_name)
        ext = ext.lower() or ".bin"
        digest = hashlib.sha256(data).hexdigest()
        stored_name = f"{base}-{digest[:12]}{ext}"
        path = os.path.join(output_dir, stored_name)
        with open(path, "wb") as handle:
            handle.write(data)

        mime_type = content_type or mimetypes.guess_type(filename)[0]
        metadata = {
            "source_type": "user_upload",
            "stored_filename": stored_name,
        }
        if note:
            metadata["note"] = note

        assets.append(
            {
                "filename": filename or safe_name,
                "mime_type": mime_type or "application/octet-stream",
                "sha256": digest,
                "size_bytes": len(data),
                "path": path,
                "metadata": metadata,
            }
        )
    return assets


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
