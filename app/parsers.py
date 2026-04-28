import asyncio
import os
import tempfile
from typing import Any

from llama_index.core import SimpleDirectoryReader

from .config import IMAGE_EXTS, PARSE_CONCURRENCY, SUPPORTED_EXTS


class ParseError(RuntimeError):
    pass


_SEMAPHORE = asyncio.Semaphore(PARSE_CONCURRENCY)


def _safe_filename(filename: str) -> str:
    name = os.path.basename(filename).strip()
    return name or "upload"


def _parse_with_llamaindex(path: str, ext: str) -> list[dict[str, Any]]:
    reader = SimpleDirectoryReader(path, required_exts=[ext], filename_as_id=True)
    docs = reader.load_data()
    return [{"text": doc.text, "metadata": dict(doc.metadata or {})} for doc in docs]


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

        if ext in IMAGE_EXTS:
            metadata = {
                "source_filename": filename,
                "content_type": content_type or "",
                "image_file": "true",
            }
            if note:
                metadata["note"] = note
            return [{"text": note or "", "metadata": metadata}]

        docs = _parse_with_llamaindex(tmpdir, ext)
        for doc in docs:
            doc_metadata = doc.setdefault("metadata", {})
            doc_metadata.setdefault("source_filename", filename)
            if content_type:
                doc_metadata.setdefault("content_type", content_type)
            if note:
                doc_metadata.setdefault("note", note)
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
