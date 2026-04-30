import asyncio
import base64
import hashlib
import mimetypes
import os
import tempfile
import zipfile
from typing import Any

from llama_index.core import SimpleDirectoryReader

from .config import (
    IMAGE_EXTS,
    INCLUDE_IMAGE_BASE64,
    IMAGE_OUTPUT_DIR,
    PARSE_CONCURRENCY,
    SUPPORTED_EXTS,
)


class ParseError(RuntimeError):
    pass


_SEMAPHORE = asyncio.Semaphore(PARSE_CONCURRENCY)


def _safe_filename(filename: str) -> str:
    name = os.path.basename(filename).strip()
    return name or "upload"


def _build_asset(
    data: bytes,
    filename: str,
    kind: str,
    *,
    page_label: str | None = None,
    metadata: dict[str, Any] | None = None,
    path: str | None = None,
    include_base64: bool = False,
) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    asset = {
        "kind": kind,
        "filename": filename,
        "mime_type": mime_type,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    if include_base64:
        asset["data_base64"] = base64.b64encode(data).decode("ascii")
    if page_label is not None:
        asset["page_label"] = page_label
    if metadata:
        asset["metadata"] = metadata
    if path:
        asset["path"] = path
    return asset


def _resolve_output_dir(image_output_dir: str | None) -> str:
    output_dir = image_output_dir or IMAGE_OUTPUT_DIR
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _save_image_bytes(output_dir: str, name_hint: str, data: bytes) -> str:
    base, ext = os.path.splitext(os.path.basename(name_hint))
    ext = ext.lower() or ".bin"
    digest = hashlib.sha256(data).hexdigest()[:12]
    filename = f"{base}-{digest}{ext}"
    path = os.path.join(output_dir, filename)
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def _extract_zip_images(
    file_path: str,
    folder_prefix: str,
    source_format: str,
    output_dir: str,
) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    with zipfile.ZipFile(file_path) as archive:
        for name in archive.namelist():
            if not name.startswith(folder_prefix) or name.endswith("/"):
                continue
            data = archive.read(name)
            file_name = os.path.basename(name) or "image.bin"
            saved_path = _save_image_bytes(output_dir, file_name, data)
            assets.append(
                _build_asset(
                    data,
                    file_name,
                    "image",
                    metadata={"source_format": source_format},
                    path=saved_path,
                    include_base64=INCLUDE_IMAGE_BASE64,
                )
            )
    return assets


def _extract_pdf_images(file_path: str, output_dir: str) -> list[dict[str, Any]]:
    try:
        import pypdf
    except ImportError as exc:
        raise ParseError("pypdf is required to extract PDF images") from exc

    reader = pypdf.PdfReader(file_path)
    assets: list[dict[str, Any]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        for image_index, image in enumerate(page.images, start=1):
            data = image.data
            name = image.name or f"page-{page_index}-image-{image_index}.bin"
            saved_path = _save_image_bytes(output_dir, name, data)
            assets.append(
                _build_asset(
                    data,
                    name,
                    "image",
                    page_label=str(page_index),
                    metadata={"source_format": "pdf", "page": page_index},
                    path=saved_path,
                    include_base64=INCLUDE_IMAGE_BASE64,
                )
            )
    return assets


def _extract_document_images(
    file_path: str,
    ext: str,
    output_dir: str,
) -> list[dict[str, Any]]:
    if ext == ".pdf":
        return _extract_pdf_images(file_path, output_dir)
    if ext == ".docx":
        return _extract_zip_images(file_path, "word/media/", "docx", output_dir)
    if ext in {".pptx", ".pptm"}:
        return _extract_zip_images(file_path, "ppt/media/", "pptx", output_dir)
    return []


def _attach_assets_to_documents(
    docs: list[dict[str, Any]],
    assets: list[dict[str, Any]],
) -> None:
    if not docs or not assets:
        return

    page_assets: dict[str, list[dict[str, Any]]] = {}
    unassigned: list[dict[str, Any]] = []
    for asset in assets:
        page_label = asset.get("page_label")
        if page_label:
            page_assets.setdefault(str(page_label), []).append(asset)
        else:
            unassigned.append(asset)

    if page_assets:
        for doc in docs:
            metadata = doc.get("metadata", {})
            label = metadata.get("page_label") or metadata.get("page")
            if label is None:
                continue
            doc.setdefault("assets", []).extend(page_assets.get(str(label), []))

    if unassigned:
        docs[0].setdefault("assets", []).extend(unassigned)


def _parse_with_llamaindex(path: str, ext: str) -> list[dict[str, Any]]:
    reader = SimpleDirectoryReader(
        input_dir=path,
        input_files=None,
        filename_as_id=True,
        required_exts=[ext],
    )
    docs = reader.load_data()
    return [{"text": doc.text or "", "metadata": dict(doc.metadata or {})} for doc in docs]


def _parse_file_bytes(
    data: bytes,
    filename: str,
    content_type: str | None,
    note: str | None,
    image_output_dir: str | None,
) -> list[dict[str, Any]]:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ParseError(f"unsupported file extension: {ext}")

    safe_name = _safe_filename(filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, safe_name)
        with open(path, "wb") as handle:
            handle.write(data)

        output_dir = _resolve_output_dir(image_output_dir)

        if ext in IMAGE_EXTS:
            docs = [{"text": note or "", "metadata": {}}]
            saved_path = _save_image_bytes(output_dir, safe_name, data)
            image_asset = _build_asset(
                data,
                safe_name,
                "image",
                metadata={"source_format": "image_upload"},
                path=saved_path,
                include_base64=INCLUDE_IMAGE_BASE64,
            )
            for doc in docs:
                doc.setdefault("assets", []).append(image_asset)
                doc.setdefault("metadata", {}).setdefault("image_file", "true")
        else:
            docs = _parse_with_llamaindex(tmpdir, ext)
            assets = _extract_document_images(path, ext, output_dir)
            _attach_assets_to_documents(docs, assets)
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
    image_output_dir: str | None,
) -> list[dict[str, Any]]:
    async with _SEMAPHORE:
        try:
            return await asyncio.to_thread(
                _parse_file_bytes, data, filename, content_type, note, image_output_dir
            )
        except Exception as exc:  # noqa: BLE001
            raise ParseError(str(exc)) from exc
