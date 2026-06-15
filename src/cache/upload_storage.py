"""入库请求暂存：API 接收入队后立即落盘，避免 bytes 占内存。"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from src.config.env_loader import project_root
from src.config.redis_config import UPLOAD_DIR
from src.rag_parsing import IndexingRequest

MANIFEST_NAME = "manifest.json"
TEXT_FORM_NAME = "_form_text.txt"


def _upload_root() -> Path:
    root = Path(UPLOAD_DIR)
    if not root.is_absolute():
        root = project_root() / root
    return root


def _safe_basename(filename: str) -> str:
    return os.path.basename(filename).strip() or "upload"


def persist_indexing_upload(job_id: str, request: IndexingRequest) -> Path:
    job_dir = _upload_root() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    stored_filename: str | None = None
    if request.file_data is not None and request.filename:
        stored_filename = _safe_basename(request.filename)
        (job_dir / stored_filename).write_bytes(request.file_data)

    text_value = (request.text or "").strip()
    if text_value:
        (job_dir / TEXT_FORM_NAME).write_text(text_value, encoding="utf-8")

    manifest = {
        "filename": request.filename,
        "stored_filename": stored_filename,
        "content_type": request.content_type,
        "note": request.note,
        "has_form_text": bool(text_value),
    }
    (job_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    return job_dir


def load_indexing_request(job_id: str) -> IndexingRequest:
    job_dir = _upload_root() / job_id
    manifest_path = job_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"upload manifest not found for job {job_id}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_data: bytes | None = None
    stored_filename = manifest.get("stored_filename")
    if stored_filename:
        file_path = job_dir / stored_filename
        if file_path.is_file():
            file_data = file_path.read_bytes()

    text: str | None = None
    text_path = job_dir / TEXT_FORM_NAME
    if text_path.is_file():
        text = text_path.read_text(encoding="utf-8")

    return IndexingRequest(
        file_data=file_data,
        filename=manifest.get("filename"),
        content_type=manifest.get("content_type"),
        text=text,
        note=manifest.get("note"),
    )


def cleanup_indexing_upload(job_id: str) -> None:
    job_dir = _upload_root() / job_id
    if job_dir.is_dir():
        shutil.rmtree(job_dir, ignore_errors=True)
