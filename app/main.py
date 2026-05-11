import os
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .config import MAX_FILE_SIZE_MB, SUPPORTED_EXTS
from .parsers import ParseError, parse_upload

app = FastAPI(title="pptGenerationSkill", version="0.1.0")


class ParsedDocument(BaseModel):
    text: str
    metadata: dict[str, Any]
    assets: list[dict[str, Any]]


class ParseResponse(BaseModel):
    filename: str
    content_type: str | None
    documents: list[ParsedDocument]


_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


def _is_image_upload(filename: str, content_type: str | None) -> bool:
    if content_type and content_type.startswith("image/"):
        return True
    ext = os.path.splitext(filename)[1].lower()
    return ext in _IMAGE_EXTS


@app.post("/v1/parse", response_model=ParseResponse)
async def parse_file(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    note: str | None = Form(default=None),
) -> ParseResponse:
    text_value = text or ""
    if file is None and not text_value.strip():
        raise HTTPException(status_code=400, detail="missing file or text")

    all_documents: list[dict[str, Any]] = []
    response_filename = "input.txt"
    response_content_type = "text/plain"

    if file is not None:
        if not file.filename:
            raise HTTPException(status_code=400, detail="missing filename")

        ext = os.path.splitext(file.filename)[1].lower()
        is_image = _is_image_upload(file.filename, file.content_type)
        if not is_image and ext not in SUPPORTED_EXTS:
            raise HTTPException(
                status_code=415,
                detail=f"unsupported file extension: {ext}",
            )

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty file")

        size_mb = len(data) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"file too large: {size_mb:.2f} MB",
            )

        response_filename = file.filename
        response_content_type = file.content_type
        if not is_image:
            try:
                file_documents = await parse_upload(
                    data=data,
                    filename=file.filename,
                    content_type=file.content_type,
                    note=note,
                )
            except ParseError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            all_documents.extend(file_documents)

    if text_value.strip():
        data = text_value.encode("utf-8")
        size_mb = len(data) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"text too large: {size_mb:.2f} MB",
            )

        try:
            text_documents = await parse_upload(
                data=data,
                filename="input.txt",
                content_type="text/plain",
                note=note,
            )
        except ParseError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        all_documents.extend(text_documents)

    if file is not None and text_value.strip():
        response_filename = "mixed-input"
        response_content_type = "multipart/form-data"

    return ParseResponse(
        filename=response_filename,
        content_type=response_content_type,
        documents=all_documents,
    )
