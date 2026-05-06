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


@app.post("/v1/parse", response_model=ParseResponse)
async def parse_file(
    file: UploadFile = File(...),
    note: str | None = Form(default=None),
) -> ParseResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTS:
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

    try:
        documents = await parse_upload(
            data=data,
            filename=file.filename,
            content_type=file.content_type,
            note=note,
        )
    except ParseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ParseResponse(
        filename=file.filename,
        content_type=file.content_type,
        documents=documents,
    )
