import asyncio
import os
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .config import MAX_FILE_SIZE_MB, SUPPORTED_EXTS
from .parsers import ParseError, parse_upload, save_image_uploads

app = FastAPI(title="pptGenerationSkill", version="0.1.0")


class ParsedDocument(BaseModel):
    text: str
    metadata: dict[str, Any]
    assets: list[dict[str, Any]]


class ParseResponse(BaseModel):
    filename: str
    content_type: str | None
    documents: list[ParsedDocument]


class ImageAsset(BaseModel):
    filename: str
    mime_type: str | None
    sha256: str
    size_bytes: int
    path: str
    metadata: dict[str, Any]


class TextImageParseResponse(BaseModel):
    text_documents: list[ParsedDocument]
    images: list[ImageAsset]


class ImageParseResponse(BaseModel):
    images: list[ImageAsset]


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


@app.post("/v1/parse_text_image", response_model=TextImageParseResponse)
async def parse_text_image(
    text: str = Form(...),
    images: list[UploadFile] | None = File(default=None),
    note: str | None = Form(default=None),
) -> TextImageParseResponse:
    if not text.strip():
        raise HTTPException(status_code=400, detail="empty text")

    text_docs = await parse_upload(
        data=text.encode("utf-8"),
        filename="input.txt",
        content_type="text/plain",
        note=note,
    )

    image_payloads: list[tuple[str, bytes, str | None]] = []
    if images:
        for image in images:
            if not image.filename:
                raise HTTPException(status_code=400, detail="missing image filename")
            if image.content_type and not image.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=415,
                    detail=f"unsupported image content type: {image.content_type}",
                )

            data = await image.read()
            if not data:
                raise HTTPException(status_code=400, detail="empty image")

            size_mb = len(data) / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                raise HTTPException(
                    status_code=413,
                    detail=f"image too large: {size_mb:.2f} MB",
                )

            image_payloads.append((image.filename, data, image.content_type))

    image_assets = await asyncio.to_thread(
        save_image_uploads, image_payloads, note=note
    )

    return TextImageParseResponse(
        text_documents=text_docs,
        images=image_assets,
    )


@app.post("/v1/parse_image", response_model=ImageParseResponse)
async def parse_image(
    file: UploadFile = File(...),
    note: str | None = Form(default=None),
) -> ImageParseResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"unsupported image content type: {file.content_type}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")

    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"image too large: {size_mb:.2f} MB",
        )

    image_assets = await asyncio.to_thread(
        save_image_uploads,
        [(file.filename, data, file.content_type)],
        note=note,
    )

    return ImageParseResponse(images=image_assets)
