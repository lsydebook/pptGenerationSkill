from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.api.http_schemas import IndexingSummary, ParseResponse, ParsedDocument
from app.rag_pipeline import IndexingError, IndexingRequest, init_rag, run_indexing, shutdown_rag


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await init_rag()
    yield
    shutdown_rag()


def create_app() -> FastAPI:
    app = FastAPI(title="pptGenerationSkill", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/parse", response_model=ParseResponse)
    async def parse_file(
        file: UploadFile | None = File(default=None),
        text: str | None = Form(default=None),
        note: str | None = Form(default=None),
    ) -> ParseResponse:
        file_data = await file.read() if file is not None else None
        try:
            result = await run_indexing(
                IndexingRequest(
                    file_data=file_data,
                    filename=file.filename if file else None,
                    content_type=file.content_type if file else None,
                    text=text,
                    note=note,
                )
            )
        except IndexingError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        return ParseResponse(
            filename=result.filename,
            content_type=result.content_type,
            documents=[ParsedDocument(**doc) for doc in result.documents],
            indexing=[IndexingSummary(**item) for item in result.indexing],
        )

    return app
