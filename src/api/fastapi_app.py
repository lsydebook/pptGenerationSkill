from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.config.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

from src.api.http_schemas import (
    IndexingSummary,
    ParseResponse,
    ParsedDocument,
    RetrieveRequest,
    RetrieveResponse,
)
from src.rag_parsing import IndexingError, IndexingRequest, init_parsing, run_indexing, shutdown_parsing
from src.rag_retrieval import RetrievalError, RetrievalRequest, init_retrieval, run_retrieval, shutdown_retrieval


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("application startup")
    await init_parsing()
    await init_retrieval()
    logger.info("application ready")
    yield
    logger.info("application shutdown")
    shutdown_retrieval()
    shutdown_parsing()


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
        logger.info(
            "POST /v1/parse filename=%s text_len=%s",
            file.filename if file else None,
            len(text or ""),
        )
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
            logger.warning("indexing failed: %s", exc)
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        logger.info(
            "indexing done documents=%s summaries=%s",
            len(result.documents),
            len(result.indexing),
        )
        return ParseResponse(
            filename=result.filename,
            content_type=result.content_type,
            documents=[ParsedDocument(**doc) for doc in result.documents],
            indexing=[IndexingSummary(**item) for item in result.indexing],
        )

    @app.post("/v1/retrieve", response_model=RetrieveResponse)
    async def retrieve(body: RetrieveRequest) -> RetrieveResponse:
        logger.info(
            "POST /v1/retrieve question_len=%s use_planner=%s top_k=%s bm25_top_k=%s",
            len(body.question),
            body.use_planner,
            body.top_k,
            body.bm25_top_k,
        )
        try:
            result = await run_retrieval(
                RetrievalRequest(
                    question=body.question,
                    top_k=body.top_k,
                    bm25_top_k=body.bm25_top_k,
                    use_planner=body.use_planner,
                )
            )
        except RetrievalError as exc:
            logger.warning("retrieval failed: %s", exc)
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        logger.info(
            "retrieval done queries=%s matches=%s snippets=%s",
            len(result.queries),
            len(result.matches),
            len(result.snippets),
        )
        return RetrieveResponse(**result.__dict__)

    return app
