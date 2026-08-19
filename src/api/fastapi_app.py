import asyncio
import os
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from src.api.http_schemas import (
    AnswerRequest,
    AnswerResponse,
    ContextSnippetOut,
    IndexingSummary,
    JobResultSummary,
    JobStatusResponse,
    ParseJobAccepted,
    RetrieveRequest,
    RetrieveResponse,
)
from src.cache.redis_client import close_redis, get_redis, init_redis
from src.concurrency.priority import coordinator
from src.config.logging_config import attach_uvicorn_probe_filter, get_logger, setup_logging
from src.jobs.ingestion_queue import JobStatus, ingestion_jobs
from src.rag_answer import init_answer, run_answer, shutdown_answer
from src.rag_parsing import (
    IndexingError,
    IndexingRequest,
    existing_document_ids,
    init_parsing,
    planned_document_ids,
    shutdown_parsing,
    validate_indexing_request,
)
from src.rag_retrieval import (
    RetrievalError,
    RetrievalRequest,
    init_retrieval,
    run_retrieval,
    shutdown_retrieval,
)

setup_logging()
logger = get_logger(__name__)


def _job_to_status_response(record: dict) -> JobStatusResponse:
    result_payload = None
    raw_result = record.get("result")
    if raw_result:
        result_payload = JobResultSummary(
            filename=raw_result["filename"],
            content_type=raw_result.get("content_type"),
            indexing=[IndexingSummary(**item) for item in raw_result.get("indexing", [])],
        )
    return JobStatusResponse(
        job_id=record["job_id"],
        status=record["status"],
        filename=record.get("filename"),
        created_at=record["created_at"],
        started_at=record.get("started_at"),
        finished_at=record.get("finished_at"),
        queue_position=record.get("queue_position"),
        error=record.get("error"),
        result=result_payload,
    )


def _health_probe_url() -> str:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    return f"http://{probe_host}:{port}/health"


async def _startup_health_probe() -> None:
    url = _health_probe_url()
    deadline = asyncio.get_running_loop().time() + 10.0
    async with httpx.AsyncClient(timeout=2.0) as client:
        while True:
            try:
                response = await client.get(url)
                payload = response.json()
                logger.info(
                    "GET /health %s status=%s redis=%s ingestion_pending=%s ingestion_running=%s retrieval_active=%s",
                    response.status_code,
                    payload.get("status"),
                    payload.get("redis"),
                    payload.get("ingestion_pending"),
                    payload.get("ingestion_running"),
                    payload.get("retrieval_active"),
                )
                return
            except (httpx.HTTPError, ValueError):
                if asyncio.get_running_loop().time() >= deadline:
                    logger.warning("startup health check failed url=%s", url)
                    return
                await asyncio.sleep(0.1)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    attach_uvicorn_probe_filter()
    logger.info("application startup")
    await init_redis()
    await init_parsing()
    await init_retrieval()
    init_answer()
    await ingestion_jobs.start()
    logger.info("application ready")
    health_probe = asyncio.create_task(_startup_health_probe())
    yield
    health_probe.cancel()
    with suppress(asyncio.CancelledError):
        await health_probe
    logger.info("application shutdown")
    await ingestion_jobs.stop()
    shutdown_answer()
    shutdown_retrieval()
    shutdown_parsing()
    await close_redis()


def create_app() -> FastAPI:
    app = FastAPI(title="pptGenerationSkill", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        redis_ok = "unknown"
        try:
            await get_redis().ping()
            redis_ok = "ok"
        except Exception:  # noqa: BLE001
            redis_ok = "error"
        return {
            "status": "ok",
            "redis": redis_ok,
            "ingestion_pending": str(ingestion_jobs.pending_count),
            "ingestion_running": str(ingestion_jobs.running_count),
            "retrieval_active": str(coordinator.retrieval_count),
        }

    @app.post(
        "/v1/parse",
        response_model=ParseJobAccepted,
        responses={
            200: {"description": "文档已在 RAG 中，未重新入库"},
            202: {"description": "已接受，请轮询 poll_url"},
        },
    )
    async def parse_file(
        file: UploadFile | None = File(default=None),
        text: str | None = Form(default=None),
        note: str | None = Form(default=None),
    ) -> JSONResponse:
        file_data = await file.read() if file is not None else None
        request = IndexingRequest(
            file_data=file_data,
            filename=file.filename if file else None,
            content_type=file.content_type if file else None,
            text=text,
            note=note,
        )
        logger.info(
            "POST /v1/parse filename=%s text_len=%s",
            file.filename if file else None,
            len(text or ""),
        )
        try:
            filename_hint = validate_indexing_request(request)
            doc_ids = planned_document_ids(request)
            already = await existing_document_ids(doc_ids)
            if doc_ids and len(already) == len(doc_ids):
                logger.info(
                    "POST /v1/parse already_in_rag filename=%s document_ids=%s",
                    filename_hint,
                    already,
                )
                payload = ParseJobAccepted(
                    status="already_indexed",
                    filename=filename_hint,
                    already_in_rag=True,
                    message="该文档已在 RAG 中",
                    document_ids=already,
                )
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content=jsonable_encoder(payload),
                )
            job_id = await ingestion_jobs.enqueue(request, filename_hint=filename_hint)
        except IndexingError as exc:
            logger.warning("indexing rejected: %s", exc)
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        job = await ingestion_jobs.get_job(job_id)
        payload = ParseJobAccepted(
            job_id=job_id,
            status=JobStatus.PENDING.value,
            poll_url=f"/v1/jobs/{job_id}",
            filename=filename_hint,
            queue_position=job.get("queue_position") if job else None,
            document_ids=doc_ids,
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=jsonable_encoder(payload),
        )

    @app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job(job_id: str) -> JobStatusResponse:
        job = await ingestion_jobs.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_to_status_response(job)

    @app.post("/v1/retrieve", response_model=RetrieveResponse)
    async def retrieve(body: RetrieveRequest) -> RetrieveResponse:
        logger.info(
            "POST /v1/retrieve question_len=%s use_planner=%s",
            len(body.question),
            body.use_planner,
        )
        try:
            result = await run_retrieval(
                RetrievalRequest(
                    question=body.question,
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

    @app.post("/v1/answer", response_model=AnswerResponse)
    async def answer(body: AnswerRequest) -> AnswerResponse:
        logger.info(
            "POST /v1/answer question_len=%s use_planner=%s",
            len(body.question),
            body.use_planner,
        )
        try:
            from src.rag_answer import AnswerRequest as AnswerRunRequest

            result = await run_answer(
                AnswerRunRequest(
                    question=body.question,
                    use_planner=body.use_planner,
                )
            )
        except RetrievalError as exc:
            logger.warning("answer failed: %s", exc)
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        return AnswerResponse(
            question=result.question,
            answer=result.answer,
            answer_value=result.answer_value,
            is_blank=result.is_blank,
            ref_ids=result.ref_ids,
            explanation=result.explanation,
            queries=result.queries,
            snippets=[ContextSnippetOut(**item) for item in result.snippets],
            retries=result.retries,
            ensemble_size=result.ensemble_size,
        )

    return app
