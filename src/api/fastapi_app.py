from contextlib import asynccontextmanager



from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status



from src.config.logging_config import get_logger, setup_logging



setup_logging()

logger = get_logger(__name__)



from src.api.http_schemas import (

    IndexingSummary,

    JobStatusResponse,

    ParseJobAccepted,

    ParseResponse,

    ParsedDocument,

    RetrieveRequest,

    RetrieveResponse,

)

from src.concurrency.priority import coordinator

from src.jobs.ingestion_queue import JobStatus, ingestion_jobs

from src.rag_parsing import (

    IndexingError,

    IndexingRequest,

    init_parsing,

    shutdown_parsing,

    validate_indexing_request,

)

from src.rag_retrieval import RetrievalError, RetrievalRequest, init_retrieval, run_retrieval, shutdown_retrieval





def _job_to_status_response(job) -> JobStatusResponse:

    result_payload = None

    if job.result:

        result_payload = ParseResponse(

            filename=job.result["filename"],

            content_type=job.result.get("content_type"),

            documents=[ParsedDocument(**doc) for doc in job.result.get("documents", [])],

            indexing=[IndexingSummary(**item) for item in job.result.get("indexing", [])],

        )

    return JobStatusResponse(

        job_id=job.job_id,

        status=job.status.value,

        filename=job.filename,

        created_at=job.created_at,

        started_at=job.started_at,

        finished_at=job.finished_at,

        queue_position=job.queue_position,

        error=job.error,

        result=result_payload,

    )





@asynccontextmanager

async def lifespan(_app: FastAPI):

    logger.info("application startup")

    await init_parsing()

    await init_retrieval()

    await ingestion_jobs.start()

    logger.info("application ready")

    yield

    logger.info("application shutdown")

    await ingestion_jobs.stop()

    shutdown_retrieval()

    shutdown_parsing()





def create_app() -> FastAPI:

    app = FastAPI(title="pptGenerationSkill", version="0.2.0", lifespan=lifespan)



    @app.get("/health")

    async def health() -> dict[str, str]:

        return {

            "status": "ok",

            "ingestion_pending": str(ingestion_jobs.pending_count),

            "ingestion_running": str(ingestion_jobs.running_count),

            "retrieval_active": str(coordinator.retrieval_count),

        }



    @app.post(

        "/v1/parse",

        response_model=ParseJobAccepted,

        status_code=status.HTTP_202_ACCEPTED,

    )

    async def parse_file(

        file: UploadFile | None = File(default=None),

        text: str | None = Form(default=None),

        note: str | None = Form(default=None),

    ) -> ParseJobAccepted:

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

            job_id = await ingestion_jobs.enqueue(request, filename_hint=filename_hint)

        except IndexingError as exc:

            logger.warning("indexing rejected: %s", exc)

            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc



        job = await ingestion_jobs.get_job(job_id)

        return ParseJobAccepted(

            job_id=job_id,

            status=JobStatus.PENDING.value,

            poll_url=f"/v1/jobs/{job_id}",

            filename=filename_hint,

            queue_position=job.queue_position if job else None,

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


