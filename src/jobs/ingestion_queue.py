"""低优先级入库（写）队列：限流 + 后台 worker，不阻塞检索（读）。"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.config.logging_config import get_logger
from src.config.queue_config import (
    INGESTION_MAX_CONCURRENT,
    INGESTION_QUEUE_MAX_SIZE,
    INGESTION_YIELD_POLL_SECONDS,
    INGESTION_YIELD_TO_RETRIEVAL,
)
from src.concurrency.priority import coordinator
from src.rag_parsing import IndexingError, IndexingRequest, IndexingResult, run_indexing

logger = get_logger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class IngestionJob:
    job_id: str
    status: JobStatus
    created_at: float
    filename: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    queue_position: int | None = None


@dataclass
class _QueuedJob:
    job_id: str
    request: IndexingRequest
    filename_hint: str | None


def _indexing_result_to_dict(result: IndexingResult) -> dict[str, Any]:
    return {
        "filename": result.filename,
        "content_type": result.content_type,
        "documents": result.documents,
        "indexing": result.indexing,
    }


class IngestionJobManager:
    """有界写队列 + 有限并发 worker；检索不走此队列。"""

    def __init__(self) -> None:
        self._jobs: dict[str, IngestionJob] = {}
        self._queue: asyncio.Queue[_QueuedJob | None] | None = None
        self._workers: list[asyncio.Task] = []
        self._lock = asyncio.Lock()
        self._pending_count = 0
        self._running_count = 0

    @property
    def pending_count(self) -> int:
        return self._pending_count

    @property
    def running_count(self) -> int:
        return self._running_count

    async def start(self) -> None:
        if self._queue is not None:
            return
        self._queue = asyncio.Queue(maxsize=INGESTION_QUEUE_MAX_SIZE)
        for idx in range(INGESTION_MAX_CONCURRENT):
            self._workers.append(asyncio.create_task(self._worker_loop(idx)))
        logger.info(
            "ingestion queue started max_concurrent=%s max_queue=%s yield_to_retrieval=%s",
            INGESTION_MAX_CONCURRENT,
            INGESTION_QUEUE_MAX_SIZE,
            INGESTION_YIELD_TO_RETRIEVAL,
        )

    async def stop(self) -> None:
        if self._queue is None:
            return
        for _ in self._workers:
            await self._queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._queue = None
        logger.info("ingestion queue stopped")

    async def enqueue(self, request: IndexingRequest, *, filename_hint: str | None) -> str:
        if self._queue is None:
            raise RuntimeError("Ingestion queue not started")

        job_id = str(uuid.uuid4())
        job = IngestionJob(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=time.time(),
            filename=filename_hint,
            queue_position=self._pending_count + 1,
        )
        async with self._lock:
            self._jobs[job_id] = job
            self._pending_count += 1

        queued = _QueuedJob(job_id=job_id, request=request, filename_hint=filename_hint)
        try:
            self._queue.put_nowait(queued)
        except asyncio.QueueFull as exc:
            async with self._lock:
                self._jobs.pop(job_id, None)
                self._pending_count -= 1
            raise IndexingError(
                f"ingestion queue full (max {INGESTION_QUEUE_MAX_SIZE})",
                status_code=503,
            ) from exc

        logger.info(
            "ingestion enqueued job_id=%s filename=%s pending=%s",
            job_id,
            filename_hint,
            self._pending_count,
        )
        return job_id

    async def get_job(self, job_id: str) -> IngestionJob | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def _worker_loop(self, worker_idx: int) -> None:
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break

            async with self._lock:
                self._pending_count = max(0, self._pending_count - 1)

            try:
                if INGESTION_YIELD_TO_RETRIEVAL:
                    while coordinator.retrieval_active:
                        await asyncio.sleep(INGESTION_YIELD_POLL_SECONDS)

                self._running_count += 1
                try:
                    await self._run_job(item, worker_idx=worker_idx)
                finally:
                    self._running_count -= 1
            finally:
                self._queue.task_done()

    async def _run_job(self, item: _QueuedJob, *, worker_idx: int) -> None:
        job_id = item.job_id
        await self._update_job(
            job_id,
            status=JobStatus.RUNNING,
            started_at=time.time(),
            queue_position=None,
        )

        logger.info(
            "ingestion job start job_id=%s worker=%s filename=%s",
            job_id,
            worker_idx,
            item.filename_hint,
        )
        try:
            result = await run_indexing(item.request)
            await self._update_job(
                job_id,
                status=JobStatus.SUCCEEDED,
                finished_at=time.time(),
                result=_indexing_result_to_dict(result),
                filename=result.filename,
            )
            logger.info(
                "ingestion job done job_id=%s documents=%s",
                job_id,
                len(result.documents),
            )
        except IndexingError as exc:
            await self._update_job(
                job_id,
                status=JobStatus.FAILED,
                finished_at=time.time(),
                error=str(exc),
            )
            logger.warning("ingestion job failed job_id=%s error=%s", job_id, exc)
        except Exception as exc:  # noqa: BLE001
            await self._update_job(
                job_id,
                status=JobStatus.FAILED,
                finished_at=time.time(),
                error=str(exc),
            )
            logger.exception("ingestion job failed job_id=%s", job_id)

    async def _update_job(self, job_id: str, **fields: Any) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)


ingestion_jobs = IngestionJobManager()
