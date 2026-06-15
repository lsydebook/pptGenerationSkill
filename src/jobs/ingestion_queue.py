"""低优先级入库（写）队列：限流 + 后台 worker；job 状态存 Redis。"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.cache.job_store import JobStore
from src.cache.retrieval_cache import retrieval_cache
from src.cache.upload_storage import (
    cleanup_indexing_upload,
    load_indexing_request,
    persist_indexing_upload,
)
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
class _QueuedJob:
    job_id: str


def _result_summary(result: IndexingResult) -> dict[str, Any]:
    return {
        "filename": result.filename,
        "content_type": result.content_type,
        "indexing": result.indexing,
    }


class IngestionJobManager:
    """有界写队列 + worker；状态板在 Redis，上传文件落盘。"""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[_QueuedJob | None] | None = None
        self._workers: list[asyncio.Task] = []
        self._lock = asyncio.Lock()
        self._pending_count = 0
        self._running_count = 0
        self._jobs = JobStore()

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
        persist_indexing_upload(job_id, request)

        async with self._lock:
            self._pending_count += 1
            queue_position = self._pending_count

        await self._jobs.create_pending(
            job_id,
            filename=filename_hint,
            queue_position=queue_position,
        )

        try:
            self._queue.put_nowait(_QueuedJob(job_id=job_id))
        except asyncio.QueueFull as exc:
            async with self._lock:
                self._pending_count = max(0, self._pending_count - 1)
            cleanup_indexing_upload(job_id)
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

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        return await self._jobs.get(job_id)

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
        await self._jobs.patch(
            job_id,
            status=JobStatus.RUNNING.value,
            started_at=time.time(),
            queue_position=None,
        )

        logger.info("ingestion job start job_id=%s worker=%s", job_id, worker_idx)
        try:
            request = load_indexing_request(job_id)
            result = await run_indexing(request)
            await self._jobs.patch(
                job_id,
                status=JobStatus.SUCCEEDED.value,
                finished_at=time.time(),
                filename=result.filename,
                result=_result_summary(result),
            )
            await retrieval_cache.bump_version()
            logger.info(
                "ingestion job done job_id=%s batches=%s",
                job_id,
                len(result.indexing),
            )
        except IndexingError as exc:
            await self._jobs.patch(
                job_id,
                status=JobStatus.FAILED.value,
                finished_at=time.time(),
                error=str(exc),
            )
            logger.warning("ingestion job failed job_id=%s error=%s", job_id, exc)
        except Exception as exc:  # noqa: BLE001
            await self._jobs.patch(
                job_id,
                status=JobStatus.FAILED.value,
                finished_at=time.time(),
                error=str(exc),
            )
            logger.exception("ingestion job failed job_id=%s", job_id)
        finally:
            cleanup_indexing_upload(job_id)


ingestion_jobs = IngestionJobManager()
