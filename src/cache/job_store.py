"""入库 job 状态板（Redis，JSON 摘要 + TTL）。"""

from __future__ import annotations

import json
import time
from typing import Any

from src.config.redis_config import JOB_KEY_PREFIX, JOB_TTL_SECONDS
from src.cache.redis_client import get_redis


class JobStore:
    def _key(self, job_id: str) -> str:
        return f"{JOB_KEY_PREFIX}{job_id}"

    async def create_pending(
        self,
        job_id: str,
        *,
        filename: str | None,
        queue_position: int | None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "job_id": job_id,
            "status": "pending",
            "filename": filename,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "queue_position": queue_position,
            "error": None,
            "result": None,
        }
        await self.save(record)
        return record

    async def save(self, record: dict[str, Any]) -> None:
        job_id = record["job_id"]
        await get_redis().set(
            self._key(job_id),
            json.dumps(record, ensure_ascii=False),
            ex=JOB_TTL_SECONDS,
        )

    async def get(self, job_id: str) -> dict[str, Any] | None:
        raw = await get_redis().get(self._key(job_id))
        if not raw:
            return None
        return json.loads(raw)

    async def patch(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        record = await self.get(job_id)
        if record is None:
            return None
        record.update(fields)
        await self.save(record)
        return record
