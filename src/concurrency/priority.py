"""读写 workload 协调：检索（读）优先于入库（写）。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class WorkloadCoordinator:
    """跟踪进行中的向量检索阶段（不含 LLM query 扩写），供入库 worker 让路。"""

    def __init__(self) -> None:
        self._retrieval_count = 0
        self._lock = asyncio.Lock()

    @property
    def retrieval_active(self) -> bool:
        return self._retrieval_count > 0

    @property
    def retrieval_count(self) -> int:
        return self._retrieval_count

    @asynccontextmanager
    async def retrieval_slot(self) -> AsyncIterator[None]:
        async with self._lock:
            self._retrieval_count += 1
        try:
            yield
        finally:
            async with self._lock:
                self._retrieval_count -= 1


coordinator = WorkloadCoordinator()
