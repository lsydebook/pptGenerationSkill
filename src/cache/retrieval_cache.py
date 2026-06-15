"""检索结果 Redis 缓存（Phase A：缓存 RetrieveResponse JSON）。"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from src.config.logging_config import get_logger
from src.config.redis_config import (
    RAG_CACHE_ENABLED,
    RAG_CACHE_PREFIX,
    RAG_CACHE_TTL_SECONDS,
    RAG_CACHE_VERSION_KEY,
)
from src.cache.redis_client import get_redis

logger = get_logger(__name__)


class _CacheableRetrievalRequest(Protocol):
    question: str
    top_k: int | None
    bm25_top_k: int | None
    use_planner: bool


def normalize_question(question: str) -> str:
    return " ".join(question.strip().split())


class RetrievalCache:
    @property
    def enabled(self) -> bool:
        return RAG_CACHE_ENABLED

    async def get_version(self) -> str:
        value = await get_redis().get(RAG_CACHE_VERSION_KEY)
        return value or "0"

    async def bump_version(self) -> int:
        version = await get_redis().incr(RAG_CACHE_VERSION_KEY)
        logger.info("rag cache version bumped ver=%s", version)
        return int(version)

    async def build_key(self, request: _CacheableRetrievalRequest) -> str:
        version = await self.get_version()
        params = {
            "top_k": request.top_k,
            "bm25_top_k": request.bm25_top_k,
            "use_planner": request.use_planner,
        }
        payload = json.dumps(
            {
                "q": normalize_question(request.question),
                "p": params,
                "v": version,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{RAG_CACHE_PREFIX}{digest}"

    async def get(self, request: _CacheableRetrievalRequest) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        key = await self.build_key(request)
        raw = await get_redis().get(key)
        if not raw:
            return None
        data: dict[str, Any] = json.loads(raw)
        logger.info(
            "retrieval cache hit question_len=%s key_suffix=%s",
            len(request.question),
            key[-12:],
        )
        return data

    async def set(self, request: _CacheableRetrievalRequest, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        key = await self.build_key(request)
        await get_redis().set(
            key,
            json.dumps(payload, ensure_ascii=False),
            ex=RAG_CACHE_TTL_SECONDS,
        )
        logger.info(
            "retrieval cache set question_len=%s key_suffix=%s ttl=%s",
            len(request.question),
            key[-12:],
            RAG_CACHE_TTL_SECONDS,
        )


retrieval_cache = RetrievalCache()
