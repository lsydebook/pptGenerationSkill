"""检索结果 Redis 缓存（Phase A：缓存 RetrieveResponse JSON）。"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from src.config.indexing_config import RAG_VEC_COLLECTION_SUFFIX
from src.config.llm_config import EMBEDDING_MODEL
from src.config.logging_config import get_logger
from src.config.redis_config import (
    RAG_CACHE_ENABLED,
    RAG_CACHE_PREFIX,
    RAG_CACHE_TTL_SECONDS,
    RAG_CACHE_VERSION_KEY,
)
from src.config.retrieval_config import (
    CHILD_DEPTH,
    DEDUPLICATE_RETRIEVAL,
    PARENT_DEPTH,
    RERANK_STRATEGY,
    SNIPPET_DEDUP,
    TOP_K_FINAL,
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


def _static_cache_fingerprint() -> dict[str, Any]:
    return {
        "rerank": RERANK_STRATEGY,
        "top_k_final": TOP_K_FINAL,
        "parent_depth": PARENT_DEPTH,
        "child_depth": CHILD_DEPTH,
        "snippet_dedup": SNIPPET_DEDUP,
        "dedup": DEDUPLICATE_RETRIEVAL,
        "collection": RAG_VEC_COLLECTION_SUFFIX,
        "embedding": EMBEDDING_MODEL,
    }


class RetrievalCache:
    def __init__(self) -> None:
        self._local_version: str | None = None

    @property
    def enabled(self) -> bool:
        return RAG_CACHE_ENABLED

    async def _ensure_version(self) -> str:
        if self._local_version is not None:
            return self._local_version
        value = await get_redis().get(RAG_CACHE_VERSION_KEY)
        self._local_version = value or "0"
        return self._local_version

    async def get_version(self) -> str:
        return await self._ensure_version()

    async def bump_version(self) -> int:
        version = await get_redis().incr(RAG_CACHE_VERSION_KEY)
        self._local_version = str(version)
        logger.info("rag cache version bumped ver=%s", version)
        return int(version)

    async def _refresh_version_if_needed(self) -> bool:
        remote = await get_redis().get(RAG_CACHE_VERSION_KEY) or "0"
        if remote == self._local_version:
            return False
        self._local_version = remote
        return True

    async def build_key(self, request: _CacheableRetrievalRequest) -> str:
        version = await self._ensure_version()
        params = {
            "top_k": request.top_k,
            "bm25_top_k": request.bm25_top_k,
            "use_planner": request.use_planner,
            "cfg": _static_cache_fingerprint(),
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
        if raw:
            data: dict[str, Any] = json.loads(raw)
            logger.info(
                "retrieval cache hit question_len=%s key_suffix=%s",
                len(request.question),
                key[-12:],
            )
            return data

        if await self._refresh_version_if_needed():
            key = await self.build_key(request)
            raw = await get_redis().get(key)
            if raw:
                data = json.loads(raw)
                logger.info(
                    "retrieval cache hit after version refresh question_len=%s key_suffix=%s",
                    len(request.question),
                    key[-12:],
                )
                return data
        return None

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
