"""Redis 异步客户端：优先 Upstash HTTPS REST，否则 REDIS_URL（TCP）。"""

from __future__ import annotations

from typing import Any, Protocol
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from src.config.logging_config import get_logger
from src.config.redis_config import (
    REDIS_URL,
    UPSTASH_REDIS_REST_TOKEN,
    UPSTASH_REDIS_REST_URL,
)

logger = get_logger(__name__)


class RedisLike(Protocol):
    async def ping(self) -> Any: ...
    async def get(self, key: str) -> Any: ...
    async def set(self, key: str, value: Any, ex: int | None = None) -> Any: ...
    async def incr(self, key: str) -> Any: ...
    async def aclose(self) -> None: ...


class _UpstashRestRedis:
    """upstash-redis 走 HTTPS；get 统一成 str，close 对齐 aclose。"""

    def __init__(self, url: str, token: str) -> None:
        from upstash_redis.asyncio import Redis

        self._client = Redis(url=url, token=token)
        self._host = urlparse(url).hostname or url

    async def ping(self) -> Any:
        return await self._client.ping()

    async def get(self, key: str) -> str | None:
        value = await self._client.get(key)
        if value is None:
            return None
        return value if isinstance(value, str) else str(value)

    async def set(self, key: str, value: Any, ex: int | None = None) -> Any:
        return await self._client.set(key, value, ex=ex)

    async def incr(self, key: str) -> Any:
        return await self._client.incr(key)

    async def aclose(self) -> None:
        await self._client.close()


class _TcpRedis:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def ping(self) -> Any:
        return await self._client.ping()

    async def get(self, key: str) -> Any:
        return await self._client.get(key)

    async def set(self, key: str, value: Any, ex: int | None = None) -> Any:
        return await self._client.set(key, value, ex=ex)

    async def incr(self, key: str) -> Any:
        return await self._client.incr(key)

    async def aclose(self) -> None:
        await self._client.aclose()


_redis: RedisLike | None = None


def _prepare_redis_url(url: str) -> str:
    """Upstash 必须用 TLS：redis:// → rediss://。"""
    cleaned = url.strip()
    if not cleaned:
        return cleaned

    parsed = urlparse(cleaned)
    host = (parsed.hostname or "").lower()
    if "upstash.io" in host and parsed.scheme == "redis":
        parsed = parsed._replace(scheme="rediss")

    query = parse_qs(parsed.query, keep_blank_values=True)
    if parsed.scheme == "rediss" and "ssl_cert_reqs" not in query:
        query["ssl_cert_reqs"] = ["none"]
        parsed = parsed._replace(query=urlencode(query, doseq=True))

    return urlunparse(parsed)


def _use_rest() -> bool:
    return bool(UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN)


async def init_redis() -> RedisLike:
    global _redis
    if _redis is not None:
        return _redis

    if _use_rest():
        _redis = _UpstashRestRedis(UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN)
        await _redis.ping()
        logger.info("redis connected via Upstash REST host=%s", urlparse(UPSTASH_REDIS_REST_URL).hostname)
        return _redis

    if not REDIS_URL:
        raise ValueError(
            "Set UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN "
            "(HTTPS) or REDIS_URL (TCP)"
        )

    from redis.asyncio import Redis

    url = _prepare_redis_url(REDIS_URL)
    if url != REDIS_URL.strip():
        logger.info("redis url normalized to TLS (rediss://) for Upstash")

    _redis = _TcpRedis(
        Redis.from_url(
            url,
            decode_responses=True,
            health_check_interval=30,
        )
    )
    await _redis.ping()
    logger.info("redis connected host=%s", urlparse(url).hostname)
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
        logger.info("redis closed")


def get_redis() -> RedisLike:
    if _redis is None:
        raise RuntimeError("Redis not initialized")
    return _redis
