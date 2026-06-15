"""Upstash / Redis 异步客户端（redis-py asyncio）。"""

from __future__ import annotations

from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from redis.asyncio import Redis

from src.config.logging_config import get_logger
from src.config.redis_config import REDIS_URL

logger = get_logger(__name__)

_redis: Redis | None = None


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
        # Windows / 部分环境默认校验证书过严会导致握手后被断开
        query["ssl_cert_reqs"] = ["none"]
        parsed = parsed._replace(query=urlencode(query, doseq=True))

    return urlunparse(parsed)


async def init_redis() -> Redis:
    global _redis
    if _redis is not None:
        return _redis
    if not REDIS_URL:
        raise ValueError("REDIS_URL is required (Upstash connection string)")

    url = _prepare_redis_url(REDIS_URL)
    if url != REDIS_URL.strip():
        logger.info("redis url normalized to TLS (rediss://) for Upstash")

    _redis = Redis.from_url(
        url,
        decode_responses=True,
        health_check_interval=30,
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


def get_redis() -> Redis:
    if _redis is None:
        raise RuntimeError("Redis not initialized")
    return _redis
