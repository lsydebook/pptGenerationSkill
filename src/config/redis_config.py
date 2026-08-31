"""Redis / Upstash 配置（job 状态板 + 检索缓存）。"""

from __future__ import annotations

from src.config.env_loader import get_bool, get_env, get_int

REDIS_URL = get_env("REDIS_URL").strip().strip('"').strip("'")
# 校园网等拦 TCP 6379 时走 Upstash HTTPS REST
UPSTASH_REDIS_REST_URL = get_env("UPSTASH_REDIS_REST_URL").strip().strip('"').strip("'")
UPSTASH_REDIS_REST_TOKEN = get_env("UPSTASH_REDIS_REST_TOKEN").strip().strip('"').strip("'")

# 入库 job 状态在 Redis 中的 TTL（秒）
JOB_TTL_SECONDS = get_int("JOB_TTL_SECONDS", 3600)
# 检索结果缓存 TTL（秒）
RAG_CACHE_TTL_SECONDS = get_int("RAG_CACHE_TTL_SECONDS", 3600)
# 是否启用检索 Redis 缓存
RAG_CACHE_ENABLED = get_bool("RAG_CACHE_ENABLED", True)

JOB_KEY_PREFIX = "job:"
RAG_CACHE_PREFIX = "cache:rag:v1:"
RAG_CACHE_VERSION_KEY = "cache:rag:ver"

# 上传文件暂存目录（入队后立刻落盘，队列不传 bytes）
UPLOAD_DIR = get_env("UPLOAD_DIR", "data/uploads")
