"""入库队列与读写优先级配置。"""

from __future__ import annotations

from src.config.env_loader import get_bool, get_float, get_int

# 入库（写）队列：同时执行的任务数（worker 数）
INGESTION_MAX_CONCURRENT = get_int("INGESTION_MAX_CONCURRENT", 2)
# 等待中的任务上限（不含正在跑的）；队列满时 POST /v1/parse 才返回 503
INGESTION_QUEUE_MAX_SIZE = get_int("INGESTION_QUEUE_MAX_SIZE", 64)
# 向量检索进行中时，入库 worker 短暂让路（不含 LLM query 扩写阶段）
INGESTION_YIELD_TO_RETRIEVAL = get_bool("INGESTION_YIELD_TO_RETRIEVAL", True)
# 入库让路时的轮询间隔（秒）
INGESTION_YIELD_POLL_SECONDS = get_float("INGESTION_YIELD_POLL_SECONDS", 0.05)
