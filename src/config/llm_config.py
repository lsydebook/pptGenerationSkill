"""LLM / Embedding 模型相关配置（对应 .env 中的模型类变量）。"""

from __future__ import annotations

from src.config.env_loader import get_bool, get_env, get_int

# --- Query Planner（OpenAI 兼容网关，用于检索前将用户问题扩写为多条 query）---

PLANNER_BASE_URL = get_env("PLANNER_BASE_URL")
PLANNER_API_KEY = get_env("PLANNER_API_KEY")
PLANNER_MODEL = get_env("PLANNER_MODEL", "qwen-latest")
PLANNER_ENABLE_THINKING = get_bool("PLANNER_ENABLE_THINKING", False)
PLANNER_TEMPERATURE = get_env("PLANNER_TEMPERATURE")
PLANNER_MAX_TOKENS = get_env("PLANNER_MAX_TOKENS")
PLANNER_MAX_QUERIES = get_int("PLANNER_MAX_QUERIES", 3)

# --- Embedding（OpenAI 兼容网关，如 bge-m3）---

EMBEDDING_MODEL = get_env("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_DIM = get_int("EMBEDDING_DIM", 1024)
EMBEDDING_BATCH_SIZE = get_int("EMBEDDING_BATCH_SIZE", 16)
# 未单独配置时默认与 Planner 共用同一网关
EMBEDDING_BASE_URL = get_env("EMBEDDING_BASE_URL") or PLANNER_BASE_URL
EMBEDDING_API_KEY = get_env("EMBEDDING_API_KEY") or PLANNER_API_KEY
