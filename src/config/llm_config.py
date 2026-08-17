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
# 单次 embedding 输入上限（bge-m3 约 8192 token；按字符留余量，0 表示不切窗）
EMBEDDING_MAX_CHARS = get_int("EMBEDDING_MAX_CHARS", 4000)
# 超长文本硬切时的窗口重叠字符数
EMBEDDING_WINDOW_OVERLAP = get_int("EMBEDDING_WINDOW_OVERLAP", 200)
# 未单独配置时默认与 Planner 共用同一网关
EMBEDDING_BASE_URL = get_env("EMBEDDING_BASE_URL") or PLANNER_BASE_URL
EMBEDDING_API_KEY = get_env("EMBEDDING_API_KEY") or PLANNER_API_KEY

# --- Cross-encoder / Rerank API（OpenAI 兼容网关的 /v1/rerank 或 TEI /rerank）---

RERANK_ENABLED = get_bool("RERANK_ENABLED", False)
RERANK_BASE_URL = get_env("RERANK_BASE_URL") or PLANNER_BASE_URL
RERANK_API_KEY = get_env("RERANK_API_KEY") or PLANNER_API_KEY
RERANK_MODEL = get_env("RERANK_MODEL", "jina-reranker-m0")
# 融合后、扩邻居前的种子命中数；扩完后 rerank 会给每个邻居单独打分
RERANK_CANDIDATES = get_int("RERANK_CANDIDATES", 40)
# rerank 后保留条数；0 表示与 TOP_K_FINAL 相同
RERANK_TOP_N = get_int("RERANK_TOP_N", 12)

# --- 生成侧（KohakuRAG answering：context→question、abstain retry、ensemble）---

ANSWER_MAX_RETRIES = get_int("ANSWER_MAX_RETRIES", 1)
ANSWER_K_DELTA = get_int("ANSWER_K_DELTA", 8)
ANSWER_ENSEMBLE_SIZE = get_int("ANSWER_ENSEMBLE_SIZE", 1)
ANSWER_TEMPERATURE = get_env("ANSWER_TEMPERATURE", "0.3")
ANSWER_MAX_TOKENS = get_env("ANSWER_MAX_TOKENS", "1024")
ANSWER_IGNORE_BLANK = get_bool("ANSWER_IGNORE_BLANK", True)
