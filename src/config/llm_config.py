"""LLM / Embedding 模型相关配置（对应 .env 中的模型类变量）。"""

from __future__ import annotations

from src.config.env_loader import get_bool, get_env, get_int

# --- Embedding（DashScope，用于入库向量化与检索 query 向量化）---

# 向量化模型名称
EMBEDDING_MODEL = get_env("EMBEDDING_MODEL", "tongyi-embedding-vision-flash-2026-03-06")
# 向量维度（需与模型输出一致；若实际返回维度不同，运行时会自动修正）
EMBEDDING_DIM = get_int("EMBEDDING_DIM", 768)
# 单次 API 请求最多编码的文本条数
EMBEDDING_BATCH_SIZE = get_int("EMBEDDING_BATCH_SIZE", 16)
# DashScope API Key
DASHSCOPE_API_KEY = get_env("DASHSCOPE_API_KEY")

# --- Query Planner（OpenAI 兼容网关，用于检索前将用户问题扩写为多条 query）---

# Chat API 基地址（如校内 LLM 网关）
PLANNER_BASE_URL = get_env("PLANNER_BASE_URL")
# Chat API Key
PLANNER_API_KEY = get_env("PLANNER_API_KEY")
# 扩写 query 使用的对话模型
PLANNER_MODEL = get_env("PLANNER_MODEL", "qwen-latest")
# 是否开启模型思考模式（qwen 等模型建议关闭以节省 token、加快响应）
PLANNER_ENABLE_THINKING = get_bool("PLANNER_ENABLE_THINKING", False)
# 采样温度；留空则使用模型默认值
PLANNER_TEMPERATURE = get_env("PLANNER_TEMPERATURE")
# 单次回复最大 token 数；留空则使用模型默认值
PLANNER_MAX_TOKENS = get_env("PLANNER_MAX_TOKENS")
# 检索 query 总数上限（含用户原始问题，其余为 LLM 扩写）
PLANNER_MAX_QUERIES = get_int("PLANNER_MAX_QUERIES", 3)
