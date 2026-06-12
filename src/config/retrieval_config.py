"""RAG 检索查询流程配置（不含模型地址，模型见 llm_config.py）。"""

from __future__ import annotations

from src.config.env_loader import get_bool, get_env, get_int

# 每条检索 query 的 Dense 向量召回数量
RETRIEVAL_TOP_K = get_int("RETRIEVAL_TOP_K", 8)
# 每条检索 query 的 BM25 关键词召回数量（0 表示关闭 BM25）
RETRIEVAL_BM25_TOP_K = get_int("RETRIEVAL_BM25_TOP_K", 4)
# 多路检索合并后是否按 node_id 去重（在重排策略为空时生效）
DEDUPLICATE_RETRIEVAL = get_bool("DEDUPLICATE_RETRIEVAL", True)
# 多路结果重排策略：frequency（频次优先）| score（总分优先）| combined（加权综合）
RERANK_STRATEGY = get_env("RERANK_STRATEGY", "combined")
# 去重/重排后保留的 match 数量上限
TOP_K_FINAL = get_int("TOP_K_FINAL", 20)
# 上下文扩展时向上包含几层父节点（如 sentence → paragraph → section）
PARENT_DEPTH = get_int("PARENT_DEPTH", 1)
# 上下文扩展时向下包含几层子节点
CHILD_DEPTH = get_int("CHILD_DEPTH", 0)
# 扩展后的 snippet 去重：none | node_id | tree（去除被父节点覆盖的子节点）
SNIPPET_DEDUP = get_env("SNIPPET_DEDUP", "tree")
