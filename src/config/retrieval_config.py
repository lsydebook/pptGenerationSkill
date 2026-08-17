"""RAG 检索查询流程配置（不含模型地址，模型见 llm_config.py）。"""

from __future__ import annotations

from src.config.env_loader import get_bool, get_env, get_int

# 每条检索 query 的 Dense 向量召回数量
RETRIEVAL_TOP_K = get_int("RETRIEVAL_TOP_K", 8)
# 每条检索 query 的 BM25 关键词召回数量（0 表示关闭 BM25）
RETRIEVAL_BM25_TOP_K = get_int("RETRIEVAL_BM25_TOP_K", 4)
# 多路检索合并后是否按 node_id 去重（在重排策略为空时生效）
DEDUPLICATE_RETRIEVAL = get_bool("DEDUPLICATE_RETRIEVAL", True)
# 多路融合策略：rrf（默认，按排名融合 Dense/BM25/多 query）|
# frequency | score | combined（KohakuRAG 启发式，仅作消融）
RERANK_STRATEGY = get_env("RERANK_STRATEGY", "rrf")
# RRF 常数 k，越大各路 rank 差异越平滑
RRF_K = get_int("RRF_K", 60)
# 去重/重排后保留的 match 数量上限
TOP_K_FINAL = get_int("TOP_K_FINAL", 20)
# 句子命中是否向上取所属段落（1=取段落；段落命中不会再升到 section）
PARENT_DEPTH = get_int("PARENT_DEPTH", 1)
# 上下文扩展时向下包含几层子节点
CHILD_DEPTH = get_int("CHILD_DEPTH", 0)
# 命中后是否拉兄弟节点（在 rerank 之前扩展，由 rerank 决定邻居是否留下）
INCLUDE_SIBLINGS = get_bool("INCLUDE_SIBLINGS", True)
# 每个命中最多扩展多少个邻近兄弟，避免一节数百段一次 query 撑爆 gRPC
MAX_SIBLINGS = get_int("MAX_SIBLINGS", 6)
# 小块召回、大块返回：parent_child 将 sentence 命中提升为所属段落（不提升到 section）
SNIPPET_RETURN_MODE = get_env("SNIPPET_RETURN_MODE", "parent_child")
# 扩展后的 snippet 去重：none | node_id | tree（去除被父节点覆盖的子节点）
SNIPPET_DEDUP = get_env("SNIPPET_DEDUP", "tree")
# 给模型的 snippet：同文档连续段落拼成一段；中间最多补拉几段（1=只补一个空洞）
STITCH_MAX_GAP = get_int("STITCH_MAX_GAP", 1)
