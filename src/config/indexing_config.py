"""RAG 入库流程配置。"""

from __future__ import annotations

from src.config.env_loader import get_env, get_int

# 并发解析/入库请求数上限
PARSE_CONCURRENCY = get_int("PARSE_CONCURRENCY", 8)
# 单文件或单段文本允许的最大体积（MB）
MAX_FILE_SIZE_MB = get_int("MAX_FILE_SIZE_MB", 50)
# 允许上传并解析的文件扩展名
SUPPORTED_EXTS = {".pdf", ".md", ".markdown", ".txt"}

# Milvus collection 名称前缀（如 rag_nodes_vec）
RAG_TABLE_PREFIX = get_env("RAG_TABLE_PREFIX", "rag_nodes")
# 段落向量策略：averaged（子句平均）| full（整段编码）| both（两种都存）
PARAGRAPH_MODE = get_env("PARAGRAPH_MODE", "averaged")
# Milvus 向量索引类型：HNSW | IVF_FLAT | FLAT
MILVUS_INDEX_TYPE = get_env("MILVUS_INDEX_TYPE", "HNSW")
# 向量距离度量：COSINE | IP | L2
MILVUS_METRIC = get_env("MILVUS_METRIC", "COSINE")
# Milvus / Zilliz 连接地址
MILVUS_URI = get_env("MILVUS_URI")
# Milvus 认证 Token
MILVUS_TOKEN = get_env("MILVUS_TOKEN")
# Milvus 数据库名
MILVUS_DB = get_env("MILVUS_DB", "default")
