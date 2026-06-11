import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    return _PROJECT_ROOT


def _load_dotenv() -> None:
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


_load_dotenv()

PARSE_CONCURRENCY = int(os.getenv("PARSE_CONCURRENCY", "8"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
SUPPORTED_EXTS = {".pdf", ".md", ".markdown", ".txt"}

# RAG / indexing
RAG_TABLE_PREFIX = os.getenv("RAG_TABLE_PREFIX", "rag_nodes")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "tongyi-embedding-vision-flash-2026-03-06",
)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
PARAGRAPH_MODE = os.getenv("PARAGRAPH_MODE", "averaged")
MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "HNSW")
MILVUS_METRIC = os.getenv("MILVUS_METRIC", "COSINE")
