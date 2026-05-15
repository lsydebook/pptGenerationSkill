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
IMAGE_UPLOAD_DIR = os.getenv("IMAGE_UPLOAD_DIR", "uploaded_images")
SUPPORTED_EXTS = {".pdf", ".md", ".markdown", ".txt"}

# RAG / indexing
RAG_TABLE_PREFIX = os.getenv("RAG_TABLE_PREFIX", "rag_nodes")
JINA_EMBEDDING_DIM = int(os.getenv("JINA_EMBEDDING_DIM", "1024"))
JINA_EMBEDDING_TASK = os.getenv("JINA_EMBEDDING_TASK", "retrieval")
JINA_PARAGRAPH_MODE = os.getenv("JINA_PARAGRAPH_MODE", "averaged")
MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "HNSW")
MILVUS_METRIC = os.getenv("MILVUS_METRIC", "COSINE")


def resolve_jina_model_path() -> str:
    """Prefer local models/ directory, then JINA_MODEL_PATH, then HuggingFace id."""
    candidates: list[Path] = []

    env_path = os.getenv("JINA_MODEL_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        candidates.append(p if p.is_absolute() else _PROJECT_ROOT / p)

    candidates.extend(
        [
            _PROJECT_ROOT / "models" / "jina-embeddings-v4",
            _PROJECT_ROOT / "models" / "jinaai" / "jina-embeddings-v4",
        ]
    )

    for path in candidates:
        if path.is_dir() and (path / "config.json").exists():
            return str(path.resolve())

    return os.getenv("JINA_MODEL_NAME", "jinaai/jina-embeddings-v4")
