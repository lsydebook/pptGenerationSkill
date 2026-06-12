"""DashScope 向量化实现（入库与检索共用）。

模型名称、API Key 等见 llm_config.py / .env。
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Sequence

import numpy as np

from src.config.llm_config import (
    DASHSCOPE_API_KEY,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
)
from src.config.logging_config import get_logger

logger = get_logger(__name__)

try:
    import dashscope
except ImportError:  # pragma: no cover
    dashscope = None  # type: ignore[assignment]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class EmbeddingModel:
    """DashScope MultiModalEmbedding 封装。"""

    def __init__(self) -> None:
        if dashscope is None:
            raise ImportError("dashscope is required")
        if not DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY is required")
        dashscope.api_key = DASHSCOPE_API_KEY
        self._model_name = EMBEDDING_MODEL
        self._batch_size = max(1, EMBEDDING_BATCH_SIZE)
        self.dimension = EMBEDDING_DIM
        self._executor = ThreadPoolExecutor(max_workers=1)

    def __del__(self) -> None:
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    def _sync_embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        response = dashscope.MultiModalEmbedding.call(
            model=self._model_name,
            input=[{"text": text} for text in texts],
        )
        if response.status_code != HTTPStatus.OK:
            message = getattr(response, "message", str(response))
            code = getattr(response, "code", "")
            raise RuntimeError(
                f"DashScope embedding failed: status={response.status_code}, "
                f"code={code}, message={message}"
            )
        items = sorted(
            (response.output or {}).get("embeddings") or [],
            key=lambda item: item.get("index", 0),
        )
        vectors = np.asarray([item["embedding"] for item in items], dtype=np.float32)
        if vectors.shape[1] != self.dimension:
            self.dimension = int(vectors.shape[1])
        return _normalize(vectors).astype(np.float32, copy=False)

    async def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        text_list = [str(t) for t in texts]
        logger.debug("embed_text start count=%s model=%s", len(text_list), self._model_name)
        batches: list[np.ndarray] = []
        loop = asyncio.get_event_loop()
        for start in range(0, len(text_list), self._batch_size):
            batch = text_list[start : start + self._batch_size]
            batches.append(
                await loop.run_in_executor(self._executor, self._sync_embed_batch, batch)
            )
        merged = np.vstack(batches)
        logger.debug("embed_text done shape=%s", merged.shape)
        return merged
