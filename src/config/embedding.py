"""OpenAI 兼容网关向量化实现（入库与检索共用）。

模型名称、API Key、Base URL 等见 llm_config.py / .env。
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np
from openai import OpenAI

from src.config.llm_config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
)
from src.config.logging_config import get_logger

logger = get_logger(__name__)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class EmbeddingModel:
    """OpenAI-compatible embeddings API（如 bge-m3）。"""

    def __init__(self) -> None:
        if not EMBEDDING_API_KEY:
            raise ValueError("EMBEDDING_API_KEY is required")
        if not EMBEDDING_BASE_URL:
            raise ValueError("EMBEDDING_BASE_URL is required")
        self._client = OpenAI(
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_BASE_URL,
        )
        self._model_name = EMBEDDING_MODEL
        self._batch_size = max(1, EMBEDDING_BATCH_SIZE)
        self.dimension = EMBEDDING_DIM
        self._executor = ThreadPoolExecutor(max_workers=1)

    def __del__(self) -> None:
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    def _sync_embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        response = self._client.embeddings.create(
            model=self._model_name,
            input=list(texts),
        )
        items = sorted(response.data, key=lambda item: item.index)
        vectors = np.asarray([item.embedding for item in items], dtype=np.float32)
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
