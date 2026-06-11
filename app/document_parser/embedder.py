"""Text embedding via DashScope MultiModalEmbedding API."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Any, Protocol, Sequence

import numpy as np

try:
    import dashscope
except ImportError:  # pragma: no cover
    dashscope = None  # type: ignore[assignment]


class EmbeddingModel(Protocol):
    """Protocol for embedding providers."""

    @property
    def dimension(self) -> int:  # pragma: no cover
        ...

    async def embed_text(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        """Return a 2D numpy array of shape (len(texts), dimension)."""


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _extract_embeddings(response: Any) -> list[list[float]]:
    if response.status_code != HTTPStatus.OK:
        message = getattr(response, "message", str(response))
        code = getattr(response, "code", "")
        raise RuntimeError(
            f"DashScope MultiModalEmbedding failed: status={response.status_code}, "
            f"code={code}, message={message}"
        )

    output = response.output or {}
    embeddings = output.get("embeddings")
    if not embeddings:
        raise RuntimeError("DashScope response missing output.embeddings")

    sorted_items = sorted(embeddings, key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in sorted_items]


class Embedder:
    """DashScope multimodal embedding wrapper with async batching."""

    def __init__(
        self,
        model_name: str = "tongyi-embedding-vision-flash-2026-03-06",
        *,
        api_key: str | None = None,
        truncate_dim: int | None = None,
        batch_size: int = 16,
        normalize: bool = True,
    ) -> None:
        if dashscope is None:
            raise ImportError(
                "dashscope is required for embedding. Install with: pip install dashscope"
            )

        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for DashScope embeddings")

        dashscope.api_key = api_key

        self._model_name = model_name
        self._batch_size = max(1, batch_size)
        self._normalize = normalize
        self._dimension = truncate_dim or 768
        self._executor = ThreadPoolExecutor(max_workers=1)

    def __del__(self) -> None:
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _sync_embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)

        str_texts = [str(text) for text in texts]
        print(
            f"[Embedder] Encoding {len(str_texts)} texts "
            f"(model={self._model_name}, dim={self._dimension})"
        )

        response = dashscope.MultiModalEmbedding.call(
            model=self._model_name,
            input=[{"text": text} for text in str_texts],
        )
        vectors = np.asarray(_extract_embeddings(response), dtype=np.float32)

        if vectors.ndim != 2:
            raise RuntimeError(f"Unexpected embedding shape: {vectors.shape}")
        if vectors.shape[0] != len(str_texts):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(str_texts)}, got {vectors.shape[0]}"
            )
        if vectors.shape[1] != self._dimension:
            self._dimension = int(vectors.shape[1])
            print(f"[Embedder] Detected embedding dimension: {self._dimension}")

        if self._normalize:
            vectors = _normalize(vectors)
        return vectors.astype(np.float32, copy=False)

    async def _embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if hasattr(dashscope, "AioMultiModalEmbedding"):
            if not texts:
                return np.zeros((0, self._dimension), dtype=np.float32)

            str_texts = [str(text) for text in texts]
            print(
                f"[Embedder] Encoding {len(str_texts)} texts "
                f"(model={self._model_name}, dim={self._dimension})"
            )
            response = await dashscope.AioMultiModalEmbedding.call(
                model=self._model_name,
                input=[{"text": text} for text in str_texts],
            )
            vectors = np.asarray(_extract_embeddings(response), dtype=np.float32)
            if vectors.shape[1] != self._dimension:
                self._dimension = int(vectors.shape[1])
            if self._normalize:
                vectors = _normalize(vectors)
            return vectors.astype(np.float32, copy=False)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_embed_batch, texts)

    async def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)

        batches: list[np.ndarray] = []
        text_list = list(texts)
        for start in range(0, len(text_list), self._batch_size):
            batch = text_list[start : start + self._batch_size]
            batches.append(await self._embed_batch(batch))
        return np.vstack(batches)

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        return await self.embed_text(texts)
