"""Text Embedding Model - Local inference with GPU support."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import torch
from transformers import AutoModel


class EmbeddingModel(Protocol):
    """Protocol for embedding providers."""

    @property
    def dimension(self) -> int:  # pragma: no cover
        ...

    async def embed_text(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        """Return a 2D numpy array of shape (len(texts), dimension)."""


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length, handling zero vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    return vectors / norms


def _detect_device() -> Any:
    """Auto-detect best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        print(f"[Embedder] Using CUDA device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if has_mps:
        print("[Embedder] Using Metal Performance Shaders (MPS) device")
        return torch.device("mps")

    print("[Embedder] Using CPU device (slower, no GPU detected)")
    return torch.device("cpu")


class Embedder:
    """Wrapper around text embedding models with local inference.

    Features:
    - Text embedding with Matryoshka dimensions (128, 256, 512, 1024, 2048)
    - Batch processing for efficiency
    - Lazy model loading (first-use initialization)
    - GPU acceleration with FP16 when available
    - Async interface for integration with async pipelines
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        *,
        task: str = "retrieval",
        truncate_dim: int = 1024,
        device: Any | None = None,
    ) -> None:
        """Initialize embedding model.

        Args:
            model_name: HuggingFace model identifier
            task: Task mode - "retrieval", "text-matching", or "code"
            truncate_dim: Matryoshka dimension (128, 256, 512, 1024, 2048)
            device: Target device (auto-detected if None)

        Raises:
            ValueError: If truncate_dim is not a valid Matryoshka dimension
        """
        # Validate truncate_dim
        valid_dims = [128, 256, 512, 1024, 2048]
        if truncate_dim not in valid_dims:
            raise ValueError(
                f"truncate_dim must be one of {valid_dims}, got {truncate_dim}"
            )

        if task not in {"retrieval", "text-matching", "code"}:
            raise ValueError(
                f"task must be 'retrieval', 'text-matching', or 'code', got {task}"
            )

        resolved_device = _detect_device() if device is None else torch.device(device)

        self._model_name = model_name
        self._task = task
        self._truncate_dim = truncate_dim
        self._device = resolved_device

        # Use FP16 on GPU for 2x speedup
        self._dtype = (
            torch.float16 if resolved_device.type in {"cuda", "mps"} else torch.float32
        )

        # Lazy initialization - model loaded on first use
        self._model: Any | None = None

        # Single-worker executor for thread-safe GPU operations
        self._executor = ThreadPoolExecutor(max_workers=1)

    def __del__(self) -> None:
        """Cleanup executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    def _ensure_model(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        local_path = Path(self._model_name)
        local_only = local_path.is_dir() and (local_path / "config.json").exists()
        source = str(local_path.resolve()) if local_only else self._model_name
        print(f"[Embedder] Loading model: {source} (local_only={local_only})...")
        model = AutoModel.from_pretrained(
            source,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        model = model.to(self._device, dtype=self._dtype)
        model.eval().requires_grad_(False)
        self._model = model
        print(f"[Embedder] Model loaded successfully")

    @property
    def dimension(self) -> int:
        """Return the configured Matryoshka dimension."""
        return self._truncate_dim

    def _sync_encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Synchronous text encoding (called via executor).

        Args:
            texts: List of text strings to encode

        Returns:
            Array of shape (len(texts), truncate_dim) with float32 dtype
        """
        self._ensure_model()
        assert self._model is not None

        if not texts:
            return np.zeros((0, self._truncate_dim), dtype=np.float32)

        # Ensure all inputs are strings
        str_texts = [str(t) for t in texts]

        print(f"[Embedder] Encoding {len(str_texts)} texts (task={self._task}, dim={self._truncate_dim})")

        # Run inference with model's encode_text method
        with torch.no_grad():
            embeddings = self._model.encode_text(
                texts=str_texts,
                task=self._task,
                prompt_name="query",
                truncate_dim=self._truncate_dim,
                max_length=8192,
            )

        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            arr = embeddings.detach().float().cpu().numpy()
        elif isinstance(embeddings, list):
            arr = torch.stack(embeddings).detach().float().cpu().numpy()
        else:
            arr = np.asarray(embeddings)

        return arr.astype(np.float32, copy=False)

    async def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into embedding vectors (async).

        Uses single-worker executor to ensure thread safety for GPU operations.

        Args:
            texts: List of text strings to encode

        Returns:
            Array of shape (len(texts), dimension) with float32 dtype
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_encode_text, texts)

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Alias for embed_text for compatibility."""
        return await self.embed_text(texts)
