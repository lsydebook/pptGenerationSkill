"""In-memory BM25 index for hybrid sparse retrieval."""

from __future__ import annotations

import re
from threading import Lock

from rank_bm25 import BM25Okapi

from src.parsing.document_types import NodeKind

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text for BM25."""
    text = text.strip().lower()
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text)


def normalize_bm25_top_k(
    hits: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """将单次 query 的 BM25 原始分按 top-k 做 min-max 归一化到 [0, 1]。

    rank_bm25 输出为非负浮点数，不能套用 KohakuRAG FTS5 的 (score+20)/20 公式。
    """
    if not hits:
        return []
    max_raw = max(score for _, score in hits)
    min_raw = min(score for _, score in hits)
    if max_raw <= min_raw:
        return [(node_id, 1.0) for node_id, _ in hits]
    span = max_raw - min_raw
    return [(node_id, (raw - min_raw) / span) for node_id, raw in hits]


class BM25Index:
    """Thread-safe BM25 index keyed by node_id."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._node_ids: list[str] = []
        self._kinds: list[str] = []
        self._corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._node_ids)

    def clear(self) -> None:
        with self._lock:
            self._node_ids = []
            self._kinds = []
            self._corpus = []
            self._bm25 = None

    def rebuild(
        self,
        entries: list[tuple[str, str, NodeKind | str]],
        *,
        kinds: set[NodeKind] | None = None,
    ) -> None:
        allowed = {k.value for k in kinds} if kinds else None
        filtered: list[tuple[str, str, str]] = []
        for node_id, text, kind in entries:
            kind_value = kind.value if isinstance(kind, NodeKind) else str(kind)
            if allowed is not None and kind_value not in allowed:
                continue
            tokens = tokenize(text)
            if not tokens:
                continue
            filtered.append((node_id, text, kind_value))

        with self._lock:
            self._node_ids = [item[0] for item in filtered]
            self._kinds = [item[2] for item in filtered]
            self._corpus = [tokenize(item[1]) for item in filtered]
            self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def upsert(
        self,
        entries: list[tuple[str, str, NodeKind | str]],
        *,
        kinds: set[NodeKind] | None = None,
    ) -> None:
        if not entries:
            return

        allowed = {k.value for k in kinds} if kinds else None
        with self._lock:
            id_to_idx = {node_id: idx for idx, node_id in enumerate(self._node_ids)}
            for node_id, text, kind in entries:
                kind_value = kind.value if isinstance(kind, NodeKind) else str(kind)
                if allowed is not None and kind_value not in allowed:
                    continue
                tokens = tokenize(text)
                if not tokens:
                    continue
                if node_id in id_to_idx:
                    idx = id_to_idx[node_id]
                    self._kinds[idx] = kind_value
                    self._corpus[idx] = tokens
                else:
                    id_to_idx[node_id] = len(self._node_ids)
                    self._node_ids.append(node_id)
                    self._kinds.append(kind_value)
                    self._corpus.append(tokens)

            self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        kinds: set[NodeKind] | None = None,
    ) -> list[tuple[str, float]]:
        """Return (node_id, normalized_score) sorted by score descending."""
        with self._lock:
            if self._bm25 is None or not self._node_ids:
                return []

            query_tokens = tokenize(query)
            if not query_tokens:
                return []

            raw_scores = self._bm25.get_scores(query_tokens)
            allowed = {kind.value for kind in kinds} if kinds else None

            raw_hits: list[tuple[str, float]] = []
            for node_id, kind_value, raw in zip(
                self._node_ids, self._kinds, raw_scores, strict=True
            ):
                if allowed is not None and kind_value not in allowed:
                    continue
                if raw <= 0:
                    continue
                raw_hits.append((node_id, float(raw)))

            raw_hits.sort(key=lambda item: item[1], reverse=True)
            return normalize_bm25_top_k(raw_hits[:k])
