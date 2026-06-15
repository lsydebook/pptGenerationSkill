"""BM25 score normalization helpers (Milvus hybrid search)."""


def normalize_bm25_top_k(
    hits: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """将单次 query 的 BM25 原始分按 top-k 做 min-max 归一化到 [0, 1]。"""
    if not hits:
        return []
    max_raw = max(score for _, score in hits)
    min_raw = min(score for _, score in hits)
    if max_raw <= min_raw:
        return [(node_id, 1.0) for node_id, _ in hits]
    span = max_raw - min_raw
    return [(node_id, (raw - min_raw) / span) for node_id, raw in hits]
