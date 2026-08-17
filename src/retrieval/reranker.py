"""OpenAI / Jina / TEI compatible rerank client."""

from __future__ import annotations

from typing import Sequence

import httpx

from src.config.llm_config import (
    RERANK_API_KEY,
    RERANK_BASE_URL,
    RERANK_CANDIDATES,
    RERANK_ENABLED,
    RERANK_MODEL,
    RERANK_TOP_N,
)
from src.config.logging_config import get_logger
from src.config.retrieval_config import TOP_K_FINAL
from src.parsing.document_types import RetrievalMatch

logger = get_logger(__name__)

_MAX_DOC_CHARS = 4000


def _endpoint_candidates(base_url: str) -> list[str]:
    root = (base_url or "").rstrip("/")
    if not root:
        return []
    if root.endswith("/rerank"):
        return [root]
    if root.endswith("/v1"):
        return [f"{root}/rerank", f"{root[:-3]}/rerank"]
    return [f"{root}/v1/rerank", f"{root}/rerank"]


async def rerank_matches(
    question: str,
    matches: Sequence[RetrievalMatch],
    *,
    top_n: int | None = None,
) -> list[RetrievalMatch]:
    """对融合后的候选做 cross-encoder rerank；失败时原样返回。"""
    if not RERANK_ENABLED or not matches:
        return list(matches)
    if not RERANK_BASE_URL or not RERANK_API_KEY:
        logger.warning("rerank skipped: missing RERANK_BASE_URL or RERANK_API_KEY")
        return list(matches)

    # 调用方已决定候选集（融合 top_k，或扩展邻居后的去重列表）
    cap = max(1, RERANK_CANDIDATES * 3)
    candidates = list(matches[:cap])
    keep = top_n if top_n is not None else (RERANK_TOP_N or TOP_K_FINAL)
    keep = max(1, keep)
    documents = [(m.node.text or "")[:_MAX_DOC_CHARS] for m in candidates]
    headers = {
        "Authorization": f"Bearer {RERANK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": RERANK_MODEL,
        "query": question,
        "documents": documents,
        "top_n": min(keep, len(documents)),
    }

    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=30.0) as client:
        for url in _endpoint_candidates(RERANK_BASE_URL):
            try:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code >= 400:
                    tei_payload = {"query": question, "texts": documents}
                    if url.endswith("/rerank") and not url.endswith("/v1/rerank"):
                        response = await client.post(
                            url, json=tei_payload, headers=headers
                        )
                response.raise_for_status()
                ranked = _parse_rerank_response(response.json(), candidates)
                if ranked:
                    logger.info(
                        "rerank done endpoint=%s in=%s out=%s",
                        url,
                        len(candidates),
                        len(ranked[:keep]),
                    )
                    return ranked[:keep]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("rerank attempt failed url=%s err=%s", url, exc)

    logger.warning("rerank disabled after failures last_error=%s", last_error)
    return list(matches)


def _parse_rerank_response(
    data: object,
    candidates: Sequence[RetrievalMatch],
) -> list[RetrievalMatch]:
    results = None
    if isinstance(data, dict):
        results = data.get("results") or data.get("data")
    elif isinstance(data, list):
        results = data
    if not isinstance(results, list):
        return []

    scored: list[RetrievalMatch] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        index = item.get("index", item.get("corpus_id"))
        score = item.get("relevance_score", item.get("score"))
        if index is None or score is None:
            continue
        try:
            idx = int(index)
            rel = float(score)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(candidates):
            continue
        match = candidates[idx]
        scored.append(RetrievalMatch(node=match.node, score=rel))
    return scored
