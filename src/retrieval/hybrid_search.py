"""混合检索：Dense + BM25 → RRF 融合 → 扩邻居 → neural rerank。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from src.config.embedding import EmbeddingModel
from src.config.llm_config import RERANK_CANDIDATES, RERANK_ENABLED
from src.config.logging_config import get_logger
from src.config.retrieval_config import (
    CHILD_DEPTH,
    DEDUPLICATE_RETRIEVAL,
    INCLUDE_SIBLINGS,
    PARENT_DEPTH,
    RERANK_STRATEGY,
    RETRIEVAL_BM25_TOP_K,
    RETRIEVAL_TOP_K,
    RRF_K,
    TOP_K_FINAL,
)
from src.parsing.document_types import ContextSnippet, NodeKind, RetrievalMatch
from src.retrieval.context_snippets import (
    expand_matches_to_retrieval,
    stitch_consecutive_matches,
)
from src.retrieval.query_planner import LLMQueryPlanner, SimpleQueryPlanner
from src.retrieval.reranker import rerank_matches as neural_rerank_matches
from src.storage.milvus_rag_node_store import MilvusPostgresNodeStore

logger = get_logger(__name__)

RerankStrategy = Literal["rrf", "frequency", "score", "combined"] | None


@dataclass
class RetrievalResult:
    question: str
    queries: list[str]
    matches: list[RetrievalMatch]
    snippets: list[ContextSnippet]


def _clean_metadata_filter(
    metadata_filter: dict[str, str] | None,
) -> dict[str, str] | None:
    if not metadata_filter:
        return None
    cleaned = {
        key: str(value).strip()
        for key, value in metadata_filter.items()
        if value is not None and str(value).strip()
    }
    return cleaned or None


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalMatch]],
    *,
    k: int = 60,
) -> list[RetrievalMatch]:
    """RRF：score(d) = Σ 1 / (k + rank_i(d))，与原始分数尺度无关。"""
    if not ranked_lists:
        return []
    rrf_k = max(1, k)
    scores: dict[str, float] = {}
    best: dict[str, RetrievalMatch] = {}
    for ranked in ranked_lists:
        seen: set[str] = set()
        for rank, match in enumerate(ranked, start=1):
            node_id = match.node.node_id
            if node_id in seen:
                continue
            seen.add(node_id)
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (rrf_k + rank)
            prev = best.get(node_id)
            if prev is None or match.score > prev.score:
                best[node_id] = match
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        RetrievalMatch(node=best[node_id].node, score=score)
        for node_id, score in ordered
    ]


def deduplicate_matches(matches: list[RetrievalMatch]) -> list[RetrievalMatch]:
    """按 node_id 去重，score 取多路命中中的最大值。"""
    best: dict[str, RetrievalMatch] = {}
    for match in matches:
        node_id = match.node.node_id
        prev = best.get(node_id)
        if prev is None or match.score > prev.score:
            best[node_id] = RetrievalMatch(node=match.node, score=match.score)
    return list(best.values())


def _aggregate_node_stats(
    matches: list[RetrievalMatch],
) -> dict[str, dict]:
    node_stats: dict[str, dict] = {}
    for match in matches:
        node_id = match.node.node_id
        if node_id not in node_stats:
            node_stats[node_id] = {
                "match": match,
                "frequency": 0,
                "total_score": 0.0,
                "max_score": 0.0,
            }
        stats = node_stats[node_id]
        stats["frequency"] += 1
        stats["total_score"] += match.score
        if match.score > stats["max_score"]:
            stats["max_score"] = match.score
            stats["match"] = match
    return node_stats


def _final_rerank_score(
    stats: dict,
    strategy: str,
    *,
    max_freq: int,
    max_total_score: float,
) -> float:
    key = strategy.lower()
    if key == "frequency":
        return stats["frequency"] / max_freq
    if key == "score":
        return stats["total_score"] / max_total_score
    if key == "combined":
        return (
            0.4 * (stats["frequency"] / max_freq)
            + 0.6 * (stats["total_score"] / max_total_score)
        )
    return stats["max_score"]


def rerank_matches(
    matches: list[RetrievalMatch],
    *,
    strategy: str | None,
) -> list[RetrievalMatch]:
    if not matches:
        return []

    node_stats = _aggregate_node_stats(matches)
    if not strategy:
        return [
            RetrievalMatch(node=stats["match"].node, score=stats["max_score"])
            for stats in node_stats.values()
        ]

    key = strategy.lower()
    max_freq = max(s["frequency"] for s in node_stats.values())
    max_total_score = max(s["total_score"] for s in node_stats.values())
    max_freq = max(max_freq, 1)
    max_total_score = max(max_total_score, 0.001)

    ranked = sorted(
        node_stats.values(),
        key=lambda stats: _final_rerank_score(
            stats,
            key,
            max_freq=max_freq,
            max_total_score=max_total_score,
        ),
        reverse=True,
    )
    return [
        RetrievalMatch(
            node=stats["match"].node,
            score=_final_rerank_score(
                stats,
                key,
                max_freq=max_freq,
                max_total_score=max_total_score,
            ),
        )
        for stats in ranked
    ]


def merge_matches_keep_max_score(
    matches: list[RetrievalMatch],
) -> list[RetrievalMatch]:
    """同 node_id 合并，保留最高分（避免 dense+bm25 或重复 query 虚增分数）。"""
    best: dict[str, RetrievalMatch] = {}
    for match in matches:
        node_id = match.node.node_id
        if node_id not in best or match.score > best[node_id].score:
            best[node_id] = match
    return list(best.values())


def tree_deduplicate_matches(matches: list[RetrievalMatch]) -> list[RetrievalMatch]:
    """若父节点已在结果中，则移除子节点（避免段落+句子重复计频）。"""
    if not matches:
        return matches

    all_ids = {m.node.node_id for m in matches}
    child_ids: set[str] = set()
    for node_id in all_ids:
        for other_id in all_ids:
            if other_id != node_id and node_id.startswith(other_id + ":"):
                child_ids.add(node_id)
                break

    if not child_ids:
        return matches
    return [m for m in matches if m.node.node_id not in child_ids]


async def hybrid_retrieve(
    question: str,
    *,
    store: MilvusPostgresNodeStore,
    embedder: EmbeddingModel,
    planner: LLMQueryPlanner | SimpleQueryPlanner,
    top_k: int | None = None,
    bm25_top_k: int | None = None,
    use_planner: bool = True,
    metadata_filter: dict[str, str] | None = None,
) -> RetrievalResult:
    """Query 扩写 → Dense + BM25 混合检索 → 融合重排 → 上下文扩展。"""
    logger.info("hybrid_retrieve start use_planner=%s", use_planner)
    active_planner = planner if use_planner else SimpleQueryPlanner()
    queries = list(await active_planner.plan(question))
    if not queries:
        raise ValueError("Planner returned no queries.")
    return await execute_hybrid_search(
        question,
        queries,
        store=store,
        embedder=embedder,
        top_k=top_k,
        bm25_top_k=bm25_top_k,
        metadata_filter=metadata_filter,
    )


async def execute_hybrid_search(
    question: str,
    queries: list[str],
    *,
    store: MilvusPostgresNodeStore,
    embedder: EmbeddingModel,
    top_k: int | None = None,
    bm25_top_k: int | None = None,
    metadata_filter: dict[str, str] | None = None,
) -> RetrievalResult:
    """向量检索 + 融合 + 上下文扩展（不含 LLM query 扩写）。"""
    if not queries:
        raise ValueError("queries must not be empty")

    filters = _clean_metadata_filter(metadata_filter)
    logger.info("execute_hybrid_search start query_count=%s filter=%s", len(queries), filters)
    logger.info("step 1/5 query_plan done count=%s queries=%s", len(queries), queries)

    query_vectors = await embedder.embed_text(queries)
    k = top_k if top_k is not None else RETRIEVAL_TOP_K
    bm25_k = bm25_top_k if bm25_top_k is not None else RETRIEVAL_BM25_TOP_K
    search_kinds = {NodeKind.SENTENCE, NodeKind.PARAGRAPH}
    logger.info(
        "step 2/5 query_embed done dim=%s dense_top_k=%s bm25_top_k=%s query_count=%s",
        query_vectors.shape[1] if len(query_vectors) else 0,
        k,
        bm25_k,
        len(queries),
    )

    vectors = [vector.tolist() for vector in query_vectors]
    use_bm25 = bm25_k > 0 and store.has_bm25_index()

    if use_bm25:
        dense_batches, bm25_batches = await asyncio.gather(
            store.search_dense_batch(
                vectors, k=k, kinds=search_kinds, metadata_filter=filters
            ),
            store.search_bm25_batch(
                queries, k=bm25_k, kinds=search_kinds, metadata_filter=filters
            ),
        )
    else:
        dense_batches = await store.search_dense_batch(
            vectors, k=k, kinds=search_kinds, metadata_filter=filters
        )
        bm25_batches = [[] for _ in queries]

    ranked_lists: list[list[RetrievalMatch]] = []
    all_matches: list[RetrievalMatch] = []
    for idx, (query_text, dense_hits, bm25_hits) in enumerate(
        zip(queries, dense_batches, bm25_batches, strict=True),
        1,
    ):
        ranked_lists.append(dense_hits)
        if use_bm25:
            ranked_lists.append(bm25_hits)
        query_hits = merge_matches_keep_max_score(dense_hits + bm25_hits)
        all_matches.extend(query_hits)
        logger.info(
            "step 3/5 hybrid_search query[%s/%s] q=%r dense=%s bm25=%s merged=%s",
            idx,
            len(queries),
            query_text[:80],
            [(m.node.node_id, round(m.score, 4)) for m in dense_hits[:3]],
            [(m.node.node_id, round(m.score, 4)) for m in bm25_hits[:3]]
            if use_bm25
            else "off",
            len(query_hits),
        )
    logger.info("step 3/5 hybrid_search done raw_matches=%s lists=%s", len(all_matches), len(ranked_lists))

    strategy = (RERANK_STRATEGY or "rrf").lower()
    before_fusion = len(all_matches)
    if strategy == "rrf":
        fused = reciprocal_rank_fusion(ranked_lists, k=RRF_K)
    else:
        fused = rerank_matches(all_matches, strategy=strategy) if strategy else deduplicate_matches(all_matches)
        if DEDUPLICATE_RETRIEVAL and not strategy:
            fused = deduplicate_matches(fused)

    fused = tree_deduplicate_matches(fused)

    snippets: list[ContextSnippet]
    if RERANK_ENABLED:
        seed = fused[: max(RERANK_CANDIDATES, TOP_K_FINAL)]
        expanded = await expand_matches_to_retrieval(
            seed,
            store,
            parent_depth=PARENT_DEPTH,
            child_depth=CHILD_DEPTH,
            include_siblings=INCLUDE_SIBLINGS,
            limit=max(RERANK_CANDIDATES * 3, 80),
        )
        logger.info(
            "step 4/5 expand_before_rerank seed=%s expanded=%s siblings=%s",
            len(seed),
            len(expanded),
            INCLUDE_SIBLINGS,
        )
        fused = await neural_rerank_matches(question, expanded)
        fused = tree_deduplicate_matches(fused)
        if TOP_K_FINAL > 0:
            fused = fused[:TOP_K_FINAL]
        snippets = await stitch_consecutive_matches(fused, store)
    else:
        if TOP_K_FINAL > 0:
            fused = fused[:TOP_K_FINAL]
        neighbor_matches = await expand_matches_to_retrieval(
            fused,
            store,
            parent_depth=PARENT_DEPTH,
            child_depth=CHILD_DEPTH,
            include_siblings=INCLUDE_SIBLINGS,
        )
        snippets = await stitch_consecutive_matches(neighbor_matches, store)

    top_preview = [
        (m.node.node_id, round(m.score, 4), m.node.text[:40].replace("\n", " "))
        for m in fused[:5]
    ]
    logger.info(
        "step 5/5 fusion_rerank done strategy=%s neural=%s before=%s after=%s snippets=%s top5=%s",
        strategy,
        RERANK_ENABLED,
        before_fusion,
        len(fused),
        len(snippets),
        top_preview,
    )
    return RetrievalResult(
        question=question,
        queries=queries,
        matches=fused,
        snippets=snippets,
    )
