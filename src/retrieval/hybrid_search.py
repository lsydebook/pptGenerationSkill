"""混合检索、去重、重排、上下文扩展。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from src.config.embedding import EmbeddingModel
from src.config.retrieval_config import (
    CHILD_DEPTH,
    DEDUPLICATE_RETRIEVAL,
    PARENT_DEPTH,
    RERANK_STRATEGY,
    RETRIEVAL_BM25_TOP_K,
    RETRIEVAL_TOP_K,
    SNIPPET_DEDUP,
    TOP_K_FINAL,
)
from src.parsing.document_types import ContextSnippet, NodeKind, RetrievalMatch
from src.retrieval.context_snippets import matches_to_snippets
from src.retrieval.query_planner import LLMQueryPlanner, SimpleQueryPlanner
from src.config.logging_config import get_logger
from src.storage.milvus_rag_node_store import MilvusPostgresNodeStore

logger = get_logger(__name__)

RerankStrategy = Literal["frequency", "score", "combined"] | None


@dataclass
class RetrievalResult:
    question: str
    queries: list[str]
    matches: list[RetrievalMatch]
    snippets: list[ContextSnippet]


def deduplicate_matches(matches: list[RetrievalMatch]) -> list[RetrievalMatch]:
    seen_ids: set[str] = set()
    unique: list[RetrievalMatch] = []
    for match in matches:
        if match.node.node_id not in seen_ids:
            seen_ids.add(match.node.node_id)
            unique.append(match)
    return unique


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


def rerank_matches(
    matches: list[RetrievalMatch],
    *,
    strategy: str | None,
) -> list[RetrievalMatch]:
    if not strategy or not matches:
        return matches

    node_stats: dict[str, dict] = {}
    for match in matches:
        node_id = match.node.node_id
        if node_id not in node_stats:
            node_stats[node_id] = {"match": match, "frequency": 0, "total_score": 0.0}
        node_stats[node_id]["frequency"] += 1
        node_stats[node_id]["total_score"] += match.score

    unique_matches = [stats["match"] for stats in node_stats.values()]
    key = strategy.lower()

    if key == "frequency":
        unique_matches.sort(
            key=lambda m: (
                node_stats[m.node.node_id]["frequency"],
                node_stats[m.node.node_id]["total_score"],
            ),
            reverse=True,
        )
    elif key == "score":
        unique_matches.sort(
            key=lambda m: node_stats[m.node.node_id]["total_score"],
            reverse=True,
        )
    elif key == "combined":
        max_freq = max(s["frequency"] for s in node_stats.values())
        max_total_score = max(s["total_score"] for s in node_stats.values())
        max_freq = max(max_freq, 1)
        max_total_score = max(max_total_score, 0.001)
        unique_matches.sort(
            key=lambda m: (
                0.4 * (node_stats[m.node.node_id]["frequency"] / max_freq)
                + 0.6
                * (node_stats[m.node.node_id]["total_score"] / max_total_score)
            ),
            reverse=True,
        )
    return unique_matches


async def hybrid_retrieve(
    question: str,
    *,
    store: MilvusPostgresNodeStore,
    embedder: EmbeddingModel,
    planner: LLMQueryPlanner | SimpleQueryPlanner,
    top_k: int | None = None,
    bm25_top_k: int | None = None,
    use_planner: bool = True,
) -> RetrievalResult:
    """Query 扩写 → Dense + BM25 混合检索 → 去重重排 → 上下文扩展。"""
    logger.info("hybrid_retrieve start use_planner=%s", use_planner)
    active_planner = planner if use_planner else SimpleQueryPlanner()
    queries = list(await active_planner.plan(question))
    if not queries:
        raise ValueError("Planner returned no queries.")
    logger.info("step 1/5 query_plan done count=%s queries=%s", len(queries), queries)

    query_vectors = await embedder.embed_text(queries)
    k = top_k if top_k is not None else RETRIEVAL_TOP_K
    bm25_k = bm25_top_k if bm25_top_k is not None else RETRIEVAL_BM25_TOP_K
    search_kinds = {NodeKind.SENTENCE, NodeKind.PARAGRAPH}
    logger.info(
        "step 2/5 query_embed done dim=%s dense_top_k=%s bm25_top_k=%s use_planner=%s",
        query_vectors.shape[1] if len(query_vectors) else 0,
        k,
        bm25_k,
        use_planner,
    )

    all_matches: list[RetrievalMatch] = []
    for idx, (query_text, vector) in enumerate(zip(queries, query_vectors, strict=True), 1):
        dense_hits = await store.search(vector.tolist(), k=k, kinds=search_kinds)
        bm25_hits: list[RetrievalMatch] = []
        if bm25_k > 0 and store.has_bm25_index():
            bm25_hits = await store.search_bm25(query_text, k=bm25_k, kinds=search_kinds)

        query_hits = merge_matches_keep_max_score(dense_hits + bm25_hits)
        all_matches.extend(query_hits)

        dense_preview = [
            (m.node.node_id, round(m.score, 4)) for m in dense_hits[:3]
        ]
        bm25_preview = [
            (m.node.node_id, round(m.score, 4)) for m in bm25_hits[:3]
        ]
        logger.info(
            "step 3/5 hybrid_search query[%s/%s] q=%r dense=%s bm25=%s merged=%s",
            idx,
            len(queries),
            query_text[:80],
            dense_preview,
            bm25_preview if bm25_k > 0 else "off",
            len(query_hits),
        )
    logger.info("step 3/5 hybrid_search done raw_matches=%s", len(all_matches))

    before_dedup = len(all_matches)
    all_matches = tree_deduplicate_matches(all_matches)
    if DEDUPLICATE_RETRIEVAL and not RERANK_STRATEGY:
        all_matches = deduplicate_matches(all_matches)
    if RERANK_STRATEGY:
        all_matches = rerank_matches(all_matches, strategy=RERANK_STRATEGY)
    if TOP_K_FINAL > 0:
        all_matches = all_matches[:TOP_K_FINAL]
    top_preview = [
        (m.node.node_id, round(m.score, 4), m.node.text[:40].replace("\n", " "))
        for m in all_matches[:5]
    ]
    logger.info(
        "step 4/5 dedup_rerank done strategy=%s before=%s after=%s top_k_final=%s top5=%s",
        RERANK_STRATEGY or "none",
        before_dedup,
        len(all_matches),
        TOP_K_FINAL,
        top_preview,
    )

    snippets = await matches_to_snippets(
        all_matches,
        store,
        parent_depth=PARENT_DEPTH,
        child_depth=CHILD_DEPTH,
        dedup=SNIPPET_DEDUP,  # type: ignore[arg-type]
    )

    logger.info(
        "step 5/5 context_expand done snippets=%s parent_depth=%s child_depth=%s dedup=%s",
        len(snippets),
        PARENT_DEPTH,
        CHILD_DEPTH,
        SNIPPET_DEDUP,
    )
    return RetrievalResult(
        question=question,
        queries=queries,
        matches=all_matches,
        snippets=snippets,
    )
