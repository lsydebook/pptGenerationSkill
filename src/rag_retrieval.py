"""RAG 检索主流程：Query 扩写 → 混合检索 → 去重重排 → 上下文扩展。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI

from src.config.llm_config import (
    PLANNER_API_KEY,
    PLANNER_BASE_URL,
    PLANNER_ENABLE_THINKING,
    PLANNER_MAX_QUERIES,
    PLANNER_MAX_TOKENS,
    PLANNER_MODEL,
    PLANNER_TEMPERATURE,
)
from src.config.logging_config import get_logger
from src.concurrency.priority import coordinator
from src.rag_parsing import get_datastore, get_embedder
from src.retrieval.hybrid_search import RetrievalResult, execute_hybrid_search
from src.retrieval.query_planner import LLMQueryPlanner, SimpleQueryPlanner

logger = get_logger(__name__)

_planner: LLMQueryPlanner | None = None


class RetrievalError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class RetrievalRequest:
    question: str
    top_k: int | None = None
    bm25_top_k: int | None = None
    use_planner: bool = True


@dataclass
class RetrievalResponse:
    question: str
    queries: list[str]
    matches: list[dict[str, Any]] = field(default_factory=list)
    snippets: list[dict[str, Any]] = field(default_factory=list)


async def init_retrieval() -> None:
    global _planner
    if _planner is not None:
        return

    if not PLANNER_API_KEY:
        raise ValueError("PLANNER_API_KEY is required")
    if not PLANNER_BASE_URL:
        raise ValueError("PLANNER_BASE_URL is required")

    llm_kwargs: dict = {
        "model": PLANNER_MODEL,
        "api_key": PLANNER_API_KEY,
        "base_url": PLANNER_BASE_URL,
    }
    if PLANNER_TEMPERATURE:
        llm_kwargs["temperature"] = float(PLANNER_TEMPERATURE)
    if PLANNER_MAX_TOKENS:
        llm_kwargs["max_tokens"] = int(PLANNER_MAX_TOKENS)
    if not PLANNER_ENABLE_THINKING:
        llm_kwargs["extra_body"] = {"enable_thinking": False}

    planner_llm = ChatOpenAI(**llm_kwargs)
    _planner = LLMQueryPlanner(llm=planner_llm, max_queries=PLANNER_MAX_QUERIES)
    logger.info("init retrieval done planner_model=%s max_queries=%s", PLANNER_MODEL, PLANNER_MAX_QUERIES)


def shutdown_retrieval() -> None:
    global _planner
    _planner = None


def _match_to_dict(match) -> dict[str, Any]:
    node = match.node
    return {
        "node_id": node.node_id,
        "kind": node.kind.value,
        "text": node.text,
        "score": match.score,
        "metadata": node.metadata,
    }


def _snippet_to_dict(snippet) -> dict[str, Any]:
    return {
        "node_id": snippet.node_id,
        "document_title": snippet.document_title,
        "text": snippet.text,
        "score": snippet.score,
        "rank": snippet.rank,
        "metadata": snippet.metadata,
    }


async def run_retrieval(request: RetrievalRequest) -> RetrievalResponse:
    question = request.question.strip()
    if not question:
        raise RetrievalError("question is required", status_code=400)
    if request.use_planner and _planner is None:
        raise RetrievalError("Retrieval pipeline not initialized", status_code=500)

    logger.info(
        "run_retrieval start question_len=%s use_planner=%s top_k=%s bm25_top_k=%s",
        len(question),
        request.use_planner,
        request.top_k,
        request.bm25_top_k,
    )

    try:
        planner: LLMQueryPlanner | SimpleQueryPlanner = (
            _planner if request.use_planner else SimpleQueryPlanner()
        )

        logger.info("run_retrieval step 1/2 query_plan start use_planner=%s", request.use_planner)
        queries = list(await planner.plan(question))
        if not queries:
            raise RetrievalError("Planner returned no queries.", status_code=500)
        logger.info(
            "run_retrieval step 1/2 query_plan done count=%s (outside retrieval_slot)",
            len(queries),
        )

        async with coordinator.retrieval_slot():
            logger.info("run_retrieval step 2/2 vector_search start (inside retrieval_slot)")
            result: RetrievalResult = await execute_hybrid_search(
                question,
                queries,
                store=get_datastore(),
                embedder=get_embedder(),
                top_k=request.top_k,
                bm25_top_k=request.bm25_top_k,
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_retrieval failed question=%r", question[:200])
        raise RetrievalError(str(exc), status_code=500) from exc

    top_matches = [
        (m.node.node_id, round(m.score, 4))
        for m in result.matches[:5]
    ]
    logger.info(
        "run_retrieval done queries=%s matches=%s snippets=%s top5=%s",
        len(result.queries),
        len(result.matches),
        len(result.snippets),
        top_matches,
    )
    return RetrievalResponse(
        question=result.question,
        queries=result.queries,
        matches=[_match_to_dict(m) for m in result.matches],
        snippets=[_snippet_to_dict(s) for s in result.snippets],
    )
