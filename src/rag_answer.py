"""KohakuRAG 生成侧：context→question、abstain 时加大 k 重试、ensemble 投票。"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config.llm_config import (
    ANSWER_ENSEMBLE_SIZE,
    ANSWER_IGNORE_BLANK,
    ANSWER_K_DELTA,
    ANSWER_MAX_RETRIES,
    ANSWER_MAX_TOKENS,
    ANSWER_TEMPERATURE,
    PLANNER_MODEL,
)
from src.config.logging_config import get_logger
from src.config.retrieval_config import RETRIEVAL_BM25_TOP_K, RETRIEVAL_TOP_K
from src.param.param_zh import ANSWER_SYSTEM_PROMPT, ANSWER_USER_TEMPLATE
from src.parsing.document_types import ContextSnippet, public_metadata
from src.concurrency.priority import coordinator
from src.rag_parsing import get_datastore, get_embedder
from src.rag_retrieval import RetrievalError, _build_planner_llm, get_planner
from src.retrieval.hybrid_search import RetrievalResult, execute_hybrid_search
from src.retrieval.query_planner import LLMQueryPlanner, SimpleQueryPlanner, _strip_thinking

logger = get_logger(__name__)

_answer_llm: ChatOpenAI | None = None


@dataclass
class AnswerRequest:
    question: str
    use_planner: bool = True


@dataclass
class AnswerResponse:
    question: str
    answer: str
    answer_value: str = ""
    is_blank: bool = False
    ref_ids: list[str] = field(default_factory=list)
    explanation: str = ""
    queries: list[str] = field(default_factory=list)
    snippets: list[dict[str, Any]] = field(default_factory=list)
    retries: int = 0
    ensemble_size: int = 1


def init_answer() -> None:
    global _answer_llm
    if _answer_llm is not None:
        return
    kwargs = _build_planner_llm()
    if ANSWER_TEMPERATURE:
        kwargs["temperature"] = float(ANSWER_TEMPERATURE)
    if ANSWER_MAX_TOKENS:
        kwargs["max_tokens"] = int(ANSWER_MAX_TOKENS)
    _answer_llm = ChatOpenAI(**kwargs)
    logger.info("init answer done model=%s ensemble=%s", PLANNER_MODEL, ANSWER_ENSEMBLE_SIZE)


def shutdown_answer() -> None:
    global _answer_llm
    _answer_llm = None


def _format_context(snippets: list[ContextSnippet]) -> str:
    if not snippets:
        return "（无检索结果）"
    blocks: list[str] = []
    for snippet in snippets:
        ref = snippet.metadata.get("document_id") or snippet.node_id
        title = snippet.document_title or snippet.metadata.get("document_title") or ""
        header = f"[ref_id={ref}] {title}".strip()
        blocks.append(f"{header}\n{snippet.text}")
    return "\n\n".join(blocks)


def _parse_answer_json(raw: str) -> dict[str, Any]:
    text = _strip_thinking(raw)
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
        if isinstance(data, dict):
            return data
    except Exception:  # noqa: BLE001
        logger.warning("answer json parse failed")
    return {
        "answer": text.strip(),
        "answer_value": "",
        "ref_ids": [],
        "is_blank": False,
        "explanation": "",
    }


def _normalize_vote_key(payload: dict[str, Any]) -> str:
    value = str(payload.get("answer_value") or "").strip()
    if value:
        return value.lower()
    return str(payload.get("answer") or "").strip().lower()


def _ensemble_vote(
    payloads: list[dict[str, Any]],
    *,
    ignore_blank: bool,
) -> dict[str, Any]:
    usable = payloads
    if ignore_blank:
        non_blank = [p for p in payloads if not bool(p.get("is_blank"))]
        if non_blank:
            usable = non_blank
    if not usable:
        return {
            "answer": "",
            "answer_value": "",
            "ref_ids": [],
            "is_blank": True,
            "explanation": "所有生成均判定证据不足",
        }
    counts = Counter(_normalize_vote_key(item) for item in usable)
    winner_key, _ = counts.most_common(1)[0]
    chosen = next(item for item in usable if _normalize_vote_key(item) == winner_key)
    ref_ids: list[str] = []
    for item in usable:
        if _normalize_vote_key(item) != winner_key:
            continue
        for ref in item.get("ref_ids") or []:
            text = str(ref).strip()
            if text and text not in ref_ids:
                ref_ids.append(text)
    chosen = dict(chosen)
    chosen["ref_ids"] = ref_ids
    chosen["is_blank"] = bool(chosen.get("is_blank"))
    return chosen


async def _generate_once(question: str, context: str) -> dict[str, Any]:
    if _answer_llm is None:
        raise RetrievalError("Answer pipeline not initialized", status_code=500)
    prompt = ANSWER_USER_TEMPLATE.format(context=context, question=question)
    response = await _answer_llm.ainvoke(
        [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    return _parse_answer_json(str(response.content or ""))


def _snippet_dicts(snippets: list[ContextSnippet]) -> list[dict[str, Any]]:
    return [
        {
            "node_id": snippet.node_id,
            "document_title": snippet.document_title,
            "text": snippet.text,
            "score": snippet.score,
            "rank": snippet.rank,
            "metadata": public_metadata(snippet.metadata),
        }
        for snippet in snippets
    ]


async def run_answer(request: AnswerRequest) -> AnswerResponse:
    question = request.question.strip()
    if not question:
        raise RetrievalError("question is required", status_code=400)
    if _answer_llm is None:
        init_answer()

    planner: LLMQueryPlanner | SimpleQueryPlanner
    if request.use_planner:
        planner = get_planner()
    else:
        planner = SimpleQueryPlanner()

    queries = list(await planner.plan(question))
    if not queries:
        raise RetrievalError("Planner returned no queries.", status_code=500)

    top_k = RETRIEVAL_TOP_K
    max_retries = max(0, ANSWER_MAX_RETRIES)
    ensemble_size = max(1, ANSWER_ENSEMBLE_SIZE)
    last_result: RetrievalResult | None = None
    chosen: dict[str, Any] | None = None
    retries = 0

    for attempt in range(max_retries + 1):
        current_k = top_k + attempt * ANSWER_K_DELTA
        async with coordinator.retrieval_slot():
            last_result = await execute_hybrid_search(
                question,
                queries,
                store=get_datastore(),
                embedder=get_embedder(),
                top_k=current_k,
                bm25_top_k=RETRIEVAL_BM25_TOP_K,
            )
        context = _format_context(last_result.snippets)
        payloads: list[dict[str, Any]] = []
        for _ in range(ensemble_size):
            payloads.append(await _generate_once(question, context))
        chosen = _ensemble_vote(payloads, ignore_blank=ANSWER_IGNORE_BLANK)
        if not chosen.get("is_blank"):
            retries = attempt
            break
        retries = attempt
        logger.info("answer abstain retry attempt=%s next_k=%s", attempt, current_k + ANSWER_K_DELTA)

    assert last_result is not None and chosen is not None
    return AnswerResponse(
        question=question,
        answer=str(chosen.get("answer") or ""),
        answer_value=str(chosen.get("answer_value") or ""),
        is_blank=bool(chosen.get("is_blank")),
        ref_ids=[str(x) for x in (chosen.get("ref_ids") or [])],
        explanation=str(chosen.get("explanation") or ""),
        queries=last_result.queries,
        snippets=_snippet_dicts(last_result.snippets),
        retries=retries,
        ensemble_size=ensemble_size,
    )
