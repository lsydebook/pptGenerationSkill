"""LLM-backed query expansion for multi-query retrieval."""

from __future__ import annotations

import json
import re
import time
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.logging_config import get_logger
from src.param.param_zh import QUERY_PLANNER_SYSTEM_PROMPT, QUERY_PLANNER_USER_TEMPLATE

logger = get_logger(__name__)

_THINKING_RE = re.compile(
    r"<\s*(?:think|redacted_reasoning)\s*>.*?<\s*/\s*(?:think|redacted_reasoning)\s*>",
    re.DOTALL | re.IGNORECASE,
)


def _strip_thinking(text: str) -> str:
    return _THINKING_RE.sub("", text).strip()


class LLMQueryPlanner:
    """Expand a user question into multiple retrieval queries."""

    def __init__(
        self,
        llm: BaseChatModel,
        max_queries: int = 3,
        *,
        system_prompt: str = QUERY_PLANNER_SYSTEM_PROMPT,
        user_template: str = QUERY_PLANNER_USER_TEMPLATE,
    ) -> None:
        self._llm = llm
        self._max_queries = max(1, max_queries)
        self._system_prompt = system_prompt
        self._user_template = user_template

    async def plan(self, question: str) -> Sequence[str]:
        question = question.strip()
        base = [question]
        max_extra = max(0, self._max_queries - 1)
        logger.info("query_planner start question_len=%s max_queries=%s", len(question), self._max_queries)

        prompt = self._user_template.format(
            question=question,
            max_extra_queries=max_extra,
        )

        t0 = time.perf_counter()
        response = await self._llm.ainvoke(
            [
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=prompt),
            ]
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = getattr(response, "response_metadata", {}) or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
        logger.info(
            "query_planner llm_done elapsed_ms=%s completion_tokens=%s total_tokens=%s",
            elapsed_ms,
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )
        raw = _strip_thinking(str(response.content or ""))

        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
            items = data.get("queries")
            extra = [str(item).strip() for item in items or [] if str(item).strip()]
        except Exception:
            logger.warning("query_planner json parse failed, fallback to original query only")
            extra = []

        seen = {q.lower() for q in base if q}
        for query in extra:
            key = query.lower()
            if key in seen:
                continue
            base.append(query)
            seen.add(key)
            if len(base) >= self._max_queries:
                break

        if len(base) == 1:
            reformulation = question.split("？", 1)[0].split("?", 1)[0].strip()
            if reformulation and reformulation.lower() not in seen:
                logger.debug("query_planner fallback reformulation=%s", reformulation)
                base.append(reformulation)
        logger.info("query_planner done count=%s queries=%s", len(base), base)
        return base


class SimpleQueryPlanner:
    async def plan(self, question: str) -> Sequence[str]:
        return [question.strip()]
