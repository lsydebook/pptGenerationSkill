"""Probe planner LLM latency."""
import asyncio
import time

import src.config.env_loader  # noqa: F401
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config.llm_config import (
    PLANNER_API_KEY,
    PLANNER_BASE_URL,
    PLANNER_ENABLE_THINKING,
    PLANNER_MAX_TOKENS,
    PLANNER_MODEL,
    PLANNER_TEMPERATURE,
)
from src.param.param_zh import QUERY_PLANNER_SYSTEM_PROMPT, QUERY_PLANNER_USER_TEMPLATE


async def timed_call(label: str, llm: ChatOpenAI, messages) -> None:
    t0 = time.perf_counter()
    response = await llm.ainvoke(messages)
    dt = time.perf_counter() - t0
    content = str(response.content or "")
    preview = content[:120].replace("\n", " ")
    print(f"[{label}] {dt:.2f}s len={len(content)} preview={preview!r}")
    meta = getattr(response, "response_metadata", {}) or {}
    usage = meta.get("token_usage") or meta.get("usage")
    if usage:
        print(f"  token_usage={usage}")


async def main() -> None:
    kwargs: dict = {
        "model": PLANNER_MODEL,
        "api_key": PLANNER_API_KEY,
        "base_url": PLANNER_BASE_URL,
    }
    if PLANNER_TEMPERATURE:
        kwargs["temperature"] = float(PLANNER_TEMPERATURE)
    if PLANNER_MAX_TOKENS:
        kwargs["max_tokens"] = int(PLANNER_MAX_TOKENS)
    if not PLANNER_ENABLE_THINKING:
        kwargs["extra_body"] = {"enable_thinking": False}

    print(
        "model=", PLANNER_MODEL,
        "base=", PLANNER_BASE_URL,
        "thinking_off=", not PLANNER_ENABLE_THINKING,
    )
    llm = ChatOpenAI(**kwargs)
    question = "查询一下青蒿素相关知识"
    prompt = QUERY_PLANNER_USER_TEMPLATE.format(question=question, max_extra_queries=2)
    messages = [
        SystemMessage(content=QUERY_PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    await timed_call("full_planner_prompt", llm, messages)

    mini = [
        HumanMessage(
            content=(
                '只输出JSON: {"queries":["青蒿素药理作用","青蒿素发现历史"]} '
                f"对应问题: {question}"
            )
        )
    ]
    await timed_call("minimal_prompt", llm, mini)


if __name__ == "__main__":
    asyncio.run(main())
