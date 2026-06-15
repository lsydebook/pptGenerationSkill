"""Verify LangChain passes chat_template_kwargs correctly."""
import asyncio
import time

import src.config.env_loader  # noqa: F401
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config.llm_config import PLANNER_API_KEY, PLANNER_BASE_URL, PLANNER_MODEL
from src.param.param_zh import QUERY_PLANNER_SYSTEM_PROMPT, QUERY_PLANNER_USER_TEMPLATE


async def try_kwargs(label: str, llm_kwargs: dict) -> None:
    llm = ChatOpenAI(
        model=PLANNER_MODEL,
        api_key=PLANNER_API_KEY,
        base_url=PLANNER_BASE_URL,
        max_tokens=256,
        **llm_kwargs,
    )
    question = "查询一下青蒿素相关知识"
    prompt = QUERY_PLANNER_USER_TEMPLATE.format(question=question, max_extra_queries=2)
    messages = [SystemMessage(content=QUERY_PLANNER_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    t0 = time.perf_counter()
    r = await llm.ainvoke(messages)
    dt = time.perf_counter() - t0
    meta = getattr(r, "response_metadata", {}) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    print(f"{label}: {dt:.2f}s usage={usage} content={(str(r.content or ''))[:80]!r}")


async def main() -> None:
    await try_kwargs("extra_body", {"extra_body": {"enable_thinking": False}})
    await try_kwargs(
        "model_kwargs_chat_template",
        {"model_kwargs": {"chat_template_kwargs": {"enable_thinking": False}}},
    )


if __name__ == "__main__":
    asyncio.run(main())
