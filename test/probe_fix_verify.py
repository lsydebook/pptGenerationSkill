"""Verify chat_template_kwargs fix with full planner prompt."""
import json
import time

import httpx

import src.config.env_loader  # noqa: F401
from src.config.llm_config import PLANNER_API_KEY, PLANNER_BASE_URL, PLANNER_MODEL
from src.param.param_zh import QUERY_PLANNER_SYSTEM_PROMPT, QUERY_PLANNER_USER_TEMPLATE

URL = f"{PLANNER_BASE_URL.rstrip('/')}/chat/completions"
question = "查询一下青蒿素相关知识"
prompt = QUERY_PLANNER_USER_TEMPLATE.format(question=question, max_extra_queries=2)


def call(label: str, body: dict) -> None:
    base = {
        "model": PLANNER_MODEL,
        "messages": [
            {"role": "system", "content": QUERY_PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 256,
    }
    base.update(body)
    t0 = time.perf_counter()
    r = httpx.post(URL, headers={"Authorization": f"Bearer {PLANNER_API_KEY}"}, json=base, timeout=60)
    dt = time.perf_counter() - t0
    data = r.json()
    u = data.get("usage", {})
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    print(
        f"{label}: {dt:.2f}s completion={u.get('completion_tokens')} "
        f"ttft={u.get('time_to_first_token_ms')}ms "
        f"content={(msg.get('content') or '')[:100]!r} reasoning_len={len(msg.get('reasoning_content') or '')}"
    )


if __name__ == "__main__":
    call("extra_body", {"extra_body": {"enable_thinking": False}})
    call("chat_template_kwargs", {"chat_template_kwargs": {"enable_thinking": False}})
