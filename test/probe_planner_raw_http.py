"""Raw HTTP probe for qwen-latest thinking tokens."""
import json
import time

import httpx

import src.config.env_loader  # noqa: F401
from src.config.llm_config import (
    PLANNER_API_KEY,
    PLANNER_BASE_URL,
    PLANNER_MODEL,
)
from src.param.param_zh import QUERY_PLANNER_SYSTEM_PROMPT, QUERY_PLANNER_USER_TEMPLATE

URL = f"{PLANNER_BASE_URL.rstrip('/')}/chat/completions"
question = "查询一下青蒿素相关知识"
prompt = QUERY_PLANNER_USER_TEMPLATE.format(question=question, max_extra_queries=2)

payloads = [
    ("enable_thinking_false", {"enable_thinking": False}),
    ("no_extra_body", None),
    ("enable_thinking_true", {"enable_thinking": True}),
]


def call(label: str, extra_body: dict | None) -> None:
    body = {
        "model": PLANNER_MODEL,
        "messages": [
            {"role": "system", "content": QUERY_PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    if extra_body is not None:
        body["extra_body"] = extra_body
        # also try top-level for some gateways
        body.update(extra_body)

    t0 = time.perf_counter()
    r = httpx.post(
        URL,
        headers={"Authorization": f"Bearer {PLANNER_API_KEY}"},
        json=body,
        timeout=120.0,
    )
    dt = time.perf_counter() - t0
    print(f"\n=== {label} status={r.status_code} elapsed={dt:.2f}s ===")
    if r.status_code != 200:
        print(r.text[:500])
        return
    data = r.json()
    usage = data.get("usage", {})
    print("usage:", json.dumps(usage, ensure_ascii=False))
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    print(f"content_len={len(content)} reasoning_len={len(reasoning)}")
    print("content:", repr(content[:200]))
    if reasoning:
        print("reasoning preview:", repr(reasoning[:300]))


if __name__ == "__main__":
    for label, extra in payloads:
        call(label, extra)
