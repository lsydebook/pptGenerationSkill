"""Try various Qwen thinking-disable parameters on BUPT gateway."""
import json
import time

import httpx

import src.config.env_loader  # noqa: F401
from src.config.llm_config import PLANNER_API_KEY, PLANNER_BASE_URL, PLANNER_MODEL

URL = f"{PLANNER_BASE_URL.rstrip('/')}/chat/completions"
MESSAGES = [
    {"role": "user", "content": '只输出JSON: {"queries":["青蒿素","屠呦呦"]}'},
]


def try_call(label: str, body: dict) -> None:
    body = {"model": PLANNER_MODEL, "messages": MESSAGES, "max_tokens": 128, **body}
    t0 = time.perf_counter()
    r = httpx.post(
        URL,
        headers={"Authorization": f"Bearer {PLANNER_API_KEY}"},
        json=body,
        timeout=60.0,
    )
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        print(f"{label}: HTTP {r.status_code} {dt:.1f}s {r.text[:200]}")
        return
    data = r.json()
    u = data.get("usage", {})
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    c = msg.get("content") or ""
    rc = msg.get("reasoning_content") or ""
    print(
        f"{label}: {dt:.1f}s "
        f"completion={u.get('completion_tokens')} "
        f"ttft={u.get('time_to_first_token_ms')}ms "
        f"content={c[:80]!r} reasoning_len={len(rc)}"
    )


if __name__ == "__main__":
    try_call("baseline", {})
    try_call("extra_enable_thinking_false", {"extra_body": {"enable_thinking": False}})
    try_call("top_enable_thinking_false", {"enable_thinking": False})
    try_call("chat_template_kwargs", {"chat_template_kwargs": {"enable_thinking": False}})
    try_call("model_qwen-plus", {"model": "qwen-plus"})
    try_call("model_qwen-turbo", {"model": "qwen-turbo"})
    try_call("model_qwen2.5-7b-instruct", {"model": "qwen2.5-7b-instruct"})
