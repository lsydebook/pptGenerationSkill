"""WriterAgent: slide_write dispatch → N × worker (parallel) → reduce.

Uses LangGraph Send API for Map-Reduce: dispatcher fans out one Send per page,
each worker calls LLM independently, reducer merges + sorts + backfills image_hint.

Reuses WRITE_SYSTEM + LLM params from agent/nodes/slide_write.py.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite, Send

from schemas.content import SlideContent
from tools.llm import get_client

from agent.nodes.slide_write import WRITE_SYSTEM
from agent.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# dispatcher: fan-out one Send per page
# ---------------------------------------------------------------------------

def slide_write_dispatch(state: AgentState):
    """Read outline → emit one Send per page to parallel worker."""
    outline = state.get("outline")
    if outline is None or not outline.pages:
        return [Send("slide_write_reduce", {})]

    brief = state.get("brief")
    parsed = state.get("parsed")
    mock = state.get("mock", False)

    return [
        Send("slide_write_worker", {
            "brief": brief,
            "outline": outline,
            "parsed": parsed,
            "write_page_index": i,
            "mock": mock,
        })
        for i in range(len(outline.pages))
    ]


# ---------------------------------------------------------------------------
# worker: per-page LLM call (reuses original params)
# ---------------------------------------------------------------------------

def _per_page_prompt(brief_json: str, outline_json: str, bundle_json: str, page_index: int) -> str:
    """Single-page prompt with full outline context (for cross-page dedup hints)."""
    total = 1  # placeholder, will be shown in log
    return f"""请为以下 PPT 大纲的**第 {page_index + 1} 页**撰写完整内容。
**只输出该页对应的单个 JSON 对象（不要输出数组，不要输出其他页）**。

完整大纲（仅供参考上下文，请勿生成大纲中其他页的内容，
且不要与其它页的标题/要点重复）：

{outline_json}

对象格式：
{{
  "page_type": "页面类型",
  "title": "幻灯片标题（10-15字，中文学术风格）",
  "subtitle": "副标题（可选，null表示无）",
  "bullets": ["要点1", "要点2", ...],
  "body_text": "正文段落（可选，null表示无）",
  "chart": null,
  "notes": "演讲者备注（可选，null表示无）",
  "source_section_id": "对应 section id",
  "image_hint": "如果大纲中有配图提示，在此给出 3-5 个英文关键词描述配图主题，无则 null",
  "callout_text": "如果本页适合加一个醒目的结论/亮点/注意框，写一句话（20-40字）；不需要则 null",
  "table_headers": null,
  "table_rows": null
}}

如果该页需要图表（大纲中 needs_chart=true），chart 字段格式：
{{"kind": "bar|line|pie", "categories": [...], "series": [{{"name": "系列名", "values": [...]}}], "y_axis_title": "Y轴标题"}}

规则：
- 标题简洁有力，10-15字
- 要点3-6条，每条不超过30字
- 图表页的标题和要点围绕图表数据展开
- results页尽量给出 callout_text 总结核心发现
- 中文学术风格

简报：
{brief_json}

原始数据（含图表数据）：
{bundle_json}
"""


def slide_write_worker(state: AgentState) -> dict:
    """Single page LLM call. Returns write_items for operator.add reducer."""
    page_index = state["write_page_index"]
    brief = state.get("brief")
    outline = state.get("outline")
    parsed = state.get("parsed")

    if brief is None or outline is None or parsed is None:
        return {"write_items": []}

    brief_json = brief.model_dump_json(exclude_none=True)
    outline_json = outline.model_dump_json(exclude_none=True)
    bundle_json = parsed.model_dump_json(exclude_none=True)

    logger.info("writing slide %d/%d via LLM (parallel)...",
                page_index + 1, outline.total_pages)

    client = get_client()
    data = client.text.chat_json(
        _per_page_prompt(brief_json, outline_json, bundle_json, page_index),
        system=WRITE_SYSTEM,
        temperature=0.4,
        max_tokens=4000,
    )
    if isinstance(data, list):
        data = data[0] if data else {}

    content = SlideContent(**data)
    return {"write_items": [{"page_index": page_index, "content": content}]}


# ---------------------------------------------------------------------------
# reducer: merge, sort, backfill image_hint
# ---------------------------------------------------------------------------

def slide_write_reduce(state: AgentState) -> dict:
    """Merge all worker outputs, sort by page_index, backfill image_hint.

    Uses Overwrite to bypass any future reducer on slide_contents.
    """
    items = state.get("write_items") or []
    outline = state.get("outline")

    items_sorted = sorted(items, key=lambda x: x["page_index"])
    contents: list[SlideContent] = [it["content"] for it in items_sorted]

    # Replicate original slide_write.py backfill logic
    if outline is not None:
        outline_pages = outline.pages
        for i, c in enumerate(contents):
            if i < len(outline_pages):
                oi = outline_pages[i]
                if not c.image_hint and oi.image_hint:
                    c.image_hint = oi.image_hint

    logger.info("reduced %d slide contents (parallel)", len(contents))
    return {"slide_contents": Overwrite(contents)}


# ---------------------------------------------------------------------------
# subgraph builder
# ---------------------------------------------------------------------------

def build_writer_agent():
    g = StateGraph(AgentState)
    g.add_node("slide_write_worker", slide_write_worker)
    g.add_node("slide_write_reduce", slide_write_reduce)

    # START → dispatch → N parallel workers → reduce
    g.add_conditional_edges(
        START,
        slide_write_dispatch,
        ["slide_write_worker", "slide_write_reduce"],
    )
    g.add_edge("slide_write_worker", "slide_write_reduce")
    g.add_edge("slide_write_reduce", END)
    return g.compile()