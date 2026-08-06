"""slide_write: produce list[SlideContent] from Outline + InputBundle via LLM (batch)."""
from __future__ import annotations

import json
import logging

from tools.llm import get_client

from ..state import AgentState

logger = logging.getLogger(__name__)

WRITE_SYSTEM = """你是一个学术 PPT 内容撰写助手。请为每页大纲撰写完整的幻灯片内容。
输出严格 JSON 数组，每个元素对应一页。"""


def _prompt(brief_json: str, outline_json: str, bundle_json: str) -> str:
    return f"""请为以下 PPT 大纲的每一页撰写完整内容。输出 JSON **数组**，每个元素格式：

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

如果页面需要图表（大纲中 needs_chart=true），chart 字段格式：
{{"kind": "bar|line|pie", "categories": [...], "series": [{{"name": "系列名", "values": [...]}}], "y_axis_title": "Y轴标题"}}

如果本页适合放表格（多方法/多指标对比等），table 格式：
"table_headers": ["列1", "列2", "列3"],
"table_rows": [["值1", "值2", "值3"], ["值4", "值5", "值6"]]

规则：
- 标题简洁有力，10-15字
- 要点每页3-6条，每条不超过30字
- 图表页的标题和要点围绕图表数据展开
- results页尽量给出 callout_text 总结核心发现
- progress页可给 callout_text 标注本周突破
- discussion页可给 callout_text 标注关键问题
- 当有3列以上的多维对比数据（如原始JSON中有多个方法性能对比）时，考虑用 table
- 中文学术风格

简报：
{brief_json}

大纲：
{outline_json}

原始数据（含图表数据）：
{bundle_json}
"""


def slide_write(state: AgentState) -> dict:
    brief = state.get("brief")
    outline = state.get("outline")
    parsed = state.get("parsed")
    if brief is None or outline is None or parsed is None:
        return {"slide_contents": []}

    brief_json = brief.model_dump_json(exclude_none=True)
    outline_json = outline.model_dump_json(exclude_none=True)
    bundle_json = parsed.model_dump_json(exclude_none=True)

    logger.info("writing %d slides via LLM (batch)...", outline.total_pages)
    client = get_client()
    data = client.text.chat_json(
        _prompt(brief_json, outline_json, bundle_json),
        system=WRITE_SYSTEM,
        temperature=0.4,
        max_tokens=4000,
    )
    if not isinstance(data, list):
        data = [data]

    from schemas.content import SlideContent
    contents = [SlideContent(**item) for item in data]

    outline_pages = outline.pages
    for i, c in enumerate(contents):
        if i < len(outline_pages):
            oi = outline_pages[i]
            if not c.image_hint and oi.image_hint:
                c.image_hint = oi.image_hint

    logger.info("wrote %d slide contents", len(contents))
    return {"slide_contents": contents}
