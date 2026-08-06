"""outline_plan: produce WeeklyDeckOutline from Brief + InputBundle via LLM."""
from __future__ import annotations

import json
import logging

from tools.llm import get_client

from ..state import AgentState

logger = logging.getLogger(__name__)

OUTLINE_SYSTEM = """你是一个学术 PPT 大纲规划助手。请根据简报和原始数据，规划出逐页大纲。目标 8-12 页。"""


def _prompt(brief_json: str, bundle_json: str) -> str:
    return f"""请为以下周报规划 PPT 逐页大纲。**目标 8-12 页**，内容丰富度优先。输出 JSON：

{{
  "title": "PPT 标题",
  "pages": [
    {{
      "page_index": 1,
      "title": "页面标题",
      "page_type": "cover|toc|section|progress|research|results|discussion|plan|thanks",
      "key_points": ["要点1", "要点2", "要点3"],
      "source_section_id": "对应输入 section 的 id，无则为 null",
      "needs_chart": false,
      "needs_image": false,
      "image_hint": "如果需要配图，给出英文配图描述；否则 null"
    }}
  ],
  "total_pages": 页数
}}

规则：
- 第1页 cover
- 第2页 toc（如果超过3个 section），目录列出所有章节名
- **重要：目标 8-12 页。内容多的 section 拆成 2-4 页**，不要吝啬页数
- 每个 section 标题页用 section 类型，内容页用 progress/research/results/discussion 类型
- 末页 thanks
- 如果 section 包含图表数据，设 needs_chart=true
- 如果有研究模型/架构/方法相关内容，设 needs_image=true 并给出英文 image_hint
- 如果没有图表数据但有实验结果/研究方向/模型架构等抽象内容，也设 needs_image=true 给配图提示

简报：
{brief_json}

原始数据：
{bundle_json}
"""


def outline_plan(state: AgentState) -> dict:
    brief = state.get("brief")
    parsed = state.get("parsed")
    if brief is None or parsed is None:
        return {"outline": None}

    brief_json = brief.model_dump_json(exclude_none=True)
    bundle_json = parsed.model_dump_json(exclude_none=True)

    logger.info("planning outline via LLM...")
    client = get_client()
    data = client.text.chat_json(_prompt(brief_json, bundle_json), system=OUTLINE_SYSTEM)
    from schemas.outline import WeeklyDeckOutline
    outline = WeeklyDeckOutline(**data)
    logger.info("outline: %d pages", outline.total_pages)
    return {"outline": outline}
