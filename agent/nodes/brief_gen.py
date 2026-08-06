"""brief_gen: produce WeeklyReportBrief from InputBundle via LLM."""
from __future__ import annotations

import json
import logging

from tools.llm import get_client

from ..state import AgentState

logger = logging.getLogger(__name__)

BRIEF_SYSTEM = """你是一个学术周报编辑助手。请阅读输入数据，提炼出简报。目标 PPT 为 8-12 页。"""


def _prompt(bundle_json: str) -> str:
    return f"""请将以下周报数据提炼成简报。输出 JSON：

{{
  "theme": "本周核心主题，一句话",
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "audience": "目标受众（导师/课题组/管理层）",
  "tone": "整体风格（学术/商务/激励）",
  "estimated_pages": 预估页数(数字, 建议 9-11),
  "section_titles": ["章节1标题", "章节2标题", ...],
  "summary": "一句话概括本周内容"
}}

周报数据：
{bundle_json}
"""


def brief_gen(state: AgentState) -> dict:
    parsed = state.get("parsed")
    if parsed is None:
        return {"brief": None}
    try:
        bundle_json = parsed.model_dump_json(exclude_none=True)
    except Exception:
        bundle_json = json.dumps(state.get("raw_input", {}), ensure_ascii=False)

    logger.info("generating brief via LLM...")
    client = get_client()
    data = client.text.chat_json(_prompt(bundle_json), system=BRIEF_SYSTEM)
    from schemas.brief import WeeklyReportBrief
    brief = WeeklyReportBrief(**data)
    logger.info("brief: theme=%s, pages=%d", brief.theme, brief.estimated_pages)
    return {"brief": brief}
