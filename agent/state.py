"""LangGraph state for multi-agent pipeline.

Shared state across 5 sub-agents: Planner → Writer → Visual → Validator → Renderer.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from schemas.brief import WeeklyReportBrief
from schemas.content import SlideContent
from schemas.input import InputBundle
from schemas.outline import WeeklyDeckOutline
from schemas.slide import WeeklySlideSpec
from schemas.visual import VisualDecision


class AgentState(TypedDict, total=False):
    # ===== 配置字段 =====
    raw_input: dict[str, Any]
    base_dir: str
    output_path: str
    mock: bool
    accent_color: str
    enable_ai_image: bool

    # ===== 数据流字段 =====
    parsed: InputBundle | None
    brief: WeeklyReportBrief | None
    outline: WeeklyDeckOutline | None

    slide_contents: list[SlideContent]
    visual_decisions: list[VisualDecision]
    slide_specs: list[WeeklySlideSpec]

    rendered_path: str | None
    warnings: list[str]

    # ===== WriterAgent 并行 Map-Reduce 瞬态字段 =====
    # 当前 worker 要生成的页码（dispatcher → worker Send payload）
    write_page_index: int
    # 并行 worker 各自产出 (page_index, content) 包，operator.add reducer 自动拼接
    write_items: Annotated[list[dict], operator.add]
    # supervisor 路由标记：当前已完成到哪个子 Agent
    agent_step: str