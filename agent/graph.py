"""Multi-agent LangGraph pipeline.

5 sub-agents wired in sequence via shared AgentState:
  Planner → Writer → Visual → Validator → Renderer

WriterAgent uses Send API for parallel Map-Reduce per-page generation.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .state import AgentState
from .subagents.planner import build_planner_agent
from .subagents.renderer import build_renderer_agent
from .subagents.validator import build_validator_agent
from .subagents.visual import build_visual_agent
from .subagents.writer import build_writer_agent


def build_graph(checkpointer: Any | None = None):
    g = StateGraph(AgentState)
    g.add_node("planner", build_planner_agent())
    g.add_node("writer", build_writer_agent())
    g.add_node("visual", build_visual_agent())
    g.add_node("validator", build_validator_agent())
    g.add_node("renderer", build_renderer_agent())

    g.add_edge(START, "planner")
    g.add_edge("planner", "writer")
    g.add_edge("writer", "visual")
    g.add_edge("visual", "validator")
    g.add_edge("validator", "renderer")
    g.add_edge("renderer", END)

    return g.compile(checkpointer=checkpointer)