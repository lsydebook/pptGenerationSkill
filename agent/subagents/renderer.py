"""RendererAgent: assets → render.

Reads slide_specs + visual_decisions + output_path + enable_ai_image → produces rendered_path.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes.assets import assets
from agent.nodes.render import render
from agent.state import AgentState


def build_renderer_agent():
    g = StateGraph(AgentState)
    g.add_node("assets", assets)
    g.add_node("render", render)
    g.add_edge(START, "assets")
    g.add_edge("assets", "render")
    g.add_edge("render", END)
    return g.compile()