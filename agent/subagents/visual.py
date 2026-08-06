"""VisualAgent: visual_plan → spec_build.

Reads slide_contents + accent_color + enable_ai_image → produces visual_decisions + slide_specs.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes.spec_build import spec_build
from agent.nodes.visual_plan import visual_plan
from agent.state import AgentState


def build_visual_agent():
    g = StateGraph(AgentState)
    g.add_node("visual_plan", visual_plan)
    g.add_node("spec_build", spec_build)
    g.add_edge(START, "visual_plan")
    g.add_edge("visual_plan", "spec_build")
    g.add_edge("spec_build", END)
    return g.compile()