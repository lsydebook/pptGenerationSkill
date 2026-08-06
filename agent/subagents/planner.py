"""PlannerAgent: ingest → brief_gen → outline_plan.

Reads raw_input → produces parsed + brief + outline.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes.brief_gen import brief_gen
from agent.nodes.ingest import ingest
from agent.nodes.outline_plan import outline_plan
from agent.state import AgentState


def build_planner_agent():
    g = StateGraph(AgentState)
    g.add_node("ingest", ingest)
    g.add_node("brief_gen", brief_gen)
    g.add_node("outline_plan", outline_plan)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "brief_gen")
    g.add_edge("brief_gen", "outline_plan")
    g.add_edge("outline_plan", END)
    return g.compile()