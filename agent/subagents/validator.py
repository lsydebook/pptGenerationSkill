"""ValidatorAgent: validate.

Reads slide_specs → produces warnings (may adjust specs).
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes.validate import validate
from agent.state import AgentState


def build_validator_agent():
    g = StateGraph(AgentState)
    g.add_node("validate", validate)
    g.add_edge(START, "validate")
    g.add_edge("validate", END)
    return g.compile()