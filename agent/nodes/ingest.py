"""ingest: validate raw input dict into a typed InputBundle."""
from __future__ import annotations

from schemas.input import InputBundle

from ..state import AgentState


def ingest(state: AgentState) -> dict:
    raw = state.get("raw_input") or {}
    bundle = InputBundle.model_validate(raw)
    return {"parsed": bundle}
