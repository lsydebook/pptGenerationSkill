"""spec_build: assemble WeeklySlideSpec from SlideContent + VisualDecision."""
from __future__ import annotations

import logging

from schemas.slide import CalloutSpec, ImageAssetRef, TableSpec, WeeklySlideSpec

from ..state import AgentState

logger = logging.getLogger(__name__)

CALLOUT_LABELS = {
    "key_finding": "核心发现",
    "highlight": "亮点",
    "warning": "注意",
    "summary": "小结",
    "note": "备注",
}


def _infer_callout_kind(page_type: str) -> str:
    if page_type == "results":
        return "key_finding"
    elif page_type == "progress":
        return "highlight"
    elif page_type == "discussion":
        return "warning"
    else:
        return "summary"


def spec_build(state: AgentState) -> dict:
    contents = state.get("slide_contents") or []
    decisions = state.get("visual_decisions") or []

    if not contents:
        return {"slide_specs": []}

    specs: list[WeeklySlideSpec] = []
    for i, content in enumerate(contents):
        vd = decisions[i] if i < len(decisions) else None
        layout = vd.layout if vd else "content"

        callout = None
        if content.callout_text:
            callout = CalloutSpec(
                kind=_infer_callout_kind(content.page_type),
                text=content.callout_text,
            )

        table = None
        if content.table_headers and content.table_rows:
            table = TableSpec(
                headers=content.table_headers,
                rows=content.table_rows,
            )

        spec = WeeklySlideSpec(
            page_index=i + 1,
            layout=layout,
            page_type=content.page_type,
            title=content.title,
            subtitle=content.subtitle,
            bullets=content.bullets,
            body_text=content.body_text,
            chart=content.chart,
            notes=content.notes,
            callout=callout,
            table=table,
            content_style=vd.content_style if vd else "default",
            bullet_count=content.bullet_count(),
            total_chars=content.char_count(),
        )
        specs.append(spec)

    callout_count = sum(1 for s in specs if s.callout)
    table_count = sum(1 for s in specs if s.table)
    logger.info("built %d slide specs (%d callouts, %d tables)", len(specs), callout_count, table_count)
    return {"slide_specs": specs}
