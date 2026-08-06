"""validate: density and structure checks on WeeklySlideSpec list."""
from __future__ import annotations

import logging

from ..state import AgentState

logger = logging.getLogger(__name__)

MAX_CHARS_PER_SLIDE = 400
MAX_BULLETS_PER_SLIDE = 8
MIN_BULLETS_CONTENT = 1


def validate(state: AgentState) -> dict:
    specs = state.get("slide_specs") or []
    warnings: list[str] = []

    for spec in specs:
        if spec.total_chars > MAX_CHARS_PER_SLIDE:
            warnings.append(
                f"slide {spec.page_index}: 内容过多 ({spec.total_chars} chars), "
                f"建议拆分"
            )
        if spec.bullet_count > MAX_BULLETS_PER_SLIDE:
            warnings.append(
                f"slide {spec.page_index}: 要点过多 ({spec.bullet_count}), "
                f"建议精简"
            )
        if spec.layout == "content" and spec.bullet_count == 0 and not spec.body_text:
            warnings.append(
                f"slide {spec.page_index}: 正文页无内容"
            )

    if warnings:
        logger.warning("validation warnings: %d", len(warnings))
        for w in warnings:
            logger.warning("  %s", w)

    return {"warnings": warnings, "slide_specs": specs}
