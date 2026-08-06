"""assets: generate chart images and AI images, attach to WeeklySlideSpec."""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from tools.llm import get_client

from ..state import AgentState

logger = logging.getLogger(__name__)


def _render_chart(spec, output_dir: str) -> str | None:
    """Render chart spec to PNG via matplotlib. Returns path or None."""
    if spec.chart is None:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
    except ImportError:
        logger.warning("matplotlib not available, skip chart")
        return None

    chart = spec.chart
    fig, ax = plt.subplots(figsize=(8, 4.5))

    try:
        fp = fm.FontProperties(family="Microsoft YaHei", size=11)
    except Exception:
        fp = None

    x = range(len(chart.categories))
    width = 0.8 / max(len(chart.series), 1)

    for si, ser in enumerate(chart.series):
        values = ser.get("values", [])
        name = ser.get("name", f"Series {si}")
        if chart.kind == "bar":
            ax.bar([i + si * width for i in x], values, width, label=name)
        elif chart.kind == "line":
            ax.plot(x, values, marker="o", label=name)
        elif chart.kind == "pie":
            ax.pie(values, labels=chart.categories, autopct="%1.1f%%")
            ax.set_title(chart.y_axis_title or "", fontproperties=fp)
            break

    if chart.kind != "pie":
        ax.set_xticks([i + width * (len(chart.series) - 1) / 2 for i in x])
        ax.set_xticklabels(chart.categories, fontproperties=fp)
        if chart.y_axis_title:
            ax.set_ylabel(chart.y_axis_title, fontproperties=fp)
        ax.legend(prop=fp)

    ax.set_title(spec.title, fontproperties=fp, fontsize=14)

    out_dir = Path(output_dir) / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"chart_{spec.page_index:02d}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    logger.info("chart saved: %s", path)
    return str(path)


def _generate_ai_image(spec, output_dir: str, decision) -> str | None:
    """Generate AI image from prompt. Returns local path or None."""
    if decision is None:
        return None
    prompt = decision.image_prompt
    if not prompt:
        return None

    logger.info("generating AI image for slide %d: %s...", spec.page_index, prompt[:40])
    client = get_client()
    try:
        out_dir = Path(output_dir) / "ai_images"
        gen = client.image_gen.generate_and_download(
            prompt=prompt,
            output_dir=str(out_dir),
            size="1920*1080",
            filename=f"ai_{spec.page_index:02d}.png",
        )
        return gen.local_path
    except Exception as e:
        logger.error("AI image generation failed: %s", e)
        return None


def assets(state: AgentState) -> dict:
    specs = state.get("slide_specs") or []
    decisions = state.get("visual_decisions") or []
    output_path = state.get("output_path", "out/weekly.pptx")
    output_dir = str(Path(output_path).parent)
    enable_ai = state.get("enable_ai_image", False)

    for i, spec in enumerate(specs):
        decision = decisions[i] if i < len(decisions) else None

        if spec.chart is not None:
            chart_path = _render_chart(spec, output_dir)
            if chart_path:
                from schemas.slide import ImageAssetRef
                spec.images.append(ImageAssetRef(
                    path=chart_path, source="chart",
                    caption=spec.chart.y_axis_title or spec.title,
                ))

        if decision and decision.image_strategy == "use_existing":
            cached = Path(output_dir) / "ai_images" / f"ai_{spec.page_index:02d}.png"
            if cached.exists():
                from schemas.slide import ImageAssetRef
                spec.images.append(ImageAssetRef(
                    path=str(cached), source="existing",
                    caption=spec.title,
                ))
                logger.info("using cached image for slide %d: %s", spec.page_index, cached)

        if enable_ai and decision and decision.image_strategy == "ai_generate":
            cached = Path(output_dir) / "ai_images" / f"ai_{spec.page_index:02d}.png"
            if cached.exists():
                from schemas.slide import ImageAssetRef
                spec.images.append(ImageAssetRef(
                    path=str(cached), source="existing",
                    caption=spec.title,
                ))
                logger.info("using cached image for slide %d: %s", spec.page_index, cached)
            else:
                img_path = _generate_ai_image(spec, output_dir, decision)
                if img_path:
                    from schemas.slide import ImageAssetRef
                    spec.images.append(ImageAssetRef(
                        path=img_path, source="ai_generated",
                        prompt=decision.image_prompt,
                        caption=spec.title,
                    ))

    return {"slide_specs": specs}
