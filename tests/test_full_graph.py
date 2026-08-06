"""End-to-end smoke test for `build_full_graph` in mock mode.

Mock mode skips every LLM call (plan / write_strict / write_free / polish /
review), so the graph runs deterministically and we can assert on the final
state without needing an API key.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.graph import build_full_graph
from tools.image_utils import ensure_sample_images


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_full_graph_end_to_end_mock(project_root: Path, tmp_path: Path) -> None:
    ensure_sample_images(project_root)
    data = json.loads(
        (project_root / "examples" / "input_sample.json").read_text(encoding="utf-8")
    )
    out = tmp_path / "weekly_full.pptx"

    graph = build_full_graph()
    final = graph.invoke(
        {
            "raw_input": data,
            "template_path": None,
            "output_path": str(out),
            "base_dir": str(project_root),
            "mock": True,
            "templates_root": str(project_root / "templates"),
            "selected_template_id": "weekly_minimal_blue",
        }
    )

    rendered = final.get("rendered_path")
    assert rendered, "full graph must produce a rendered_path"
    assert out.exists() and out.stat().st_size > 5000

    outline = final.get("outline") or []
    drafted = final.get("drafted") or []
    assert len(outline) >= 4
    assert len(drafted) == len(outline), (
        "drafted should keep one entry per outline slide after batched write"
    )

    # In mock mode review must not request a revise (no LLM call happens).
    assert final.get("needs_revise") is False
