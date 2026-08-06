import json
from pathlib import Path

import pytest

from agent.graph import build_min_graph
from tools.image_utils import ensure_sample_images


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_min_graph_end_to_end(project_root: Path, tmp_path: Path) -> None:
    ensure_sample_images(project_root)
    data = json.loads(
        (project_root / "examples" / "input_sample.json").read_text(encoding="utf-8")
    )
    out = tmp_path / "weekly.pptx"
    graph = build_min_graph()
    final = graph.invoke(
        {
            "raw_input": data,
            "template_path": None,
            "output_path": str(out),
            "base_dir": str(project_root),
            "mock": True,
        }
    )
    assert final.get("rendered_path")
    assert out.exists() and out.stat().st_size > 5000
    outline = final.get("outline") or []
    drafted = final.get("drafted") or []
    assert len(outline) >= 4
    assert len(drafted) == len(outline)
