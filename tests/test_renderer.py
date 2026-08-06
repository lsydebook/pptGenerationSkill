import json
from pathlib import Path

import pytest

from schemas import InputBundle, SlideSpec
from tools.image_utils import ensure_sample_images
from tools.pptx_renderer import render_to_pptx


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def bundle(project_root: Path) -> InputBundle:
    data = json.loads(
        (project_root / "examples" / "input_sample.json").read_text(encoding="utf-8")
    )
    return InputBundle.model_validate(data)


def test_renderer_produces_nonempty_pptx(project_root: Path, bundle: InputBundle, tmp_path: Path):
    ensure_sample_images(project_root)
    specs = [
        SlideSpec(layout="cover", title="T", page_type="cover"),
        SlideSpec(
            layout="content",
            title="本周进展",
            bullets=["A", "B", "C"],
            page_type="progress",
        ),
        SlideSpec(
            layout="image",
            title="实验",
            image_refs=["img_results"],
            page_type="results",
        ),
        SlideSpec(layout="thanks", title="Thanks", page_type="thanks"),
    ]
    out = tmp_path / "x.pptx"
    rendered = render_to_pptx(
        specs,
        out,
        template_path=None,
        bundle=bundle,
        base_dir=project_root,
    )
    assert Path(rendered).exists()
    assert Path(rendered).stat().st_size > 5000


def test_renderer_handles_missing_image(tmp_path: Path):
    bundle = InputBundle()
    specs = [
        SlideSpec(layout="image", title="X", image_refs=["nope"], page_type="research"),
    ]
    out = tmp_path / "y.pptx"
    rendered = render_to_pptx(
        specs,
        out,
        template_path=None,
        bundle=bundle,
        base_dir=tmp_path,
    )
    assert Path(rendered).exists()
