from pathlib import Path


from schemas import ChartSpec, InputBundle, SlideSpec
from tools.image_utils import ensure_sample_images
from tools.polish import find_highlight_spans
from tools.pptx_renderer import render_to_pptx


def test_highlight_finds_numbers_and_keywords():
    spans = find_highlight_spans("提升 +3.2 F1，相对 baseline 5x faster, SOTA 维持")
    text = "提升 +3.2 F1，相对 baseline 5x faster, SOTA 维持"
    matched = [text[s:e] for s, e in spans]
    assert any("3.2" in m for m in matched)
    assert "baseline" in matched
    assert "SOTA" in matched


def test_chart_layout_renders(tmp_path: Path):
    bundle = InputBundle()
    spec = SlideSpec(
        layout="chart",
        title="对比",
        chart=ChartSpec(
            kind="bar",
            categories=["E5", "E10"],
            series=[
                {"name": "baseline", "values": [72.0, 76.4]},
                {"name": "ours", "values": [74.5, 79.6]},
            ],
        ),
        page_type="results",
    )
    out = tmp_path / "chart.pptx"
    rendered = render_to_pptx(
        [spec], out, template_path=None, bundle=bundle, base_dir=tmp_path
    )
    assert Path(rendered).exists()
    assert Path(rendered).stat().st_size > 5000


def test_full_render_with_themed_template(tmp_path: Path):
    project_root = Path(__file__).resolve().parent.parent
    ensure_sample_images(project_root)
    bundle = InputBundle()
    specs = [
        SlideSpec(layout="cover", title="T", page_type="cover"),
        SlideSpec(
            layout="content",
            title="本周进展",
            bullets=["完成 +12k 标注", "在 baseline 上达 78.2 F1"],
            page_type="progress",
        ),
        SlideSpec(layout="thanks", title="Thanks", page_type="thanks"),
    ]
    template_path = project_root / "templates" / "weekly_minimal_blue" / "template.pptx"
    out = tmp_path / "themed.pptx"
    rendered = render_to_pptx(
        specs,
        out,
        template_path=str(template_path) if template_path.exists() else None,
        bundle=bundle,
        base_dir=project_root,
        theme_color="#3366CC",
        accent_color="#FF8C00",
    )
    assert Path(rendered).exists()
