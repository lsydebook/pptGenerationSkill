from pathlib import Path

from tools.template_preview import render_preview
from tools.template_registry import TemplateRegistry


def test_render_preview_lists_all_templates():
    reg = TemplateRegistry(Path(__file__).resolve().parent.parent / "templates")
    out = render_preview(reg.list())
    assert "模板库" in out
    for t in reg.list():
        assert t.meta.name in out
        assert t.id in out


def test_render_preview_marks_recommendation():
    reg = TemplateRegistry(Path(__file__).resolve().parent.parent / "templates")
    templates = reg.list()
    if not templates:
        return
    rid = templates[0].id
    out = render_preview(
        templates, recommended_id=rid, recommend_reason="测试理由"
    )
    assert "推荐" in out
    assert "测试理由" in out
