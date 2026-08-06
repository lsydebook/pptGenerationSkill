from pathlib import Path

from tools.template_registry import TemplateRegistry


def test_registry_finds_templates(tmp_path: Path):
    # Use the project's real templates dir
    reg = TemplateRegistry(Path(__file__).resolve().parent.parent / "templates")
    ids = reg.ids()
    assert "weekly_default" in ids
    # starter templates should exist after `python -m tools.starter_templates --all`
    assert any(i.startswith("weekly_") and i != "weekly_default" for i in ids), ids


def test_registry_summary_shape():
    reg = TemplateRegistry(Path(__file__).resolve().parent.parent / "templates")
    for t in reg.list():
        s = t.summary_for_llm()
        assert {"id", "name", "style", "audience", "theme_color"} <= set(s)
