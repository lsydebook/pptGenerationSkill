import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from schemas import InputBundle, SlideSpec


def test_input_bundle_parses_sample():
    sample = json.loads(
        (Path(__file__).resolve().parent.parent / "examples" / "input_sample.json").read_text(
            encoding="utf-8"
        )
    )
    bundle = InputBundle.model_validate(sample)
    assert len(bundle.sections) == 4
    assert bundle.sections[0].kind == "progress"
    assert bundle.image_by_id("img_results") is not None
    assert bundle.image_by_id("nope") is None


def test_image_asset_requires_path_or_base64():
    with pytest.raises(ValidationError):
        InputBundle.model_validate(
            {"images": [{"id": "x"}]}
        )


def test_slide_spec_defaults_and_keypage():
    s = SlideSpec(layout="content", title="T", page_type="progress")
    assert s.strict is True
    assert s.image_refs == []
    assert SlideSpec.is_key_page("progress") is True
    assert SlideSpec.is_key_page("research") is False
