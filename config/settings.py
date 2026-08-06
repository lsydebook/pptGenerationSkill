"""Runtime configuration loaded from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    dashscope_api_key: str | None
    dashscope_base_url: str
    image_gen_base_url: str
    text_model: str
    vision_model: str
    image_gen_model: str
    output_dir: Path
    project_root: Path

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            dashscope_base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            image_gen_base_url="https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            text_model=os.getenv("TEXT_MODEL", "qwen3.6-plus"),
            vision_model=os.getenv("VISION_MODEL", "qwen3-vl-plus"),
            image_gen_model=os.getenv("IMAGE_GEN_MODEL", "qwen-image-2.0"),
            output_dir=Path(os.getenv("WEEKLY_OUTPUT_DIR", "out")),
            project_root=PROJECT_ROOT,
        )


settings = Settings.load()
