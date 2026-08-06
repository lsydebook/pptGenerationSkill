"""VisualDecision: layout and image strategy for a single slide."""

from pydantic import BaseModel, Field

from .slide import ContentStyle, LayoutName


class VisualDecision(BaseModel):
    layout: LayoutName = Field(description="选定版式")
    content_style: ContentStyle = Field(default="default", description="内容排版风格: cards/numbered/separated/grid/default")
    image_strategy: str = Field(
        default="none",
        description="图片策略: none / ai_generate / use_existing / chart_only",
    )
    image_prompt: str | None = Field(
        default=None, description="AI 生图的 prompt（仅 ai_generate 策略）"
    )
    existing_image_id: str | None = Field(
        default=None, description="已有图片 id（仅 use_existing 策略）"
    )
    color_accent: str | None = Field(
        default=None, description="配色强调色 hex"
    )
    reasoning: str = Field(default="", description="视觉决策理由")
