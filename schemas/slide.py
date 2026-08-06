"""WeeklySlideSpec: the final IR consumed by LayoutEngine and Renderer."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

LayoutName = Literal[
    "cover",
    "toc",
    "section",
    "content",
    "dual_col",
    "two_col",
    "image",
    "image_only",
    "chart",
    "thanks",
]

PageType = Literal[
    "cover",
    "toc",
    "section",
    "progress",
    "research",
    "results",
    "discussion",
    "plan",
    "thanks",
]

KEY_PAGE_TYPES: set[PageType] = {"cover", "progress", "results", "plan", "thanks"}

ChartKind = Literal["bar", "line", "pie"]


class ChartSpec(BaseModel):
    kind: ChartKind = "bar"
    categories: list[str] = Field(default_factory=list)
    series: list[dict[str, Any]] = Field(default_factory=list)
    y_axis_title: str | None = None


class ImageAssetRef(BaseModel):
    path: str = Field(description="本地图片文件路径")
    source: Literal["ai_generated", "existing", "chart"] = "existing"
    prompt: str | None = Field(default=None, description="AI 生图 prompt")
    caption: str | None = Field(default=None, description="图片说明")


CalloutType = Literal["key_finding", "highlight", "warning", "summary", "note"]
ContentStyle = Literal["cards", "numbered", "separated", "grid", "default"]


class CalloutSpec(BaseModel):
    kind: CalloutType = "highlight"
    text: str = Field(description="callout 文字内容")


class TableSpec(BaseModel):
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    caption: str | None = None


class WeeklySlideSpec(BaseModel):
    page_index: int = Field(description="页码，从 1 开始")
    layout: LayoutName = Field(description="版式名")
    page_type: PageType = Field(description="页面类型")
    title: str = Field(description="幻灯片标题")
    subtitle: str | None = None
    bullets: list[str] = Field(default_factory=list)
    body_text: str | None = None
    chart: ChartSpec | None = None
    images: list[ImageAssetRef] = Field(default_factory=list)
    callout: CalloutSpec | None = None
    table: TableSpec | None = None
    notes: str | None = None
    color_accent: str | None = None
    content_style: ContentStyle = "default"
    bullet_count: int = 0
    total_chars: int = 0
