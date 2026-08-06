"""WeeklyDeckOutline: page-by-page deck outline from LLM planner."""

from pydantic import BaseModel, Field

from .slide import PageType


class OutlineItem(BaseModel):
    page_index: int = Field(description="页码，从 1 开始")
    title: str = Field(description="幻灯片标题")
    page_type: PageType = Field(description="页面类型")
    key_points: list[str] = Field(
        default_factory=list, description="该页 3-6 个核心要点"
    )
    source_section_id: str | None = Field(
        default=None, description="对应的上游 InputBundle section id"
    )
    needs_chart: bool = Field(default=False, description="是否需要图表")
    needs_image: bool = Field(default=False, description="是否适合配图")
    image_hint: str | None = Field(default=None, description="配图方向提示")


class WeeklyDeckOutline(BaseModel):
    title: str = Field(description="PPT 标题")
    pages: list[OutlineItem] = Field(description="逐页大纲")
    total_pages: int = Field(description="总页数")
