"""SlideContent: full page content produced by LLM for a single slide."""

from pydantic import BaseModel, Field

from .slide import ChartSpec, PageType


class SlideContent(BaseModel):
    page_type: PageType = Field(description="页面类型")
    title: str = Field(description="幻灯片标题")
    subtitle: str | None = Field(default=None, description="副标题")
    bullets: list[str] = Field(default_factory=list, description="要点列表")
    body_text: str | None = Field(default=None, description="正文段落")
    chart: ChartSpec | None = Field(default=None, description="内置图表数据")
    notes: str | None = Field(default=None, description="演讲者备注")
    source_section_id: str | None = Field(default=None)
    image_hint: str | None = Field(default=None, description="配图方向提示（英文关键词）")
    callout_text: str | None = Field(default=None, description="亮点/结论 callout 文字")
    table_headers: list[str] | None = Field(default=None, description="表格列头")
    table_rows: list[list[str]] | None = Field(default=None, description="表格数据行")

    def char_count(self) -> int:
        n = len(self.title) + len(self.subtitle or "")
        n += sum(len(b) for b in self.bullets)
        n += len(self.body_text or "")
        return n

    def bullet_count(self) -> int:
        return len(self.bullets)

    def content_summary(self) -> str:
        parts = [f"Title: {self.title}"]
        if self.bullets:
            parts.append(f"Key points: {'; '.join(self.bullets[:4])}")
        if self.body_text:
            parts.append(f"Body: {self.body_text[:100]}")
        if self.image_hint:
            parts.append(f"Visual hint: {self.image_hint}")
        return " | ".join(parts)
