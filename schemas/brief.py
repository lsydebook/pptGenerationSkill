"""WeeklyReportBrief: condensed summary produced by LLM from upstream InputBundle."""

from pydantic import BaseModel, Field


class WeeklyReportBrief(BaseModel):
    theme: str = Field(description="本周核心主题，一句话概括")
    keywords: list[str] = Field(default_factory=list, description="3-5 个关键词")
    audience: str = Field(default="课题组", description="目标受众")
    tone: str = Field(default="学术", description="整体风格基调")
    estimated_pages: int = Field(default=8, description="预估总页数")
    section_titles: list[str] = Field(default_factory=list, description="建议章节标题列表")
    summary: str = Field(default="", description="一句话概况本周内容")
