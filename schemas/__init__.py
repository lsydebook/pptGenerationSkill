"""Pydantic schemas: upstream input + internal pipeline IR."""
from .brief import WeeklyReportBrief
from .content import SlideContent
from .input import (
    Block,
    BulletsBlock,
    ChartBlock,
    FormulaBlock,
    ImageAsset,
    ImageRefBlock,
    InputBundle,
    Meta,
    Section,
    TextBlock,
)
from .outline import OutlineItem, WeeklyDeckOutline
from .slide import (
    CalloutSpec,
    CalloutType,
    ChartSpec,
    ImageAssetRef,
    LayoutName,
    PageType,
    TableSpec,
    WeeklySlideSpec,
)
from .visual import VisualDecision

__all__ = [
    "InputBundle",
    "Meta",
    "Section",
    "Block",
    "TextBlock",
    "BulletsBlock",
    "ImageRefBlock",
    "FormulaBlock",
    "ChartBlock",
    "ImageAsset",
    "WeeklyReportBrief",
    "WeeklyDeckOutline",
    "OutlineItem",
    "SlideContent",
    "VisualDecision",
    "WeeklySlideSpec",
    "ChartSpec",
    "CalloutSpec",
    "CalloutType",
    "TableSpec",
    "ImageAssetRef",
    "PageType",
    "LayoutName",
]
