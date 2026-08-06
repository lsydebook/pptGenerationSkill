"""Upstream input JSON contract.

This is a *guessed* schema describing what the upstream slicing module
is expected to produce. Tweak field names once we have a real sample.

Top-level shape::

    {
      "meta": { "title": "...", "author": "...", "date": "2026-04-30",
                "week_index": 17, "lab": "..." },
      "sections": [
        {
          "id": "progress",
          "title": "本周进展",
          "kind": "progress" | "research" | "results" | "discussion" | ...,
          "blocks": [
            { "type": "text", "text": "..." },
            { "type": "bullets", "items": ["...", "..."] },
            { "type": "image_ref", "image_id": "img_001", "caption": "..." }
          ]
        }
      ],
      "images": [
        {
          "id": "img_001",
          "path": "examples/images/foo.png",   # or absolute path
          "caption": "图1：...",
          "description": "...",                # multimodal-generated text
          "embedding": [0.01, ...]              # optional vector
        }
      ]
    }
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator


class Meta(BaseModel):
    title: str = Field(default="周报")
    author: str | None = None
    date: str | None = None
    week_index: int | None = None
    lab: str | None = None
    audience: str | None = None
    extra: dict = Field(default_factory=dict)


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class BulletsBlock(BaseModel):
    type: Literal["bullets"] = "bullets"
    items: list[str]


class ImageRefBlock(BaseModel):
    type: Literal["image_ref"] = "image_ref"
    image_id: str
    caption: str | None = None


class FormulaBlock(BaseModel):
    type: Literal["formula"] = "formula"
    latex: str
    caption: str | None = None


class ChartBlock(BaseModel):
    """Inline chart data: small tabular numbers we can render as bar/line/pie."""

    type: Literal["chart"] = "chart"
    kind: Literal["bar", "line", "pie"] = "bar"
    title: str | None = None
    categories: list[str]
    series: list[dict] = Field(
        default_factory=list,
        description='List of {"name": str, "values": list[float]}',
    )
    y_axis_title: str | None = None


Block = Annotated[
    Union[TextBlock, BulletsBlock, ImageRefBlock, FormulaBlock, ChartBlock],
    Field(discriminator="type"),
]


SectionKind = Literal[
    "cover",
    "progress",
    "research",
    "results",
    "discussion",
    "plan",
    "thanks",
    "other",
]


class Section(BaseModel):
    id: str
    title: str
    kind: SectionKind = "other"
    blocks: list[Block] = Field(default_factory=list)


class ImageAsset(BaseModel):
    id: str
    path: str | None = None
    base64: str | None = None
    caption: str | None = None
    description: str | None = None
    embedding: list[float] | None = None

    @model_validator(mode="after")
    def _has_source(self) -> "ImageAsset":
        if not self.path and not self.base64:
            raise ValueError(
                f"ImageAsset {self.id!r} must have either 'path' or 'base64'"
            )
        return self


class InputBundle(BaseModel):
    meta: Meta = Field(default_factory=Meta)
    sections: list[Section] = Field(default_factory=list)
    images: list[ImageAsset] = Field(default_factory=list)

    def image_by_id(self, image_id: str) -> ImageAsset | None:
        for img in self.images:
            if img.id == image_id:
                return img
        return None
