from typing import Any

from pydantic import BaseModel


class ParsedDocument(BaseModel):
    text: str
    metadata: dict[str, Any]


class IndexingSummary(BaseModel):
    document_id: str
    nodes_indexed: int


class ParseResponse(BaseModel):
    filename: str
    content_type: str | None
    documents: list[ParsedDocument]
    indexing: list[IndexingSummary]
