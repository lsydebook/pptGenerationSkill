from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeKind(str, Enum):
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    ATTACHMENT = "attachment"


@dataclass
class SentencePayload:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParagraphPayload:
    text: str
    sentences: list[SentencePayload] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SectionPayload:
    title: str
    paragraphs: list[ParagraphPayload]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentPayload:
    document_id: str
    title: str
    text: str
    metadata: dict[str, Any]
    sections: list[SectionPayload] | None = None


@dataclass
class TreeNode:
    node_id: str
    kind: NodeKind
    text: str
    title: str
    metadata: dict[str, Any]
    parent_id: str | None = None
    children: list["TreeNode"] = field(default_factory=list)


@dataclass
class StoredNode:
    node_id: str
    parent_id: str | None
    kind: NodeKind
    title: str
    text: str
    metadata: dict[str, Any]
    embedding: list[float]
    child_ids: list[str] = field(default_factory=list)


@dataclass
class RetrievalMatch:
    node: StoredNode
    score: float
