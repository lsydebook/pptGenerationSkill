from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# Stored on paragraph nodes when PARAGRAPH_MODE=both so the para-full
# collection can be written. Never expose this on retrieve/answer APIs.
INTERNAL_METADATA_KEYS = frozenset({"full_embedding"})


def public_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    if not meta:
        return {}
    return {k: v for k, v in meta.items() if k not in INTERNAL_METADATA_KEYS}


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
    embedding: np.ndarray | None = field(default=None, repr=False, compare=False)


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
    created_at: int = 0  # Unix timestamp (seconds), 0 = unknown


@dataclass
class RetrievalMatch:
    node: StoredNode
    score: float


@dataclass
class ContextSnippet:
    node_id: str
    document_title: str
    text: str
    metadata: dict[str, Any]
    rank: int
    score: float
