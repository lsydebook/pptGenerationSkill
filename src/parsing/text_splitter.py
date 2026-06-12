import re

SENTENCE_RE = re.compile(r"(?<=[!?;。！？；])\s*")
PARAGRAPH_RE = re.compile(r"\n\s*\n+")


def split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    parts = [segment.strip() for segment in SENTENCE_RE.split(stripped) if segment.strip()]
    return parts if parts else [stripped]


def build_document_summary(
    title: str,
    full_text: str,
    *,
    max_chars: int = 2000,
) -> str:
    """为 DOCUMENT 根节点生成短摘要（标题 + 正文摘录），避免 Milvus text 字段存全文。"""
    limit = max(64, max_chars)
    title_line = (title or "").strip()
    body = " ".join((full_text or "").split())
    if not title_line and not body:
        return ""
    if not body:
        return title_line[:limit]
    prefix = f"{title_line}\n\n" if title_line else ""
    budget = limit - len(prefix)
    if budget <= 0:
        return title_line[:limit]
    if len(body) > budget:
        cut = max(0, budget - 1)
        excerpt = f"{body[:cut].rstrip()}…"
    else:
        excerpt = body
    summary = f"{prefix}{excerpt}"
    return summary[:limit]


def split_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_paragraphs = PARAGRAPH_RE.split(normalized)
    paragraphs = [paragraph.strip() for paragraph in raw_paragraphs if paragraph.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    if "\n" in normalized:
        lines = [line.strip() for line in normalized.split("\n") if line.strip()]
        return lines if lines else [normalized.strip()]

    return paragraphs if paragraphs else [normalized.strip()]
