import re

SENTENCE_RE = re.compile(r"(?<=[!?;。！？；])\s*")
PARAGRAPH_RE = re.compile(r"\n\s*\n+")


def split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    parts = [segment.strip() for segment in SENTENCE_RE.split(stripped) if segment.strip()]
    return parts if parts else [stripped]


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
