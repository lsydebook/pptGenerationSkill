from __future__ import annotations

import re
from collections.abc import Sequence

from src.config.indexing_config import MIN_SENTENCE_CHARS

SENTENCE_RE = re.compile(r"(?<=[!?;。！？；])\s*")
PARAGRAPH_RE = re.compile(r"\n\s*\n+")


def merge_short_sentences(
    sentences: list[str],
    *,
    min_chars: int | None = None,
) -> list[str]:
    """将过短句子并入相邻句，避免半句单独 embed。"""
    threshold = MIN_SENTENCE_CHARS if min_chars is None else min_chars
    if threshold <= 0:
        return [s for s in sentences if s.strip()]

    merged: list[str] = []
    buffer = ""
    for raw in sentences:
        piece = raw.strip()
        if not piece:
            continue
        if not buffer:
            buffer = piece
            continue
        if len(buffer) < threshold:
            joiner = "" if _cjk_adjacent(buffer, piece) else " "
            buffer = f"{buffer}{joiner}{piece}"
            continue
        merged.append(buffer)
        buffer = piece

    if buffer:
        if merged and len(buffer) < threshold:
            joiner = "" if _cjk_adjacent(merged[-1], buffer) else " "
            merged[-1] = f"{merged[-1]}{joiner}{buffer}"
        else:
            merged.append(buffer)
    return merged


def _cjk_adjacent(left: str, right: str) -> bool:
    if not left or not right:
        return True
    return _is_cjk(left[-1]) or _is_cjk(right[0])


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFFEF
    )


def split_sentences(text: str, *, min_chars: int | None = None) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    parts = [segment.strip() for segment in SENTENCE_RE.split(stripped) if segment.strip()]
    raw = parts if parts else [stripped]
    return merge_short_sentences(raw, min_chars=min_chars)


def build_document_summary(
    title: str,
    full_text: str,
    *,
    max_chars: int = 2000,
) -> str:
    """为 DOCUMENT / SECTION 生成短摘要（标题 + 正文摘录）。"""
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


def split_to_max_chars(
    text: str,
    max_chars: int,
    *,
    overlap: int = 0,
) -> list[str]:
    """将超长文本切到 max_chars 以内：优先按句打包，否则按行/词/字符硬切。"""
    stripped = (text or "").strip()
    if not stripped:
        return []
    if max_chars <= 0 or len(stripped) <= max_chars:
        return [stripped]
    sentences = split_sentences(stripped, min_chars=0)
    if len(sentences) >= 2:
        return pack_texts(sentences, max_chars)
    return split_by_limit(stripped, max_chars, overlap=overlap)


def pack_texts(parts: Sequence[str], max_chars: int) -> list[str]:
    """按 CJK 感知连接符把相邻片段打包到 max_chars 以内。"""
    if max_chars <= 0:
        return [part.strip() for part in parts if part.strip()]
    out: list[str] = []
    buf = ""
    for raw in parts:
        piece = (raw or "").strip()
        if not piece:
            continue
        if len(piece) > max_chars:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(split_by_limit(piece, max_chars))
            continue
        if not buf:
            buf = piece
            continue
        joiner = "" if _cjk_adjacent(buf, piece) else " "
        candidate = f"{buf}{joiner}{piece}"
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            out.append(buf)
            buf = piece
    if buf:
        out.append(buf)
    return out


def split_by_limit(text: str, max_chars: int, *, overlap: int = 0) -> list[str]:
    """按换行 → 空格 → 字符窗口切开，保证每片不超过 max_chars。"""
    stripped = (text or "").strip()
    if not stripped:
        return []
    if max_chars <= 0 or len(stripped) <= max_chars:
        return [stripped]
    if "\n" in stripped:
        lines = [line.strip() for line in stripped.split("\n") if line.strip()]
        return _pack_or_hard(lines, max_chars, joiner="\n", overlap=overlap)
    if " " in stripped:
        words = [word for word in stripped.split() if word]
        return _pack_or_hard(words, max_chars, joiner=" ", overlap=overlap)
    return _hard_windows(stripped, max_chars, overlap)


def _pack_or_hard(
    parts: Sequence[str],
    max_chars: int,
    *,
    joiner: str,
    overlap: int,
) -> list[str]:
    out: list[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        if len(part) > max_chars:
            if buf:
                out.append(buf)
                buf = ""
            if "\n" in part or (joiner != " " and " " in part):
                out.extend(split_by_limit(part, max_chars, overlap=overlap))
            else:
                out.extend(_hard_windows(part, max_chars, overlap))
            continue
        candidate = part if not buf else f"{buf}{joiner}{part}"
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            out.append(buf)
            buf = part
    if buf:
        out.append(buf)
    return out


def _hard_windows(text: str, max_chars: int, overlap: int = 0) -> list[str]:
    if max_chars <= 0:
        return [text] if text else []
    overlap = max(0, min(overlap, max_chars // 4))
    step = max(1, max_chars - overlap)
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + max_chars)
        chunks.append(text[start:end])
        if end >= length:
            break
        start += step
    return chunks
