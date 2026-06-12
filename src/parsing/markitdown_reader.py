"""Convert office / PDF / HTML 等文件为 Markdown 文本（MarkItDown）。"""

from __future__ import annotations

from pathlib import Path

from src.config.logging_config import get_logger

logger = get_logger(__name__)

try:
    from markitdown import MarkItDown
except ImportError:  # pragma: no cover
    MarkItDown = None  # type: ignore[misc, assignment]

_converter: MarkItDown | None = None


def _get_converter() -> MarkItDown:
    global _converter
    if MarkItDown is None:
        raise ImportError("markitdown is required; install with: uv add 'markitdown[all]'")
    if _converter is None:
        _converter = MarkItDown()
        logger.info("markitdown converter initialized")
    return _converter


def convert_path_to_markdown(path: Path) -> str:
    """将本地文件转为 Markdown/纯文本，供 markdown_parser 继续结构化。"""
    resolved = path.resolve()
    logger.info(
        "markitdown convert start path=%s size_bytes=%s",
        resolved.name,
        resolved.stat().st_size if resolved.is_file() else 0,
    )
    result = _get_converter().convert(str(resolved))
    text = (result.text_content or "").strip()
    if not text:
        raise ValueError(f"MarkItDown returned empty content for {resolved.name}")
    logger.info(
        "markitdown convert done path=%s text_len=%s",
        resolved.name,
        len(text),
    )
    return text
