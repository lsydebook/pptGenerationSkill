"""Tests for structured Markdown parsing."""

from __future__ import annotations

from src.parsing.markdown_parser import parse_markdown_sections


def test_heading_and_paragraph_sections() -> None:
    md = """# 总览

第一段正文。

第二段正文。

## 细节

子章节内容。
"""
    sections = parse_markdown_sections(md, default_title="默认标题")
    assert [s.title for s in sections] == ["总览", "细节"]
    assert sections[0].paragraphs[0].text == "第一段正文。"
    assert sections[0].paragraphs[1].text == "第二段正文。"
    assert sections[1].paragraphs[0].text == "子章节内容。"


def test_table_rows_split_without_relabeling_headers() -> None:
    md = """## Sheet1

| query | output |
| --- | --- |
| 北京天气 | 明天晴 |
| 你是谁 | 我是助手 |
"""
    sections = parse_markdown_sections(md, default_title="doc")
    assert len(sections) == 1
    assert sections[0].title == "Sheet1"
    assert len(sections[0].paragraphs) == 2
    first = sections[0].paragraphs[0]
    assert first.metadata["block_type"] == "table_row"
    assert first.text == "query: 北京天气\noutput: 明天晴"
    assert first.metadata["table_headers"] == ["query", "output"]
    assert len(first.sentences or []) == 1


def test_code_fence_is_atomic_block() -> None:
    md = """## API

```python
def add(a, b):
    return a + b
```
"""
    sections = parse_markdown_sections(md, default_title="doc")
    paragraph = sections[0].paragraphs[0]
    assert paragraph.metadata["block_type"] == "code"
    assert paragraph.metadata["language"] == "python"
    assert "def add" in paragraph.text
    assert len(paragraph.sentences or []) == 1


def test_list_items_are_separate_paragraphs() -> None:
    md = """- 条目一
- 条目二
"""
    sections = parse_markdown_sections(md, default_title="doc")
    assert len(sections[0].paragraphs) == 2
    assert sections[0].paragraphs[0].metadata["block_type"] == "list_item"
    assert sections[0].paragraphs[0].text == "条目一"


def test_blockquote_preserved() -> None:
    md = """> 引用第一行
> 引用第二行
"""
    sections = parse_markdown_sections(md, default_title="doc")
    paragraph = sections[0].paragraphs[0]
    assert paragraph.metadata["block_type"] == "blockquote"
    assert "引用第一行" in paragraph.text
    assert "引用第二行" in paragraph.text
