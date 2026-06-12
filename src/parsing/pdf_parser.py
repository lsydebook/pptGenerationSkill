"""Utilities for parsing PDFs into structured payloads."""

from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .document_types import DocumentPayload, ParagraphPayload, SectionPayload, SentencePayload
from .text_splitter import split_paragraphs, split_sentences


def pdf_to_document_payload(
    pdf_path: Path,
    *,
    doc_id: str,
    title: str,
    metadata: dict[str, Any],
) -> DocumentPayload:
    reader = PdfReader(str(pdf_path))
    sections: list[SectionPayload] = []
    all_paragraph_texts: list[str] = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        paragraphs = []
        for paragraph_text in split_paragraphs(raw_text):
            sentences = [
                SentencePayload(text=sentence)
                for sentence in split_sentences(paragraph_text)
            ]
            paragraphs.append(
                ParagraphPayload(
                    text=paragraph_text,
                    sentences=sentences or None,
                    metadata={"page": page_num},
                )
            )
            all_paragraph_texts.append(paragraph_text)

        if paragraphs:
            sections.append(
                SectionPayload(
                    title=f"Page {page_num}",
                    paragraphs=paragraphs,
                    metadata={"page": page_num},
                )
            )
    combined_text = "\n\n".join(all_paragraph_texts)
    return DocumentPayload(
        document_id=doc_id,
        title=title,
        text=combined_text,
        metadata=metadata,
        sections=sections,
    )
