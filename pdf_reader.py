"""Extract text from PDF files using PyMuPDF with pdfplumber fallback."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import fitz


def _extract_text_with_pymupdf(source: Union[str, Path, bytes]) -> str:
    if isinstance(source, (str, Path)):
        doc = fitz.open(source)
    else:
        doc = fitz.open(stream=source, filetype="pdf")

    try:
        pages: list[str] = []
        for page in doc:
            blocks = page.get_text("blocks", sort=True) or []
            lines: list[str] = []
            for block in blocks:
                if len(block) < 5:
                    continue
                text = str(block[4] or "").strip()
                if text:
                    lines.append(text)
            page_text = "\n".join(lines).strip()
            pages.append(page_text)
        return "\n\n".join(p for p in pages if p).strip()
    finally:
        doc.close()


def _extract_text_with_pdfplumber(source: Union[str, Path, bytes]) -> str:
    try:
        import pdfplumber  # type: ignore[import-not-found]
    except ImportError:
        return ""

    handle = source if isinstance(source, (str, Path)) else io.BytesIO(source)
    with pdfplumber.open(handle) as pdf:
        pages: list[str] = []
        for page in pdf.pages:
            text = (page.extract_text(layout=True) or "").strip()
            if text:
                pages.append(text)
        return "\n\n".join(pages).strip()


def extract_text_from_pdf(source: Union[str, Path, bytes]) -> str:
    """
    Extract plain text from a PDF file.

    Args:
        source: Path to a PDF file (str or Path) or raw PDF bytes (e.g. from upload).

    Returns:
        Concatenated text from all pages.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If file is not a valid PDF or cannot be read.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError("File must have .pdf extension")
        normalized_source: Union[str, Path, bytes] = path
    elif isinstance(source, bytes):
        if len(source) == 0:
            raise ValueError("PDF data is empty")
        normalized_source = source
    else:
        raise ValueError("Source must be a PDF path or PDF bytes")

    try:
        text = _extract_text_with_pymupdf(normalized_source)
    except Exception as exc:
        raise ValueError(f"Cannot read PDF with PyMuPDF: {exc}") from exc

    if len(text) >= 200:
        return text

    fallback = _extract_text_with_pdfplumber(normalized_source)
    return fallback or text
