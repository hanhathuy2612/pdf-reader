"""Extract text from PDF files using PyMuPDF (fitz)."""

from pathlib import Path
from typing import Union

import fitz


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
        if not path.suffix.lower() == ".pdf":
            raise ValueError("File must have .pdf extension")
        doc = fitz.open(path)
    elif isinstance(source, bytes):
        if len(source) == 0:
            raise ValueError("PDF data is empty")
        doc = fitz.open(stream=source, filetype="pdf")

    try:
        parts = []
        for page in doc:
            parts.append(page.get_text())
        return "\n".join(parts).strip() or ""
    finally:
        doc.close()
