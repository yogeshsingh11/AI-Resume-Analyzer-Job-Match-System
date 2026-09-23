"""
pdf_parser.py
-------------
Extracts raw text from an uploaded resume PDF using PyMuPDF (fitz).

Pipeline position:
    Resume.pdf -> PyMuPDF -> plain text
"""

from __future__ import annotations

import io
from typing import Union

import fitz  # PyMuPDF


class PDFParsingError(Exception):
    """Raised when a PDF cannot be opened or contains no extractable text."""


def extract_text_from_pdf(file_source: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extract all text from a PDF.

    Parameters
    ----------
    file_source:
        Either a filesystem path (str), raw PDF bytes, or a BytesIO buffer
        (e.g. the object returned by Streamlit's `st.file_uploader`).

    Returns
    -------
    str
        The concatenated text of every page, cleaned up.
    """
    try:
        if isinstance(file_source, str):
            doc = fitz.open(file_source)
        elif isinstance(file_source, (bytes, bytearray)):
            doc = fitz.open(stream=file_source, filetype="pdf")
        elif isinstance(file_source, io.BytesIO):
            doc = fitz.open(stream=file_source.getvalue(), filetype="pdf")
        else:
            # Streamlit's UploadedFile behaves like a file-like object
            data = file_source.read()
            doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise PDFParsingError(f"Could not open PDF: {exc}") from exc

    pages_text = []
    try:
        for page in doc:
            pages_text.append(page.get_text("text"))
    finally:
        doc.close()

    full_text = "\n".join(pages_text)
    full_text = _clean_text(full_text)

    if not full_text.strip():
        raise PDFParsingError(
            "No extractable text found in this PDF. It may be a scanned "
            "image — try a text-based PDF export of the resume instead."
        )

    return full_text


def _clean_text(text: str) -> str:
    """Light cleanup: normalize whitespace/line breaks without losing structure."""
    lines = [line.rstrip() for line in text.splitlines()]
    # Collapse 3+ blank lines down to a single blank line
    cleaned_lines = []
    blank_run = 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()
