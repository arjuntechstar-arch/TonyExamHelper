import io
import re

from docx import Document
from pypdf import PdfReader
from pptx import Presentation


class DocumentProcessingError(ValueError):
    pass


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_pages(filename: str, content: bytes) -> list[tuple[int, str]]:
    suffix = filename.rsplit(".", 1)[-1].lower()
    if suffix == "txt":
        return [(1, clean_text(content.decode("utf-8")))]
    if suffix == "pdf":
        return [(index + 1, clean_text(page.extract_text() or "")) for index, page in enumerate(PdfReader(io.BytesIO(content)).pages)]
    if suffix == "docx":
        return [(1, clean_text(" ".join(item.text for item in Document(io.BytesIO(content)).paragraphs)))]
    if suffix == "pptx":
        return [(index + 1, clean_text(" ".join(shape.text for shape in slide.shapes if hasattr(shape, "text")))) for index, slide in enumerate(Presentation(io.BytesIO(content)).slides)]
    raise DocumentProcessingError("Unsupported document type.")


def chunk_pages(pages: list[tuple[int, str]], chunk_size: int = 1000, overlap: int = 150) -> list[dict]:
    chunks: list[dict] = []
    for page, text in pages:
        for start in range(0, len(text), chunk_size - overlap):
            value = text[start:start + chunk_size]
            if value:
                chunks.append({"page_number": page, "content": value})
            if start + chunk_size >= len(text):
                break
    return chunks
