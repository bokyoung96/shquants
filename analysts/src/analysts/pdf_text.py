from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import ArasConfig


@dataclass(frozen=True)
class PdfPageText:
    page_number: int
    text: str
    char_count: int


@dataclass(frozen=True)
class PdfTextExtraction:
    method: str
    quality: str
    reason: str | None
    full_text: str
    pages: list[PdfPageText]
    fulltext_path: Path
    metadata_path: Path


class PdfTextExtractor:
    def __init__(self, config: ArasConfig) -> None:
        self.config = config

    def extract(self, *, pdf_path: Path, slug: str, fallback_text: str = "") -> PdfTextExtraction:
        text = ""
        pages: list[PdfPageText] = []
        method = "fallback"
        quality = "degraded"
        reason: str | None = "no_pdf_text_extracted"

        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            page_texts: list[str] = []
            for index, page in enumerate(doc, start=1):
                page_text = page.get_text("text").strip()
                pages.append(PdfPageText(page_number=index, text=page_text, char_count=len(page_text)))
                if page_text:
                    page_texts.append(page_text)
            text = "\n\n".join(part for part in page_texts if part).strip()
            if text:
                method = "pymupdf"
                quality = "high"
                reason = None
        except Exception as exc:  # pragma: no cover - exact extractor failure varies by environment
            reason = f"pymupdf_failed:{type(exc).__name__}"

        if not text and fallback_text.strip():
            text = fallback_text.strip()
            method = "fallback"
            quality = "fallback"
            reason = reason or "used_fallback_text"

        fulltext_path = self.config.paths.processed_dir / f"{slug}-fulltext.txt"
        metadata_path = self.config.paths.processed_dir / f"{slug}-extraction.json"
        fulltext_path.write_text(text + ("\n" if text and not text.endswith("\n") else ""))
        metadata_path.write_text(
            json.dumps(
                {
                    "method": method,
                    "quality": quality,
                    "reason": reason,
                    "page_count": len(pages),
                    "pages": [
                        {
                            "page_number": page.page_number,
                            "char_count": page.char_count,
                        }
                        for page in pages
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        return PdfTextExtraction(
            method=method,
            quality=quality,
            reason=reason,
            full_text=text,
            pages=pages,
            fulltext_path=fulltext_path,
            metadata_path=metadata_path,
        )
