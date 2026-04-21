from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import ArasConfig


@dataclass(frozen=True)
class PdfImageMetadata:
    page_number: int
    image_count: int
    preview_path: str | None = None


class PdfImageExtractor:
    def __init__(self, config: ArasConfig) -> None:
        self.config = config

    def extract_metadata(self, *, pdf_path: Path, slug: str) -> tuple[list[PdfImageMetadata], Path]:
        metadata: list[PdfImageMetadata] = []
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            preview_dir = self.config.paths.processed_dir / f"{slug}-pages"
            if self.config.summary.render_page_previews:
                preview_dir.mkdir(parents=True, exist_ok=True)
            for index, page in enumerate(doc, start=1):
                preview_path = None
                if self.config.summary.render_page_previews and index <= self.config.summary.max_preview_pages:
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), alpha=False)
                    preview_file = preview_dir / f"page-{index}.png"
                    pix.save(preview_file)
                    preview_path = str(preview_file.relative_to(self.config.paths.base_dir))
                metadata.append(
                    PdfImageMetadata(
                        page_number=index,
                        image_count=len(page.get_images(full=True)),
                        preview_path=preview_path,
                    )
                )
        except Exception:
            metadata = []
        path = self.config.paths.processed_dir / f"{slug}-images.json"
        path.write_text(
            json.dumps(
                [
                    {"page_number": item.page_number, "image_count": item.image_count, "preview_path": item.preview_path}
                    for item in metadata
                ],
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        return metadata, path

    def render_previews_for_pages(self, *, pdf_path: Path, slug: str, page_numbers: list[int]) -> dict[int, str]:
        rendered: dict[int, str] = {}
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            preview_dir = self.config.paths.processed_dir / f"{slug}-pages"
            preview_dir.mkdir(parents=True, exist_ok=True)
            wanted = set(page_numbers)
            for index, page in enumerate(doc, start=1):
                if index not in wanted:
                    continue
                preview_file = preview_dir / f"page-{index}.png"
                pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), alpha=False)
                pix.save(preview_file)
                rendered[index] = str(preview_file.relative_to(self.config.paths.base_dir))
        except Exception:
            return {}
        return rendered
