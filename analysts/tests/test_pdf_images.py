from pathlib import Path

import fitz

from analysts.config import build_config
from analysts.pdf_images import PdfImageExtractor


def test_pdf_image_extractor_writes_preview_paths_for_first_pages(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    pdf_path = tmp_path / 'sample.pdf'
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f'Page {i+1}')
    doc.save(pdf_path)
    doc.close()

    metadata, json_path = PdfImageExtractor(config).extract_metadata(pdf_path=pdf_path, slug='report-1')

    assert json_path.exists()
    assert len(metadata) == 3
    assert metadata[0].preview_path is not None
    assert metadata[1].preview_path is not None
    assert metadata[2].preview_path is None
    assert (tmp_path / metadata[0].preview_path).exists()


def test_pdf_image_extractor_can_render_selected_pages(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    pdf_path = tmp_path / 'sample-selected.pdf'
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f'Page {i+1}')
    doc.save(pdf_path)
    doc.close()

    rendered = PdfImageExtractor(config).render_previews_for_pages(pdf_path=pdf_path, slug='report-2', page_numbers=[2, 3])

    assert 2 in rendered and 3 in rendered
    assert (tmp_path / rendered[2]).exists()
