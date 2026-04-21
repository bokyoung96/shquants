from pathlib import Path

from analysts.config import build_config
from analysts.page_selection import ImportantPageSelector
from analysts.pdf_images import PdfImageMetadata
from analysts.pdf_text import PdfPageText



def test_important_page_selector_ranks_numeric_and_image_dense_pages(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    selector = ImportantPageSelector(config)
    page_texts = [
        PdfPageText(page_number=1, text='Executive Summary\nSales 10% margin 20% growth 30%', char_count=60),
        PdfPageText(page_number=2, text='plain page', char_count=10),
        PdfPageText(page_number=3, text='Table page 100 200 300 400 500', char_count=40),
    ]
    image_meta = [
        PdfImageMetadata(page_number=1, image_count=1, preview_path='data/processed/report-1-pages/page-1.png'),
        PdfImageMetadata(page_number=2, image_count=0, preview_path=None),
        PdfImageMetadata(page_number=3, image_count=3, preview_path='data/processed/report-1-pages/page-3.png'),
    ]

    selected, path = selector.select(page_texts=page_texts, image_metadata=image_meta, slug='report-1')

    assert path.exists()
    assert len(selected) == 3
    assert selected[0].page_number in {1, 3}
    assert selected[0].score >= selected[1].score


def test_important_page_selector_scores_chart_and_table_hints(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    selector = ImportantPageSelector(config)
    page_texts = [
        PdfPageText(page_number=1, text='자료: SK증권\nYoY 10% QoQ 5% Spot Price 80%\n100 200 300 400', char_count=80),
        PdfPageText(page_number=2, text='plain commentary page', char_count=30),
    ]
    image_meta = [
        PdfImageMetadata(page_number=1, image_count=2, preview_path='data/processed/report-2-pages/page-1.png'),
        PdfImageMetadata(page_number=2, image_count=0, preview_path=None),
    ]

    selected, _ = selector.select(page_texts=page_texts, image_metadata=image_meta, slug='report-2')

    assert selected[0].page_number == 1
    assert 'chart_hint:3' in selected[0].reasons or any(reason.startswith('chart_hint:') for reason in selected[0].reasons)
    assert 'table_like_pattern' in selected[0].reasons
