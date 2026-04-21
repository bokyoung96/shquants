from pathlib import Path

from analysts.config import build_config
from analysts.pdf_text import PdfTextExtractor



def test_pdf_text_extractor_uses_fallback_text_when_pdf_is_unreadable(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    pdf_path = tmp_path / 'sample.pdf'
    pdf_path.write_bytes(b'not a text pdf')

    result = PdfTextExtractor(config).extract(
        pdf_path=pdf_path,
        slug='report-1',
        fallback_text='fallback summary text',
    )

    assert result.fulltext_path.exists()
    assert result.metadata_path.exists()
    assert result.full_text == 'fallback summary text'
    assert result.quality in {'fallback', 'degraded'}
