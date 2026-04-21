from pathlib import Path

from analysts.config import build_config
from analysts.domain import ParseQuality, ParsedDocument, ReportRecord, RouteDecision
from analysts.page_selection import ImportantPage
from analysts.pdf_images import PdfImageMetadata
from analysts.pdf_ingest import PdfIngestionPipeline
from analysts.pdf_text import PdfPageText, PdfTextExtraction


def test_pdf_ingest_pipeline_carries_selected_page_metadata_into_summary_packet(
    tmp_path: Path,
) -> None:
    config = build_config(tmp_path)
    pipeline = PdfIngestionPipeline(config)
    slug = 'report-7'

    fulltext_path = config.paths.processed_dir / f'{slug}-fulltext.txt'
    metadata_path = config.paths.processed_dir / f'{slug}-extraction.json'
    images_path = config.paths.processed_dir / f'{slug}-images.json'
    important_pages_path = config.paths.processed_dir / f'{slug}-important-pages.json'
    chunks_path = config.paths.processed_dir / f'{slug}-chunks.json'
    embeddings_path = config.paths.processed_dir / f'{slug}-embeddings.json'
    for path in [fulltext_path, metadata_path, images_path, important_pages_path, chunks_path, embeddings_path]:
        path.write_text('{}\n')

    pipeline.pdf_text.extract = lambda **kwargs: PdfTextExtraction(
        method='fallback',
        quality='fallback',
        reason='used_caption_fallback',
        full_text='full body text',
        pages=[PdfPageText(page_number=1, text='first page', char_count=10)],
        fulltext_path=fulltext_path,
        metadata_path=metadata_path,
    )
    pipeline.pdf_images.extract_metadata = lambda **kwargs: (
        [PdfImageMetadata(page_number=2, image_count=3, preview_path='data/processed/report-7-pages/page-2-old.png')],
        images_path,
    )
    pipeline.chunker.chunk_text = lambda **kwargs: ([{'text': 'chunk-1'}], chunks_path)
    pipeline.embeddings.build_pending_records = lambda **kwargs: ([{'chunk_id': 'chunk-1'}], embeddings_path)
    pipeline.page_selector.select = lambda **kwargs: (
        [
            ImportantPage(
                page_number=2,
                score=7,
                reasons=['numeric_density:8'],
                preview_path='data/processed/report-7-pages/page-2-old.png',
            )
        ],
        important_pages_path,
    )
    pipeline.pdf_images.render_previews_for_pages = (
        lambda **kwargs: {2: 'data/processed/report-7-pages/page-2-selected.png'}
    )

    report = ReportRecord(
        id=7,
        source='telegram',
        channel='DOC_POOL',
        message_id=700,
        published_at='2026-04-15T00:00:00Z',
        title='Report title',
        pdf_path=tmp_path / 'report.pdf',
        content='',
        metadata={'telegram_caption_text': 'caption text'},
    )
    parsed = ParsedDocument(
        title='Report title',
        content='parsed body',
        sections=['parsed body'],
        entities=['NVIDIA'],
        tickers=['NVDA'],
        routes=[],
        parse_quality=ParseQuality.HIGH,
    )
    routes = [RouteDecision(topic='general', lane='macro', rationale='fallback')]

    result = pipeline.ingest(report=report, parsed=parsed, routes=routes)

    assert result.packet.preferred_text == 'full body text'
    assert result.packet.important_pages == [2]
    assert result.packet.page_previews == ['data/processed/report-7-pages/page-2-selected.png']
    assert result.packet.text_excerpt == 'full body text'
    assert [path.name for path in result.processed_files] == [
        'report-7-fulltext.txt',
        'report-7-extraction.json',
        'report-7-images.json',
        'report-7-important-pages.json',
        'report-7-chunks.json',
        'report-7-embeddings.json',
        'report-7-raw-text.txt',
        'report-7-summary-input.json',
    ]
