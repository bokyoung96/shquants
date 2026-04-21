from pathlib import Path

from analysts.config import build_config
from analysts.domain import ParseQuality, ParsedDocument, ReportRecord, RouteDecision
from analysts.extraction import SummaryReadyExtractor



def test_extraction_packet_falls_back_to_report_title_when_pdf_text_is_unreadable(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    extractor = SummaryReadyExtractor(config)
    report = ReportRecord(
        id=7,
        source='telegram',
        channel='DOC_POOL',
        message_id=163007,
        published_at='2026-04-15T00:34:10Z',
        title='사모신용 이슈가 시스템 리스크가 아닌 이유\n핵심 요약',
        pdf_path=tmp_path / 'raw.pdf',
        content='',
        metadata={},
    )
    parsed = ParsedDocument(
        title=report.title,
        content='',
        sections=[],
        entities=[],
        tickers=[],
        routes=[],
        parse_quality=ParseQuality.DEGRADED,
        degraded_reason='unable_to_decode_pdf_payload',
    )
    routes = [RouteDecision(topic='general', lane='macro', rationale='fallback')]

    packet = extractor.build_packet(report=report, parsed=parsed, routes=routes)

    assert packet.extraction_quality == 'fallback'
    assert packet.extraction_reason == 'unable_to_decode_pdf_payload'
    assert '사모신용' in packet.text_excerpt
    assert packet.route_hints == ['macro:general']



def test_write_artifacts_persists_raw_text_and_summary_input(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    extractor = SummaryReadyExtractor(config)
    report = ReportRecord(
        id=3,
        source='telegram',
        channel='DOC_POOL',
        message_id=77,
        published_at='2026-04-15T00:00:00Z',
        title='Title',
        pdf_path=tmp_path / 'raw.pdf',
        content='',
        metadata={},
    )
    parsed = ParsedDocument(
        title='Title',
        content='Line one\n\nLine two',
        sections=['Line one', 'Line two'],
        entities=['NVIDIA'],
        tickers=['NVDA'],
        routes=[],
        parse_quality=ParseQuality.HIGH,
        degraded_reason=None,
    )
    packet = extractor.build_packet(report=report, parsed=parsed, routes=[])

    artifacts = extractor.write_artifacts(packet)

    assert artifacts.raw_text_path.exists()
    assert artifacts.summary_input_path.exists()
    assert 'Line one' in artifacts.raw_text_path.read_text()
    assert 'NVDA' in artifacts.summary_input_path.read_text()


def test_extraction_packet_carries_page_previews(tmp_path):
    from analysts.domain import ParsedDocument, ReportRecord, RouteDecision, ParseQuality
    from analysts.extraction import SummaryReadyExtractor
    config = build_config(tmp_path)
    report = ReportRecord(id=1, source='telegram', channel='DOC_POOL', message_id=1, published_at=None, title='t', pdf_path=tmp_path / 'a.pdf', content='', metadata={'page_previews': ['data/processed/report-1-pages/page-1.png']})
    parsed = ParsedDocument(title='t', content='hello', sections=[], entities=[], tickers=[], routes=[], parse_quality=ParseQuality.HIGH)
    packet = SummaryReadyExtractor(config).build_packet(report=report, parsed=parsed, routes=[RouteDecision(topic='general', lane='macro', rationale='x')])
    assert packet.page_previews == ['data/processed/report-1-pages/page-1.png']
