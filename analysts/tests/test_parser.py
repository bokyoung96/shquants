from pathlib import Path

from analysts.domain import ParseQuality, ReportRecord
from analysts.parser import DocumentParser


def _make_report(
    tmp_path: Path,
    *,
    title: str,
    content: str = "",
    pdf_bytes: bytes | None = None,
) -> ReportRecord:
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(pdf_bytes if pdf_bytes is not None else content.encode("utf-8"))
    return ReportRecord(
        id=7,
        source="telegram",
        channel="DOC_POOL",
        message_id=501,
        published_at="2024-04-14T00:30:00Z",
        title=title,
        pdf_path=pdf_path,
        content=content,
        metadata={"file_unique_id": "uniq-001"},
    )


def test_parses_sections_entities_and_tickers_from_text_heavy_report(tmp_path: Path) -> None:
    parser = DocumentParser()
    report = _make_report(
        tmp_path,
        title="AI Capacity Update",
        content=(
            "Executive Summary:\n"
            "NVIDIA (NVDA) and TSMC are adding AI packaging capacity.\n\n"
            "Risks:\n"
            "Samsung could pressure pricing if memory supply recovers faster than expected."
        ),
    )

    parsed = parser.parse(report)

    assert parsed.title == "AI Capacity Update"
    assert parsed.parse_quality is ParseQuality.HIGH
    assert parsed.degraded_reason is None
    assert parsed.sections == [
        "Executive Summary:\nNVIDIA (NVDA) and TSMC are adding AI packaging capacity.",
        "Risks:\nSamsung could pressure pricing if memory supply recovers faster than expected.",
    ]
    assert parsed.entities == ["NVIDIA", "Samsung", "TSMC"]
    assert parsed.tickers == ["NVDA"]


def test_marks_undecodable_pdf_payloads_as_degraded_with_empty_structures(tmp_path: Path) -> None:
    parser = DocumentParser()
    report = _make_report(
        tmp_path,
        title="Scanned Rates Note",
        pdf_bytes=b"\xff\xfe\x00\x81binary-gibberish",
    )

    parsed = parser.parse(report)

    assert parsed.title == "Scanned Rates Note"
    assert parsed.content == ""
    assert parsed.sections == []
    assert parsed.entities == []
    assert parsed.tickers == []
    assert parsed.routes == []
    assert parsed.parse_quality is ParseQuality.DEGRADED
    assert parsed.degraded_reason == "unable_to_decode_pdf_payload"
