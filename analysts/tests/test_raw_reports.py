from pathlib import Path

from analysts.raw_reports import RawReportCatalog



def test_raw_report_catalog_reconstructs_reports_from_filenames(tmp_path: Path) -> None:
    raw_dir = tmp_path / 'data' / 'raw' / 'telegram'
    raw_dir.mkdir(parents=True)
    (raw_dir / 'live-123-telethon-abc-sample_report.pdf').write_bytes(b'x')
    catalog = RawReportCatalog(raw_dir=raw_dir)

    reports = catalog.list_reports()

    assert len(reports) == 1
    assert reports[0].message_id == 123
    assert reports[0].title == 'sample report'
