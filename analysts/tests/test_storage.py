from pathlib import Path

from analysts.domain import ReportRecord
from analysts.storage import SqliteArasStore



def test_records_unique_reports_and_persists_next_update_offset(tmp_path: Path) -> None:
    store = SqliteArasStore(tmp_path / 'data' / 'state' / 'aras.sqlite3')
    report = ReportRecord(
        id=None,
        source='telegram',
        channel='DOC_POOL',
        message_id=501,
        published_at='2024-04-14T00:30:00Z',
        title='TSMC capacity update',
        pdf_path=tmp_path / 'data' / 'raw' / 'tsmc-capacity.pdf',
        content='',
        metadata={'file_unique_id': 'uniq-001', 'telegram_file_id': 'file-001'},
    )

    assert store.record_download(report) is True
    assert store.record_download(report) is False
    assert len(store.list_reports()) == 1
    assert store.get_next_update_offset() is None
    store.set_next_update_offset(103)
    assert store.get_next_update_offset() == 103



def test_checks_seen_file_unique_ids_without_reinserting(tmp_path: Path) -> None:
    store = SqliteArasStore(tmp_path / 'aras.sqlite3')
    report = ReportRecord(
        id=None,
        source='telegram',
        channel='DOC_POOL',
        message_id=600,
        published_at='2024-04-14T01:00:00Z',
        title='Rates daily',
        pdf_path=tmp_path / 'rates-daily.pdf',
        content='',
        metadata={'file_unique_id': 'uniq-rates-1', 'telegram_file_id': 'file-rates-1'},
    )

    assert store.has_seen_file('uniq-rates-1') is False
    assert store.record_download(report) is True
    assert store.has_seen_file('uniq-rates-1') is True



def test_persists_last_seen_message_ids_per_channel_and_reads_latest_report(tmp_path: Path) -> None:
    store = SqliteArasStore(tmp_path / 'aras.sqlite3')
    older = ReportRecord(
        id=None,
        source='telegram',
        channel='DOC_POOL',
        message_id=10,
        published_at='2024-04-14T00:00:00Z',
        title='Older',
        pdf_path=tmp_path / 'older.pdf',
        content='',
        metadata={'file_unique_id': 'older'},
    )
    newer = ReportRecord(
        id=None,
        source='telegram',
        channel='DOC_POOL',
        message_id=11,
        published_at='2024-04-14T01:00:00Z',
        title='Newer',
        pdf_path=tmp_path / 'newer.pdf',
        content='',
        metadata={'file_unique_id': 'newer'},
    )
    store.record_download(older)
    store.record_download(newer)
    store.set_last_seen_message_id('DOC_POOL', 501)
    store.set_last_seen_message_id('OTHER_POOL', 99)

    assert store.get_last_seen_message_id('DOC_POOL') == 501
    assert store.get_last_seen_message_id('OTHER_POOL') == 99
    assert store.get_latest_report('DOC_POOL').message_id == 11
