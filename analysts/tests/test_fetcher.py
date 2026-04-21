import json
from pathlib import Path

import pytest

from analysts.config import build_config
from analysts.domain import ReportRecord
from analysts.fetcher import TelegramFetcher
from analysts.storage import SqliteArasStore



def _load_fixture(name: str) -> list[dict]:
    fixture_path = Path(__file__).parent / 'fixtures' / name
    return json.loads(fixture_path.read_text())


class FakeTelegramClient:
    def __init__(self, updates: list[dict], *, fail_file_ids: set[str] | None = None) -> None:
        self._updates = updates
        self._fail_file_ids = fail_file_ids or set()
        self.requested_offsets: list[int | None] = []
        self.file_requests: list[str] = []
        self.downloaded_paths: list[str] = []

    def get_updates(self, *, offset: int | None, limit: int) -> list[dict]:
        self.requested_offsets.append(offset)
        return [update for update in self._updates if offset is None or update['update_id'] >= offset][:limit]

    def get_file(self, file_id: str) -> dict[str, str]:
        self.file_requests.append(file_id)
        return {'file_path': f'docs/{file_id}.pdf'}

    def download_file(self, file_path: str) -> bytes:
        self.downloaded_paths.append(file_path)
        file_id = Path(file_path).stem
        if file_id in self._fail_file_ids:
            raise RuntimeError(f'download failed for {file_id}')
        return f'PDF bytes for {file_id}'.encode()


class FakeTelethonClient:
    def __init__(self, messages: list[dict], *, fail_file_ids: set[str] | None = None) -> None:
        self._messages = messages
        self._fail_file_ids = fail_file_ids or set()
        self.latest_channel_requests: list[str] = []
        self.channel_requests: list[tuple[str, int | None, int]] = []
        self.downloaded_file_ids: list[str] = []

    def get_latest_message_id(self, *, channel: str) -> int | None:
        self.latest_channel_requests.append(channel)
        matching = [message['message_id'] for message in self._messages if message['chat']['title'] == channel]
        return max(matching) if matching else None

    def iter_channel_messages(self, *, channel: str, after_message_id: int | None, limit: int) -> list[dict]:
        self.channel_requests.append((channel, after_message_id, limit))
        return [
            message
            for message in self._messages
            if message['chat']['title'] == channel and (after_message_id is None or message['message_id'] > after_message_id)
        ][:limit]

    def download_document(self, message: dict) -> bytes:
        file_id = message['document']['file_id']
        self.downloaded_file_ids.append(file_id)
        if file_id in self._fail_file_ids:
            raise RuntimeError(f'download failed for {file_id}')
        return f'PDF bytes for {file_id}'.encode()



def test_downloads_only_unseen_pdf_messages_and_advances_next_offset(tmp_path: Path) -> None:
    client = FakeTelegramClient(_load_fixture('sample_updates.json'))
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    fetcher = TelegramFetcher(client=client, store=store, config=config)

    result = fetcher.poll_once(channel='DOC_POOL')

    assert client.requested_offsets == [None]
    assert [report.metadata['file_unique_id'] for report in result.downloaded] == ['uniq-001']
    assert [item['update_id'] for item in result.skipped_duplicates] == [102]
    assert result.ignored_updates == [101]
    assert result.next_offset == 103
    assert store.get_next_update_offset() == 103

    saved_path = result.downloaded[0].pdf_path
    assert saved_path.exists()
    assert saved_path.read_bytes() == b'PDF bytes for file-001'
    assert saved_path.parent == config.paths.telegram_raw_dir



def test_keeps_stored_offset_when_a_download_fails(tmp_path: Path) -> None:
    client = FakeTelegramClient(_load_fixture('sample_updates.json'), fail_file_ids={'file-001'})
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    store.set_next_update_offset(77)
    fetcher = TelegramFetcher(client=client, store=store, config=config)

    with pytest.raises(RuntimeError, match='download failed for file-001'):
        fetcher.poll_once(channel='DOC_POOL')

    assert client.requested_offsets == [77]
    assert store.get_next_update_offset() == 77
    assert store.list_reports() == []



def test_seeds_last_seen_message_id_on_first_run_without_downloading_history(tmp_path: Path) -> None:
    client = FakeTelethonClient(
        [
            {
                'message_id': 500,
                'date': 1713081000,
                'chat': {'title': 'DOC_POOL'},
                'caption': 'Old report',
                'document': {
                    'file_id': 'file-500',
                    'file_unique_id': 'uniq-500',
                    'file_name': 'old.pdf',
                    'mime_type': 'application/pdf',
                },
            },
            {
                'message_id': 501,
                'date': 1713081060,
                'chat': {'title': 'DOC_POOL'},
                'caption': 'Latest report',
                'document': {
                    'file_id': 'file-501',
                    'file_unique_id': 'uniq-501',
                    'file_name': 'latest.pdf',
                    'mime_type': 'application/pdf',
                },
            },
        ]
    )
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    fetcher = TelegramFetcher(client=client, store=store, config=config)

    result = fetcher.poll_once(channel='DOC_POOL')

    assert client.latest_channel_requests == ['DOC_POOL']
    assert client.channel_requests == []
    assert client.downloaded_file_ids == []
    assert result.downloaded == []
    assert result.next_offset == 501
    assert store.get_last_seen_message_id('DOC_POOL') == 501


def test_infers_last_seen_from_latest_stored_report_when_state_is_missing(tmp_path: Path) -> None:
    client = FakeTelethonClient(
        [
            {
                'message_id': 502,
                'date': 1713081120,
                'chat': {'title': 'DOC_POOL'},
                'caption': 'Text-only update',
            },
            {
                'message_id': 503,
                'date': 1713081180,
                'chat': {'title': 'DOC_POOL'},
                'caption': 'Fresh PDF',
                'document': {
                    'file_id': 'file-503',
                    'file_unique_id': 'uniq-503',
                    'file_name': 'fresh.pdf',
                    'mime_type': 'application/pdf',
                },
            },
        ]
    )
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    existing_pdf = config.paths.raw_dir / 'existing.pdf'
    existing_pdf.parent.mkdir(parents=True, exist_ok=True)
    existing_pdf.write_bytes(b'PDF bytes for existing')
    store.record_download(
        ReportRecord(
            id=None,
            source='telegram',
            channel='DOC_POOL',
            message_id=501,
            published_at='2026-04-15T00:00:00Z',
            title='Existing report',
            pdf_path=existing_pdf,
            content='',
            metadata={'file_unique_id': 'uniq-501'},
        )
    )
    fetcher = TelegramFetcher(client=client, store=store, config=config)

    result = fetcher.poll_once(channel='DOC_POOL')

    assert client.latest_channel_requests == []
    assert client.channel_requests == [('DOC_POOL', 501, config.polling_limit)]
    assert result.ignored_updates == [502]
    assert [report.message_id for report in result.downloaded] == [503]
    assert store.get_last_seen_message_id('DOC_POOL') == 503



def test_downloads_only_new_pdf_messages_after_last_seen_seed(tmp_path: Path) -> None:
    client = FakeTelethonClient(
        [
            {
                'message_id': 502,
                'date': 1713081120,
                'chat': {'title': 'DOC_POOL'},
                'caption': 'Text-only update',
            },
            {
                'message_id': 503,
                'date': 1713081180,
                'chat': {'title': 'DOC_POOL'},
                'caption': 'Fresh PDF',
                'document': {
                    'file_id': 'file-503',
                    'file_unique_id': 'uniq-503',
                    'file_name': 'fresh.pdf',
                    'mime_type': 'application/pdf',
                },
            },
        ]
    )
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    store.set_last_seen_message_id('DOC_POOL', 501)
    fetcher = TelegramFetcher(client=client, store=store, config=config)

    result = fetcher.poll_once(channel='DOC_POOL')

    assert client.channel_requests == [('DOC_POOL', 501, config.polling_limit)]
    assert result.ignored_updates == [502]
    assert [report.message_id for report in result.downloaded] == [503]
    assert result.next_offset == 503
    assert store.get_last_seen_message_id('DOC_POOL') == 503
    assert client.downloaded_file_ids == ['file-503']


def test_skips_generic_message_like_documents_without_pdf_mime_or_filename(tmp_path: Path) -> None:
    client = FakeTelethonClient(
        [
            {
                'message_id': 504,
                'date': 1713081240,
                'chat': {'title': 'DOC_POOL'},
                'caption': '',
                'document': {
                    'file_id': 'file-504',
                    'file_unique_id': 'uniq-504',
                    'file_name': None,
                    'mime_type': '',
                },
            }
        ]
    )
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    store.set_last_seen_message_id('DOC_POOL', 503)
    fetcher = TelegramFetcher(client=client, store=store, config=config)

    result = fetcher.poll_once(channel='DOC_POOL')

    assert result.downloaded == []
    assert result.ignored_updates == [504]
    assert store.get_last_seen_message_id('DOC_POOL') == 504
    assert client.downloaded_file_ids == []



def test_does_not_advance_last_seen_message_id_when_telethon_download_fails(tmp_path: Path) -> None:
    client = FakeTelethonClient(
        [
            {
                'message_id': 503,
                'date': 1713081180,
                'chat': {'title': 'DOC_POOL'},
                'caption': 'Fresh PDF',
                'document': {
                    'file_id': 'file-503',
                    'file_unique_id': 'uniq-503',
                    'file_name': 'fresh.pdf',
                    'mime_type': 'application/pdf',
                },
            }
        ],
        fail_file_ids={'file-503'},
    )
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    store.set_last_seen_message_id('DOC_POOL', 501)
    fetcher = TelegramFetcher(client=client, store=store, config=config)

    with pytest.raises(RuntimeError, match='download failed for file-503'):
        fetcher.poll_once(channel='DOC_POOL')

    assert store.get_last_seen_message_id('DOC_POOL') == 501
    assert store.list_reports() == []


def test_advances_last_seen_to_last_successful_message_before_later_failure(tmp_path: Path) -> None:
    client = FakeTelethonClient(
        [
            {
                'message_id': 502,
                'date': 1713081120,
                'chat': {'title': 'DOC_POOL'},
                'caption': 'Fresh PDF',
                'document': {
                    'file_id': 'file-502',
                    'file_unique_id': 'uniq-502',
                    'file_name': 'first.pdf',
                    'mime_type': 'application/pdf',
                },
            },
            {
                'message_id': 503,
                'date': 1713081180,
                'chat': {'title': 'DOC_POOL'},
                'caption': 'Fresh PDF',
                'document': {
                    'file_id': 'file-503',
                    'file_unique_id': 'uniq-503',
                    'file_name': 'second.pdf',
                    'mime_type': 'application/pdf',
                },
            },
        ],
        fail_file_ids={'file-503'},
    )
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    store.set_last_seen_message_id('DOC_POOL', 501)
    fetcher = TelegramFetcher(client=client, store=store, config=config)

    with pytest.raises(RuntimeError, match='download failed for file-503'):
        fetcher.poll_once(channel='DOC_POOL')

    assert store.get_last_seen_message_id('DOC_POOL') == 502
    reports = store.list_reports()
    assert [report.message_id for report in reports] == [502]
    assert client.downloaded_file_ids == ['file-502', 'file-503']


def test_ingest_message_downloads_new_pdf_and_updates_last_seen(tmp_path: Path) -> None:
    client = FakeTelethonClient(
        [
            {
                'message_id': 700,
                'date': 1713082000,
                'chat': {'title': 'DOC_POOL'},
                'caption': 'Fresh PDF',
                'document': {
                    'file_id': 'file-700',
                    'file_unique_id': 'uniq-700',
                    'file_name': 'fresh-700.pdf',
                    'mime_type': 'application/pdf',
                },
            }
        ]
    )
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    fetcher = TelegramFetcher(client=client, store=store, config=config)

    result = fetcher.ingest_message(channel='DOC_POOL', message=client._messages[0])

    assert result.status == 'downloaded'
    assert result.report is not None
    assert result.report.message_id == 700
    assert result.report.metadata['telegram_caption_text'] == 'Fresh PDF'
    assert store.get_last_seen_message_id('DOC_POOL') == 700
    assert client.downloaded_file_ids == ['file-700']


def test_ingest_message_skips_generic_message_like_document_without_pdf_mime(tmp_path: Path) -> None:
    client = FakeTelethonClient(
        [
            {
                'message_id': 705,
                'date': 1713082120,
                'chat': {'title': 'DOC_POOL'},
                'caption': '',
                'document': {
                    'file_id': 'file-705',
                    'file_unique_id': 'uniq-705',
                    'file_name': None,
                    'mime_type': '',
                },
            }
        ]
    )
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    fetcher = TelegramFetcher(client=client, store=store, config=config)

    result = fetcher.ingest_message(channel='DOC_POOL', message=client._messages[0])

    assert result.status == 'ignored'
    assert result.report is None
    assert store.get_last_seen_message_id('DOC_POOL') == 705
    assert client.downloaded_file_ids == []


def test_ingest_message_downloads_pdf_when_mime_type_marks_pdf_even_without_filename(tmp_path: Path) -> None:
    client = FakeTelethonClient(
        [
            {
                'message_id': 706,
                'date': 1713082180,
                'chat': {'title': 'DOC_POOL'},
                'caption': '',
                'document': {
                    'file_id': 'file-706',
                    'file_unique_id': 'uniq-706',
                    'file_name': None,
                    'mime_type': 'application/pdf',
                },
            }
        ]
    )
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    fetcher = TelegramFetcher(client=client, store=store, config=config)

    result = fetcher.ingest_message(channel='DOC_POOL', message=client._messages[0])

    assert result.status == 'downloaded'
    assert result.report is not None
    assert result.report.message_id == 706
    assert result.report.title == 'report'
    assert store.get_last_seen_message_id('DOC_POOL') == 706
    assert client.downloaded_file_ids == ['file-706']


def test_ingest_message_skips_duplicate_without_resummarizing_download(tmp_path: Path) -> None:
    message = {
        'message_id': 701,
        'date': 1713082060,
        'chat': {'title': 'DOC_POOL'},
        'caption': 'Duplicate PDF',
        'document': {
            'file_id': 'file-701',
            'file_unique_id': 'uniq-701',
            'file_name': 'duplicate-701.pdf',
            'mime_type': 'application/pdf',
        },
    }
    client = FakeTelethonClient([message])
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    fetcher = TelegramFetcher(client=client, store=store, config=config)
    first = fetcher.ingest_message(channel='DOC_POOL', message=message)
    assert first.status == 'downloaded'
    stored = store.get_report_by_file_unique_id('uniq-701')
    assert stored is not None
    summary_path = config.paths.processed_dir / f'report-{stored.id}-summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text('{}\n')

    duplicate = fetcher.ingest_message(channel='DOC_POOL', message=message)

    assert duplicate.status == 'duplicate'
    assert duplicate.report is None
    assert store.get_last_seen_message_id('DOC_POOL') == 701
    assert client.downloaded_file_ids == ['file-701']


def test_ingest_message_recovers_existing_report_when_summary_is_missing(tmp_path: Path) -> None:
    message = {
        'message_id': 702,
        'date': 1713082120,
        'chat': {'title': 'DOC_POOL'},
        'caption': 'Needs summary recovery',
        'document': {
            'file_id': 'file-702',
            'file_unique_id': 'uniq-702',
            'file_name': 'recovery-702.pdf',
            'mime_type': 'application/pdf',
        },
    }
    client = FakeTelethonClient([message])
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    fetcher = TelegramFetcher(client=client, store=store, config=config)

    first = fetcher.ingest_message(channel='DOC_POOL', message=message)
    assert first.status == 'downloaded'
    assert first.report is not None

    recovered = fetcher.ingest_message(channel='DOC_POOL', message=message)

    assert recovered.status == 'existing_unsummarized'
    assert recovered.report is not None
    assert recovered.report.message_id == 702
    assert client.downloaded_file_ids == ['file-702']
