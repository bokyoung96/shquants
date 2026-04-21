import json
import sys
import types
from dataclasses import replace
from pathlib import Path

from analysts.cli import main
from analysts.config import build_config
from analysts.domain import AnalystSummary, CanonicalDocument, ReportRecord
from analysts.pipeline import ArasPipeline
from analysts.storage import SqliteArasStore


class FakeTelegramClient:
    def __init__(self, updates: list[dict], file_payloads: dict[str, bytes]) -> None:
        self._updates = updates
        self._file_payloads = file_payloads

    def get_updates(self, *, offset: int | None, limit: int) -> list[dict]:
        return [update for update in self._updates if offset is None or update['update_id'] >= offset][:limit]

    @staticmethod
    def get_file(file_id: str) -> dict[str, str]:
        return {'file_path': f'docs/{file_id}.pdf'}

    def download_file(self, file_path: str) -> bytes:
        return self._file_payloads[Path(file_path).stem]


class FakeTelethonClient:
    def __init__(self, messages: list[dict]) -> None:
        self._messages = messages

    def get_latest_message_id(self, *, channel: str) -> int | None:
        message_ids = [message['message_id'] for message in self._messages if message['chat']['title'] == channel]
        return max(message_ids) if message_ids else None

    def iter_channel_messages(self, *, channel: str, after_message_id: int | None, limit: int) -> list[dict]:
        return [
            message
            for message in self._messages
            if message['chat']['title'] == channel and (after_message_id is None or message['message_id'] > after_message_id)
        ][:limit]

    def download_document(self, message: dict) -> bytes:
        return message['payload']


class FakeSummarizer:
    @staticmethod
    def lane_plan(packet) -> list[tuple[str, str]]:
        return [('sector', 'general'), ('macro', 'general')]

    def summarize(self, *, packet, lane: str, topic: str) -> AnalystSummary:
        return AnalystSummary(
            lane=lane,
            topic=topic,
            headline=f'{lane} headline',
            executive_summary=f'{lane} summary',
            key_points=[f'{lane} point'],
            key_numbers=['42%'],
            risks=[f'{lane} risk'],
            confidence='medium',
            cited_pages=[1],
            follow_up_questions=[f'{lane} question'],
        )


def build_fixture_pipeline(tmp_path: Path) -> ArasPipeline:
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    return ArasPipeline(client=object(), store=store, config=config, summarizer=FakeSummarizer())



def test_run_once_writes_full_ingestion_and_summary_artifacts(tmp_path: Path) -> None:
    updates = [
        {
            'update_id': 100,
            'channel_post': {
                'message_id': 501,
                'date': 1713081000,
                'chat': {'id': -1001234567890, 'title': 'DOC_POOL'},
                'caption': 'AI Capacity Update',
                'document': {
                    'file_id': 'file-001',
                    'file_unique_id': 'uniq-001',
                    'file_name': 'ai-capacity-update.pdf',
                    'mime_type': 'application/pdf',
                },
            },
        }
    ]
    file_payloads = {'file-001': b'Executive Summary:\nNVIDIA expands packaging.\n\nRisks:\nSupply concentration.'}
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    pipeline = ArasPipeline(client=FakeTelegramClient(updates, file_payloads), store=store, config=config, summarizer=FakeSummarizer())

    execution = pipeline.run_once(channel='DOC_POOL')

    assert execution.summary.downloaded == 1
    assert len(execution.summaries) == 2
    processed = {path.name for path in execution.processed_files}
    assert 'report-1-fulltext.txt' in processed
    assert 'report-1-extraction.json' in processed
    assert 'report-1-images.json' in processed
    assert 'report-1-important-pages.json' in processed
    assert 'report-1-chunks.json' in processed
    assert 'report-1-embeddings.json' in processed
    assert 'report-1-summary-input.json' in processed
    assert 'report-1-summary.json' in processed
    assert 'report-1-summary.md' in processed



def test_summarize_latest_uses_existing_downloaded_report(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    pdf_path = tmp_path / 'data' / 'raw' / 'live.pdf'
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b'not readable pdf bytes')
    store.record_download(
        ReportRecord(
            id=None,
            source='telegram',
            channel='DOC_POOL',
            message_id=163007,
            published_at='2026-04-15T00:34:10Z',
            title='사모신용 이슈가 시스템 리스크가 아닌 이유',
            pdf_path=pdf_path,
            content='',
            metadata={'file_unique_id': 'telethon-163007', 'telegram_caption_text': 'caption fallback'},
        )
    )
    pipeline = ArasPipeline(client=object(), store=store, config=config, summarizer=FakeSummarizer())

    execution = pipeline.summarize_latest(channel='DOC_POOL')

    assert len(execution.summaries) == 2
    assert execution.summary.next_offset == 163007


def test_summarize_latest_resolves_relative_stored_pdf_paths_against_base_dir(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    relative_pdf_path = Path('data') / 'raw' / 'relative-live.pdf'
    absolute_pdf_path = tmp_path / relative_pdf_path
    absolute_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_pdf_path.write_bytes(b'not readable pdf bytes')
    store.record_download(
        ReportRecord(
            id=None,
            source='telegram',
            channel='DOC_POOL',
            message_id=163090,
            published_at='2026-04-15T02:56:28Z',
            title='AI 로 돈을 번다는 것',
            pdf_path=relative_pdf_path,
            content='',
            metadata={'file_unique_id': 'telethon-163090', 'telegram_caption_text': 'caption fallback'},
        )
    )
    pipeline = ArasPipeline(client=object(), store=store, config=config, summarizer=FakeSummarizer())

    execution = pipeline.summarize_latest(channel='DOC_POOL')

    assert execution.summary.next_offset == 163090
    assert len(execution.summaries) == 2


def test_summarize_latest_resolves_paths_with_redundant_analysts_prefix(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    actual_pdf_path = tmp_path / 'data' / 'raw' / 'prefixed-live.pdf'
    actual_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    actual_pdf_path.write_bytes(b'not readable pdf bytes')
    store.record_download(
        ReportRecord(
            id=None,
            source='telegram',
            channel='DOC_POOL',
            message_id=163091,
            published_at='2026-04-15T02:56:29Z',
            title='Prefixed relative path',
            pdf_path=Path('analysts') / 'data' / 'raw' / 'prefixed-live.pdf',
            content='',
            metadata={'file_unique_id': 'telethon-163091', 'telegram_caption_text': 'caption fallback'},
        )
    )
    pipeline = ArasPipeline(client=object(), store=store, config=config, summarizer=FakeSummarizer())

    execution = pipeline.summarize_latest(channel='DOC_POOL')

    assert execution.summary.next_offset == 163091
    assert len(execution.summaries) == 2


def test_summarize_latest_falls_back_to_message_id_prefixed_raw_file_when_stored_path_is_stale(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    actual_pdf_path = tmp_path / 'data' / 'raw' / '163092-telethon-live-actual.pdf'
    actual_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    actual_pdf_path.write_bytes(b'not readable pdf bytes')
    store.record_download(
        ReportRecord(
            id=None,
            source='telegram',
            channel='DOC_POOL',
            message_id=163092,
            published_at='2026-04-15T02:56:30Z',
            title='Stale stored path',
            pdf_path=Path('analysts') / 'data' / 'raw' / '163092-telethon-stale-name.pdf',
            content='',
            metadata={'file_unique_id': 'telethon-163092', 'telegram_caption_text': 'caption fallback'},
        )
    )
    pipeline = ArasPipeline(client=object(), store=store, config=config, summarizer=FakeSummarizer())

    execution = pipeline.summarize_latest(channel='DOC_POOL')

    assert execution.summary.next_offset == 163092
    assert len(execution.summaries) == 2




def test_summarize_latest_falls_back_to_raw_reports_when_db_is_empty(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    raw_dir = tmp_path / 'data' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = raw_dir / 'live-999-telethon-abc-raw_title.pdf'
    pdf_path.write_bytes(b'not readable pdf bytes')
    pipeline = ArasPipeline(client=object(), store=store, config=config, summarizer=FakeSummarizer())

    execution = pipeline.summarize_latest(channel='DOC_POOL')

    assert execution.summary.next_offset == 999
    assert len(execution.summaries) == 2

def test_show_config_prints_serialized_paths(tmp_path: Path, capsys) -> None:
    assert main(['show-config', '--base-dir', str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['paths']['base_dir'] == str(tmp_path)


def test_canonical_document_supports_gmail_body_source(tmp_path: Path) -> None:
    document = CanonicalDocument(
        source="gmail",
        source_message_id="msg-1",
        source_thread_id="thread-1",
        source_feed="reports-primary",
        document_kind="email_body",
        title="Morning broker wrap",
        published_at="2026-04-16T06:00:00Z",
        sender_or_origin="broker@example.com",
        mime_type="text/plain",
        dedupe_key="body::msg-1::hash",
        raw_path=tmp_path / "raw.txt",
        normalized_text_path=tmp_path / "normalized.txt",
        metadata={"label_ids": ["Label_Reports"]},
    )

    assert document.source == "gmail"
    assert document.document_kind == "email_body"


def test_pipeline_summarizes_text_backed_canonical_document(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    raw_path = tmp_path / "data" / "raw" / "gmail-body.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("Revenue up 10%\n\nMargin stable\n\nRisk: FX")
    pipeline = ArasPipeline(client=object(), store=store, config=config, summarizer=FakeSummarizer())
    document = CanonicalDocument(
        source="gmail",
        source_message_id="msg-1",
        source_thread_id="thread-1",
        source_feed="reports-primary",
        document_kind="email_body",
        title="Morning wrap",
        published_at="2026-04-16T06:00:00Z",
        sender_or_origin="broker@example.com",
        mime_type="text/plain",
        dedupe_key="body::msg-1::hash",
        raw_path=raw_path,
        normalized_text_path=raw_path,
        metadata={},
    )

    execution = pipeline.summarize_canonical(document)

    assert len(execution.summaries) == 2
    assert execution.summary.next_offset == "msg-1"
    assert {path.name for path in execution.processed_files} == {
        "report-msg-1-raw-text.txt",
        "report-msg-1-summary-input.json",
        "report-msg-1-summary.json",
        "report-msg-1-summary.md",
    }


def test_pipeline_summarizes_pdf_backed_canonical_document(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    raw_path = tmp_path / "data" / "raw" / "gmail-report.pdf"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"%PDF-1.4 fake")
    pipeline = ArasPipeline(client=object(), store=store, config=config, summarizer=FakeSummarizer())
    document = CanonicalDocument(
        source="gmail",
        source_message_id="msg-2",
        source_thread_id="thread-2",
        source_feed="reports-primary",
        document_kind="attachment_pdf",
        title="Attached report",
        published_at="2026-04-16T06:10:00Z",
        sender_or_origin="broker@example.com",
        mime_type="application/pdf",
        dedupe_key="pdf::msg-2::hash",
        raw_path=raw_path,
        normalized_text_path=None,
        metadata={},
    )

    execution = pipeline.summarize_canonical(document)

    assert len(execution.summaries) == 2
    assert execution.summary.next_offset == "msg-2"



def test_auth_login_dispatches_to_telethon_adapter(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[Path, object]] = []
    def fake_auth_login(*, base_dir: Path, config) -> None:
        calls.append((base_dir, config))

    monkeypatch.setattr('analysts.cli.telegram_auth_login', fake_auth_login)

    assert main(['auth-login', '--base-dir', str(tmp_path)]) == 0
    assert calls and calls[0][0] == tmp_path



def test_summarize_recent_cli_reports_counts(tmp_path: Path, monkeypatch, capsys) -> None:
    config = build_config(tmp_path)
    store = SqliteArasStore(config.paths.state_db)
    raw_dir = tmp_path / 'data' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(2):
        pdf_path = raw_dir / f'live-{idx}.pdf'
        pdf_path.write_bytes(b'not readable pdf bytes')
        store.record_download(
            ReportRecord(
                id=None,
                source='telegram',
                channel='DOC_POOL',
                message_id=100 + idx,
                published_at='2026-04-15T00:00:00Z',
                title=f'title-{idx}',
                pdf_path=pdf_path,
                content='',
                metadata={'file_unique_id': f'u-{idx}', 'telegram_caption_text': f'caption {idx}'},
            )
        )

    class FixtureTelegramClient:
        @classmethod
        def from_fixture_path(cls, fixture_path: Path):
            raise AssertionError('not used')

    class TelethonChannelClient:
        def __init__(self, *, base_dir: Path, config) -> None:
            self.base_dir = base_dir
            self.config = config

    module = types.ModuleType('analysts.telethon_client')
    module.auth_login = lambda **kwargs: None
    module.FixtureTelegramClient = FixtureTelegramClient
    module.TelethonChannelClient = TelethonChannelClient
    monkeypatch.setitem(sys.modules, 'analysts.telethon_client', module)
    monkeypatch.setattr('analysts.cli.build_default_pipeline', lambda **kwargs: ArasPipeline(client=object(), store=store, config=config, summarizer=FakeSummarizer()))

    assert main(['summarize-recent', '--channel', 'DOC_POOL', '--limit', '2', '--base-dir', str(tmp_path)]) == 0
    output = capsys.readouterr().out.strip()
    assert 'reports=2' in output
    assert 'summaries=4' in output
