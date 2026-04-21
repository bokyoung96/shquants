import json
import subprocess
import sys
import types
from pathlib import Path

from analysts.cli import build_default_pipeline, main, run_watch_until
from analysts.runner_entry import main as runner_entry_main
from analysts.watcher import AsyncWatchResult



def test_show_config_redacts_local_telethon_secrets(tmp_path: Path, capsys) -> None:
    (tmp_path / 'config.local.json').write_text(
        json.dumps(
            {
                'telegram': {
                    'mode': 'telethon',
                    'api_id': 123456,
                    'api_hash': 'super-secret-hash',
                    'phone_number': '+821012345678',
                    'channel': 'DOC_POOL',
                    'session_name': 'doc-pool',
                }
            }
        )
    )

    exit_code = main(['show-config', '--base-dir', str(tmp_path)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload['telethon']['api_hash'] == '<redacted>'
    assert payload['telethon']['channel'] == 'DOC_POOL'


def test_show_config_includes_gmail_paths_and_redacts_inline_client_secret(tmp_path: Path, capsys) -> None:
    (tmp_path / 'config.local.json').write_text(
        json.dumps(
            {
                'gmail': {
                    'account_name': 'reports-primary',
                    'client_secret_json': {
                        'installed': {
                            'client_id': 'client-id',
                            'client_secret': 'super-secret-client-secret',
                            'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
                            'token_uri': 'https://oauth2.googleapis.com/token',
                        }
                    },
                    'token_path': 'data/state/gmail-token.json',
                    'query': 'label:broker-reports',
                    'label_filters': ['Label_Reports'],
                    'body_candidate_rules': {
                        'min_chars': 200,
                        'require_structure': True,
                    },
                    'zip_allow_extensions': ['.pdf', '.txt', '.html'],
                }
            }
        )
    )

    exit_code = main(['show-config', '--base-dir', str(tmp_path)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload['gmail']['account_name'] == 'reports-primary'
    assert payload['gmail']['client_secret_json'] == '<inline>'
    assert payload['gmail']['token_path'] == 'data/state/gmail-token.json'
    assert payload['gmail']['body_candidate_rules'] == {
        'min_chars': 200,
        'require_structure': True,
    }
    assert payload['gmail']['zip_allow_extensions'] == ['.pdf', '.txt', '.html']


def test_build_default_pipeline_prefers_fixture_client_when_fixture_path_is_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture_calls: list[Path] = []
    live_calls: list[tuple[Path, object]] = []
    fixture_client = object()
    live_client = object()
    module = types.ModuleType('analysts.telethon_client')

    class FixtureTelegramClient:
        @classmethod
        def from_fixture_path(cls, fixture_path: Path):
            fixture_calls.append(fixture_path)
            return fixture_client

    class TelethonChannelClient:
        def __init__(self, *, base_dir: Path, config) -> None:
            live_calls.append((base_dir, config))

        def __new__(cls, *args, **kwargs):
            return live_client

    module.auth_login = lambda **kwargs: None
    module.FixtureTelegramClient = FixtureTelegramClient
    module.TelethonChannelClient = TelethonChannelClient
    monkeypatch.setitem(sys.modules, 'analysts.telethon_client', module)

    pipeline = build_default_pipeline(base_dir=tmp_path, fixtures_path='fixtures/sample.json')

    assert pipeline.client is fixture_client
    assert fixture_calls == [Path('fixtures/sample.json')]
    assert live_calls == []


def test_graphify_update_command_prints_manifest_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    processed_dir = tmp_path / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / 'report-1-summary.json').write_text(
        json.dumps(
            {
                'report_title': 'Report Title',
                'message_id': 123,
                'raw_pdf_path': 'data/raw/sample.pdf',
                'important_pages': [1],
                'summaries': [
                    {
                        'lane': 'sector',
                        'headline': 'Headline',
                        'confidence': 'high',
                        'cited_pages': [1],
                        'executive_summary': 'Summary body',
                        'key_points': ['A'],
                        'key_numbers': ['10%'],
                        'risks': ['Risk'],
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=kwargs.get('args', args[0] if args else []), returncode=0)

    monkeypatch.setattr('analysts.graphify.subprocess.run', fake_run)

    assert main(['graphify-update', '--base-dir', str(tmp_path)]) == 0

    output = capsys.readouterr().out.strip()
    assert 'reports=1' in output
    assert 'graphify_invoked=true' in output
    assert 'manifest=' in output


def test_watch_until_command_dispatches_to_async_runner(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    calls: list[tuple[Path, str, str]] = []

    def fake_run_watch_until(*, base_dir: Path, channel: str, until: str) -> int:
        calls.append((base_dir, channel, until))
        print('downloaded=1 duplicates=0 ignored=0 summarized=1 retries=0')
        return 0

    monkeypatch.setattr('analysts.cli.run_watch_until', fake_run_watch_until)

    exit_code = main(
        [
            'watch-until',
            '--channel',
            'DOC_POOL',
            '--until',
            '2026-04-15T17:30:00+09:00',
            '--base-dir',
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert calls == [(tmp_path, 'DOC_POOL', '2026-04-15T17:30:00+09:00')]
    assert 'summarized=1' in capsys.readouterr().out


def test_watch_until_command_accepts_multiple_channels(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    calls: list[tuple[Path, list[str], str]] = []

    def fake_run_watch_until(*, base_dir: Path, channels: list[str], until: str) -> int:
        calls.append((base_dir, channels, until))
        print('downloaded=2 duplicates=0 ignored=0 summarized=2 retries=0')
        return 0

    monkeypatch.setattr('analysts.cli.run_watch_until', fake_run_watch_until)

    exit_code = main(
        [
            'watch-until',
            '--channel',
            'DOC_POOL',
            '--channel',
            'report_figure_by_offset',
            '--until',
            '2026-04-15T17:30:00+09:00',
            '--base-dir',
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert calls == [(tmp_path, ['DOC_POOL', 'report_figure_by_offset'], '2026-04-15T17:30:00+09:00')]
    assert 'summarized=2' in capsys.readouterr().out


def test_run_watch_until_catches_up_backlog_before_subscribing(tmp_path: Path, monkeypatch, capsys) -> None:
    catchup_calls: list[str] = []
    watched: list[tuple[str, str]] = []
    logged: list[tuple[str, tuple[object, ...]]] = []

    class FakeLogger:
        def info(self, message: str, *args) -> None:
            logged.append((message, args))

    class FakePipeline:
        def run_once(self, *, channel: str):
            catchup_calls.append(channel)
            return types.SimpleNamespace(
                summary=types.SimpleNamespace(downloaded=1, duplicates=2, ignored=3, next_offset=123),
                processed_files=[Path('a.json'), Path('b.md')],
                summaries=['sector', 'macro'],
            )

    class FakeRunner:
        logger = None

        async def watch_until(self, *, channel: str, until):
            watched.append((channel, until.isoformat()))
            return AsyncWatchResult(seen=4, downloaded=5, duplicates=6, ignored=7, message_failures=0, summarized=8)

    monkeypatch.setattr('analysts.cli.build_default_pipeline', lambda *, base_dir, fixtures_path=None: FakePipeline())
    monkeypatch.setattr('analysts.cli.build_watch_runner', lambda *, base_dir: FakeRunner())
    monkeypatch.setattr('analysts.cli.configure_watch_logger', lambda *, base_dir: FakeLogger())

    exit_code = run_watch_until(
        base_dir=tmp_path,
        channel='DOC_POOL',
        until='2026-04-15T17:30:00+09:00',
    )

    assert exit_code == 0
    assert catchup_calls == ['DOC_POOL']
    assert watched == [('DOC_POOL', '2026-04-15T17:30:00+09:00')]
    output = capsys.readouterr().out
    assert 'downloaded=6' in output
    assert 'duplicates=8' in output
    assert 'ignored=10' in output
    assert 'summarized=10' in output
    assert ('watch_catchup_started channels=%s', ('DOC_POOL',)) in logged
    assert ('watch_catchup_finished channels=%s downloaded=%s duplicates=%s ignored=%s summarized=%s', ('DOC_POOL', 1, 2, 3, 2)) in logged


def test_runner_entry_uses_doc_pool_default_and_supplied_base_dir(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[Path, list[str], str]] = []

    def fake_run_watch_until(*, base_dir: Path, channels: list[str], until: str) -> int:
        calls.append((base_dir, channels, until))
        return 0

    monkeypatch.setattr('analysts.runner_entry.run_watch_until', fake_run_watch_until)

    exit_code = runner_entry_main(
        ['--until', '2026-04-15T17:30:00+09:00'],
        default_base_dir=tmp_path,
    )

    assert exit_code == 0
    assert calls == [(tmp_path, ['DOC_POOL', 'report_figure_by_offset'], '2026-04-15T17:30:00+09:00')]


def test_runner_entry_allows_multiple_explicit_channels(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[Path, list[str], str]] = []

    def fake_run_watch_until(*, base_dir: Path, channels: list[str], until: str) -> int:
        calls.append((base_dir, channels, until))
        return 0

    monkeypatch.setattr('analysts.runner_entry.run_watch_until', fake_run_watch_until)

    exit_code = runner_entry_main(
        [
            '--channel',
            'DOC_POOL',
            '--channel',
            'report_figure_by_offset',
            '--until',
            '2026-04-15T17:30:00+09:00',
        ],
        default_base_dir=tmp_path,
    )

    assert exit_code == 0
    assert calls == [(tmp_path, ['DOC_POOL', 'report_figure_by_offset'], '2026-04-15T17:30:00+09:00')]
