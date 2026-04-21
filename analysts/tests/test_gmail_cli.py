from pathlib import Path

from analysts.cli import main


def test_gmail_auth_login_command_runs(monkeypatch) -> None:
    called = {}

    def fake_run(*, base_dir: Path) -> int:
        called["base_dir"] = str(base_dir)
        return 0

    monkeypatch.setattr("analysts.cli.run_gmail_auth_login", fake_run)

    exit_code = main(["gmail-auth-login", "--base-dir", "analysts"])

    assert exit_code == 0
    assert called == {"base_dir": "analysts"}


def test_gmail_sync_once_command_runs(monkeypatch) -> None:
    called = {}

    def fake_run(*, base_dir: Path, limit: int) -> int:
        called["base_dir"] = str(base_dir)
        called["limit"] = limit
        return 0

    monkeypatch.setattr("analysts.cli.run_gmail_sync_once", fake_run)

    exit_code = main(["gmail-sync-once", "--base-dir", "analysts", "--limit", "5"])

    assert exit_code == 0
    assert called == {"base_dir": "analysts", "limit": 5}


def test_gmail_sync_recent_command_runs(monkeypatch) -> None:
    called = {}

    def fake_run(*, base_dir: Path, limit: int) -> int:
        called["base_dir"] = str(base_dir)
        called["limit"] = limit
        return 0

    monkeypatch.setattr("analysts.cli.run_gmail_sync_recent", fake_run)

    exit_code = main(["gmail-sync-recent", "--base-dir", "analysts", "--limit", "7"])

    assert exit_code == 0
    assert called == {"base_dir": "analysts", "limit": 7}


def test_gmail_summarize_latest_command_runs(monkeypatch) -> None:
    called = {}

    def fake_run(*, base_dir: Path) -> int:
        called["base_dir"] = str(base_dir)
        return 0

    monkeypatch.setattr("analysts.cli.run_gmail_summarize_latest", fake_run)

    exit_code = main(["gmail-summarize-latest", "--base-dir", "analysts"])

    assert exit_code == 0
    assert called == {"base_dir": "analysts"}


def test_gmail_summarize_recent_command_runs(monkeypatch) -> None:
    called = {}

    def fake_run(*, base_dir: Path, limit: int) -> int:
        called["base_dir"] = str(base_dir)
        called["limit"] = limit
        return 0

    monkeypatch.setattr("analysts.cli.run_gmail_summarize_recent", fake_run)

    exit_code = main(["gmail-summarize-recent", "--base-dir", "analysts", "--limit", "3"])

    assert exit_code == 0
    assert called == {"base_dir": "analysts", "limit": 3}
