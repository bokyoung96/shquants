from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


def _copy_run_script(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    run_path = root / "run.sh"
    target = tmp_path / "run.sh"
    shutil.copy2(run_path, target)
    return target


def _write_fake_python(tmp_path: Path) -> Path:
    path = tmp_path / "fake-python"
    path.write_text(
        """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

args = sys.argv[1:]

if args and args[0] == "-":
    print(os.environ.get("TEST_CHANNEL", "DOC_POOL"))
    raise SystemExit(0)

if len(args) >= 3 and args[0] == "-m" and args[1] == "analysts.cli":
    cmd = args[2]
    if cmd == "watch-until":
        log = Path("data/state/telegram.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("watch started\\n", encoding="utf-8")
        print("watch started")
        raise SystemExit(0)
    if cmd == "gmail-sync-once":
        print("gmail sync ok")
        raise SystemExit(0)
    if cmd == "gmail-summarize-latest":
        print("gmail summarize ok")
        raise SystemExit(0)

print("unexpected", args)
raise SystemExit(0)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_run_sh_without_args_prints_usage(tmp_path: Path) -> None:
    run_path = _copy_run_script(tmp_path)

    result = subprocess.run(
        ["bash", str(run_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "./run.sh telegram" in result.stdout
    assert "./run.sh gmail" in result.stdout


def test_run_sh_gmail_appends_gmail_log(tmp_path: Path) -> None:
    run_path = _copy_run_script(tmp_path)
    fake_python = _write_fake_python(tmp_path)

    result = subprocess.run(
        ["bash", str(run_path), "gmail"],
        cwd=tmp_path,
        env={**os.environ, "PYTHON_BIN": str(fake_python)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    log_path = tmp_path / "data" / "state" / "gmail.log"
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "gmail sync ok" in log_text
    assert "gmail summarize ok" in log_text


def test_run_sh_telegram_uses_telegram_log(tmp_path: Path) -> None:
    run_path = _copy_run_script(tmp_path)
    fake_python = _write_fake_python(tmp_path)

    result = subprocess.run(
        ["bash", str(run_path), "telegram"],
        cwd=tmp_path,
        env={**os.environ, "PYTHON_BIN": str(fake_python), "TEST_CHANNEL": "DOC_POOL"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "channel: DOC_POOL" in result.stdout
    log_path = tmp_path / "data" / "state" / "telegram.log"
    assert log_path.exists()
    assert "watch started" in log_path.read_text(encoding="utf-8")
