from __future__ import annotations

from pathlib import Path
import shlex
import subprocess


def _copy_run_script(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    run_path = root / "run.sh"
    target = tmp_path / "run.sh"
    target.write_text(run_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    target.chmod(0o755)
    return target


def _bash_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.drive:
        drive = resolved.drive.rstrip(":").lower()
        rest = resolved.as_posix().split(":", 1)[1].lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return resolved.as_posix()


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
        newline="\n",
    )
    path.chmod(0o755)
    return path


def _bash_run(script: Path, *args: str, env: dict[str, str] | None = None, cwd: Path):
    exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in (env or {}).items())
    command = " ".join([exports, shlex.quote(_bash_path(script)), *(shlex.quote(arg) for arg in args)]).strip()
    return subprocess.run(
        ["bash", "-lc", command],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_run_sh_without_args_prints_usage(tmp_path: Path) -> None:
    run_path = _copy_run_script(tmp_path)

    result = _bash_run(run_path, cwd=tmp_path)

    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "./run.sh telegram" in result.stdout
    assert "./run.sh gmail" in result.stdout


def test_run_sh_gmail_appends_gmail_log(tmp_path: Path) -> None:
    run_path = _copy_run_script(tmp_path)
    fake_python = _write_fake_python(tmp_path)

    result = _bash_run(run_path, "gmail", env={"PYTHON_BIN": _bash_path(fake_python)}, cwd=tmp_path)

    assert result.returncode == 0
    log_path = tmp_path / "data" / "state" / "gmail.log"
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "gmail sync ok" in log_text
    assert "gmail summarize ok" in log_text


def test_run_sh_telegram_uses_telegram_log(tmp_path: Path) -> None:
    run_path = _copy_run_script(tmp_path)
    fake_python = _write_fake_python(tmp_path)

    result = _bash_run(
        run_path,
        "telegram",
        env={"PYTHON_BIN": _bash_path(fake_python), "TEST_CHANNEL": "DOC_POOL"},
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert "channel: DOC_POOL" in result.stdout
    log_path = tmp_path / "data" / "state" / "telegram.log"
    assert log_path.exists()
    assert "watch started" in log_path.read_text(encoding="utf-8")
