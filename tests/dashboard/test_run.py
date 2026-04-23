from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from dashboard.run import build_frontend, build_parser, launch_dashboard
from dashboard.strategies import DEFAULT_LAUNCH_CONFIG

EXPECTED_NPM_COMMAND = "npm.cmd" if sys.platform.startswith("win") else "npm"


def test_launch_dashboard_executes_missing_momentum_preset(tmp_path: Path, monkeypatch) -> None:
    observed_configs = []
    captured_defaults = {}

    class FakeRunner:
        def run(self, config):
            observed_configs.append(config)
            return type("Report", (), {"output_dir": tmp_path / f"{config.strategy}_20260405_120000"})()

    monkeypatch.setattr("dashboard.run.build_frontend", lambda frontend_dir: None)
    monkeypatch.setattr("dashboard.run.DEFAULT_LAUNCH_CONFIG", DEFAULT_LAUNCH_CONFIG)
    monkeypatch.setattr("dashboard.run.BacktestRunner", lambda result_dir=None: FakeRunner())
    monkeypatch.setattr(
        "dashboard.run.create_app",
        lambda default_selected_run_ids, frontend_dist=None: captured_defaults.setdefault("run_ids", default_selected_run_ids)
        or object(),
    )
    monkeypatch.setattr("dashboard.run.uvicorn.run", lambda app, host, port: None)
    monkeypatch.setattr(
        "dashboard.run.LaunchResolutionService.resolve",
        lambda self, config: type(
            "Plan",
            (),
            {
                "resolved_runs": (),
                "missing_presets": (config.strategies[0],),
                "selected_run_ids": [],
            },
        )(),
    )

    launch_dashboard(runs_root=tmp_path, host="127.0.0.1", port=8000)

    assert [config.strategy for config in observed_configs] == ["momentum"]
    assert observed_configs[0].benchmark_code == DEFAULT_LAUNCH_CONFIG.strategies[0].benchmark.code
    assert observed_configs[0].benchmark_name == DEFAULT_LAUNCH_CONFIG.strategies[0].benchmark.name
    assert observed_configs[0].warmup_days == DEFAULT_LAUNCH_CONFIG.strategies[0].warmup.extra_days
    assert captured_defaults["run_ids"] == ["momentum_20260405_120000"]


def test_launch_dashboard_passes_universe_id_to_backtest_runner(tmp_path: Path, monkeypatch) -> None:
    observed_configs = []
    kosdaq_preset = replace(DEFAULT_LAUNCH_CONFIG.strategies[0], universe_id="kosdaq150")
    launch_config = replace(DEFAULT_LAUNCH_CONFIG, strategies=(kosdaq_preset,))

    class FakeRunner:
        def run(self, config):
            observed_configs.append(config)
            return type("Report", (), {"output_dir": tmp_path / f"{config.strategy}_20260405_120000"})()

    monkeypatch.setattr("dashboard.run.build_frontend", lambda frontend_dir: None)
    monkeypatch.setattr("dashboard.run.DEFAULT_LAUNCH_CONFIG", launch_config)
    monkeypatch.setattr("dashboard.run.BacktestRunner", lambda result_dir=None: FakeRunner())
    monkeypatch.setattr(
        "dashboard.run.create_app",
        lambda default_selected_run_ids, frontend_dist=None: object(),
    )
    monkeypatch.setattr("dashboard.run.uvicorn.run", lambda app, host, port: None)
    monkeypatch.setattr(
        "dashboard.run.LaunchResolutionService.resolve",
        lambda self, config: type(
            "Plan",
            (),
            {
                "resolved_runs": (type("ResolvedRun", (), {"run_id": "momentum_20260405_100000", "strategy_name": "momentum"})(),),
                "missing_presets": (config.strategies[0],),
                "selected_run_ids": ["momentum_20260405_100000"],
            },
        )(),
    )

    launch_dashboard(runs_root=tmp_path, host="127.0.0.1", port=8000)

    assert observed_configs[0].universe_id == "kosdaq150"
    assert observed_configs[0].use_k200 is False


def test_build_frontend_runs_npm_build_without_install_when_lockfile_matches(tmp_path: Path, monkeypatch) -> None:
    observed_commands = []
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir()
    package_lock = frontend_dir / "package-lock.json"
    node_modules = frontend_dir / "node_modules"
    package_lock.write_text('{"lockfileVersion":3}', encoding="utf-8")
    node_modules.mkdir()
    (node_modules / ".package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")

    def fake_run(command: list[str], *, cwd: Path, check: bool) -> None:
        observed_commands.append((command, cwd, check))

    monkeypatch.setattr("dashboard.run.subprocess.run", fake_run)

    build_frontend(frontend_dir)

    assert observed_commands == [
        ([EXPECTED_NPM_COMMAND, "run", "build"], frontend_dir, True),
    ]


def test_build_frontend_installs_dependencies_when_node_modules_are_missing(tmp_path: Path, monkeypatch) -> None:
    observed_commands = []
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir()
    (frontend_dir / "package.json").write_text('{"name":"dashboard"}', encoding="utf-8")
    (frontend_dir / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")

    def fake_run(command: list[str], *, cwd: Path, check: bool) -> None:
        observed_commands.append((command, cwd, check))

    monkeypatch.setattr("dashboard.run.subprocess.run", fake_run)

    build_frontend(frontend_dir)

    assert observed_commands == [
        ([EXPECTED_NPM_COMMAND, "ci"], frontend_dir, True),
        ([EXPECTED_NPM_COMMAND, "run", "build"], frontend_dir, True),
    ]


def test_build_frontend_reinstalls_when_lockfile_is_newer_than_install_marker(tmp_path: Path, monkeypatch) -> None:
    observed_commands = []
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir()
    package_lock = frontend_dir / "package-lock.json"
    node_modules = frontend_dir / "node_modules"
    node_modules.mkdir()
    install_marker = node_modules / ".package-lock.json"
    install_marker.write_text('{"lockfileVersion":2}', encoding="utf-8")
    stale_time = time.time() - 60
    os.utime(install_marker, (stale_time, stale_time))
    package_lock.write_text('{"lockfileVersion":3}', encoding="utf-8")

    def fake_run(command: list[str], *, cwd: Path, check: bool) -> None:
        observed_commands.append((command, cwd, check))

    monkeypatch.setattr("dashboard.run.subprocess.run", fake_run)

    build_frontend(frontend_dir)

    assert observed_commands == [
        ([EXPECTED_NPM_COMMAND, "ci"], frontend_dir, True),
        ([EXPECTED_NPM_COMMAND, "run", "build"], frontend_dir, True),
    ]


def test_build_frontend_retries_after_clearing_corrupt_node_modules(tmp_path: Path, monkeypatch) -> None:
    observed_commands = []
    removed_paths = []
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir()
    node_modules = frontend_dir / "node_modules"
    node_modules.mkdir()
    (frontend_dir / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")

    install_error = subprocess.CalledProcessError(returncode=190, cmd=["npm", "ci"])

    def fake_run(command: list[str], *, cwd: Path, check: bool) -> None:
        observed_commands.append((command, cwd, check))
        if len(observed_commands) == 1:
            raise install_error

    def fake_rmtree(path: Path, ignore_errors: bool = False) -> None:
        removed_paths.append((path, ignore_errors))

    monkeypatch.setattr("dashboard.run.subprocess.run", fake_run)
    monkeypatch.setattr("dashboard.run.shutil.rmtree", fake_rmtree)

    build_frontend(frontend_dir)

    assert observed_commands == [
        ([EXPECTED_NPM_COMMAND, "ci"], frontend_dir, True),
        ([EXPECTED_NPM_COMMAND, "ci"], frontend_dir, True),
        ([EXPECTED_NPM_COMMAND, "run", "build"], frontend_dir, True),
    ]
    assert removed_paths == [(node_modules, True)]


def test_build_parser_prints_dashboard_help(capsys) -> None:
    parser = build_parser()

    parser.print_help()

    assert "Launch the 1W1A dashboard" in capsys.readouterr().out


def test_run_script_help_works_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [sys.executable, "dashboard/run.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Launch the 1W1A dashboard" in result.stdout
