from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from backtesting.run import BacktestRunner, RunConfig
from dashboard.backend.main import create_app
from dashboard.backend.services.launch_resolution import LaunchResolutionService
from dashboard.strategies import DEFAULT_LAUNCH_CONFIG, StrategyPreset
from root import ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--runs-root", type=Path, default=ROOT.results_path / "backtests")
    parser.add_argument("--frontend-dir", type=Path, default=Path(__file__).resolve().parent / "frontend")
    return parser


def build_frontend(frontend_dir: Path) -> None:
    npm_command = _resolve_npm_command()
    if _needs_npm_install(frontend_dir):
        _install_frontend_dependencies(frontend_dir, npm_command=npm_command)
    subprocess.run([npm_command, "run", "build"], cwd=frontend_dir, check=True)


def launch_dashboard(
    *,
    runs_root: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    frontend_dir: Path | None = None,
) -> None:
    resolved_runs_root = runs_root or (ROOT.results_path / "backtests")
    resolved_frontend_dir = frontend_dir or (Path(__file__).resolve().parent / "frontend")
    build_frontend(resolved_frontend_dir)

    plan = LaunchResolutionService(resolved_runs_root).resolve(DEFAULT_LAUNCH_CONFIG)
    runner = BacktestRunner(result_dir=resolved_runs_root, write_report_assets=False, profile=True)
    selected_run_ids = list(plan.selected_run_ids)

    for preset in plan.missing_presets:
        report = runner.run(_build_run_config(preset))
        if report.output_dir is None:
            continue
        selected_run_ids.append(report.output_dir.name)

    app = create_app(
        default_selected_run_ids=selected_run_ids,
        frontend_dist=resolved_frontend_dir / "dist",
    )
    uvicorn.run(app, host=host, port=port)


def _build_run_config(preset: StrategyPreset) -> RunConfig:
    config = DEFAULT_LAUNCH_CONFIG.global_config
    use_k200 = config.use_k200 if preset.universe_id in (None, "legacy_k200") else False
    return RunConfig(
        start=config.start,
        end=config.end,
        capital=config.capital,
        strategy=preset.strategy_name,
        name=preset.display_label,
        schedule=preset.schedule or config.schedule,
        fill_mode=preset.fill_mode or config.fill_mode,
        fee=config.fee,
        sell_tax=config.sell_tax,
        slippage=config.slippage,
        use_k200=use_k200,
        allow_fractional=config.allow_fractional,
        universe_id=preset.universe_id,
        benchmark_code=preset.benchmark.code,
        benchmark_name=preset.benchmark.name,
        benchmark_dataset=preset.benchmark.dataset,
        warmup_days=preset.warmup.extra_days,
        **dict(preset.params),
    )


def _needs_npm_install(frontend_dir: Path) -> bool:
    node_modules = frontend_dir / "node_modules"
    if not node_modules.is_dir():
        return True

    package_lock = frontend_dir / "package-lock.json"
    install_marker = node_modules / ".package-lock.json"
    if package_lock.is_file():
        if not install_marker.is_file():
            return True
        return install_marker.stat().st_mtime < package_lock.stat().st_mtime

    return False


def _resolve_npm_command() -> str:
    candidates = ["npm.cmd", "npm"] if sys.platform.startswith("win") else ["npm"]
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
    raise FileNotFoundError("npm executable not found. Install Node.js and ensure npm is available on PATH.")


def _install_frontend_dependencies(frontend_dir: Path, *, npm_command: str) -> None:
    install_command = (
        [npm_command, "ci"]
        if (frontend_dir / "package-lock.json").is_file()
        else [npm_command, "install"]
    )

    try:
        subprocess.run(install_command, cwd=frontend_dir, check=True)
    except subprocess.CalledProcessError:
        node_modules = frontend_dir / "node_modules"
        if not node_modules.exists():
            raise

        shutil.rmtree(node_modules, ignore_errors=True)
        subprocess.run(install_command, cwd=frontend_dir, check=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    launch_dashboard(
        runs_root=args.runs_root,
        host=args.host,
        port=args.port,
        frontend_dir=args.frontend_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
