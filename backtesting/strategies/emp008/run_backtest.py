from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backtesting.reporting.cli import ReportCli
from backtesting.run import BacktestRunner
from backtesting.specs.models import ExecutionSpec, ScheduleSpec, TargetWeightsSpec
from backtesting.strategies.emp008.run_weights import (
    configure_logging,
    timed,
    timestamp,
)


DEFAULT_WEIGHTS_NAME = "mfbt_emp008"
DEFAULT_NAME = "mfbt_emp008"


def build_target_weight_spec(
    *,
    name: str,
    weights_csv: Path,
    dates: tuple[str, ...],
    end: str,
    fill_mode: str,
    capital: float = 100_000_000.0,
    fee: float = 0.0,
    sell_tax: float = 0.0,
    slippage: float = 0.0,
    allow_fractional: bool = True,
) -> ExecutionSpec:
    if not dates:
        raise ValueError("target weights must contain at least one date")
    return ExecutionSpec(
        start=dates[0],
        end=end,
        capital=capital,
        name=name,
        target_weights=TargetWeightsSpec(kind="file", path=str(weights_csv)),
        schedule=ScheduleSpec(kind="custom_dates", name=None, dates=dates),
        fill_mode=fill_mode,
        fee=fee,
        sell_tax=sell_tax,
        slippage=slippage,
        use_k200=True,
        allow_fractional=allow_fractional,
        spec_source="emp008_cli",
    )


def backtest_summary(report: Any) -> dict[str, object]:
    return {
        "output_dir": None if report.output_dir is None else str(report.output_dir),
        "plan_source": None if report.execution_resolution is None else report.execution_resolution.get("plan_source"),
        "summary": report.summary,
        "config": asdict(report.config),
        "rows": int(len(report.result.equity)),
        "date_start": report.result.equity.index.min().date().isoformat(),
        "date_end": report.result.equity.index.max().date().isoformat(),
        "timing": report.timing,
    }


def default_active_weights_path(weights_csv: Path) -> Path:
    return weights_csv.parent / "active_weights.parquet"


def write_active_share(active_weights_parquet: Path, output_dir: Path | None = None) -> dict[str, str]:
    if not active_weights_parquet.exists():
        raise FileNotFoundError(f"missing active weights parquet: {active_weights_parquet}")
    active_weights = pd.read_parquet(active_weights_parquet).astype(float)
    active_weights.index = pd.to_datetime(active_weights.index)
    active_share = (active_weights.abs().sum(axis=1) * 0.5).rename("active_share").to_frame()
    active_share["active_share_pct"] = active_share["active_share"] * 100.0
    active_share.index.name = "date"

    destination = output_dir or active_weights_parquet.parent
    destination.mkdir(parents=True, exist_ok=True)
    parquet_path = destination / "active_share.parquet"
    csv_path = destination / "active_share.csv"
    active_share.to_parquet(parquet_path, engine="pyarrow")
    active_share.reset_index().assign(date=lambda frame: frame["date"].dt.strftime("%Y-%m-%d")).to_csv(csv_path, index=False)
    return {
        "active_share_parquet": str(parquet_path),
        "active_share_csv": str(csv_path),
    }


def active_share_summary(active_share_parquet: Path) -> dict[str, object]:
    active_share = pd.read_parquet(active_share_parquet).astype(float)
    active_share.index = pd.to_datetime(active_share.index)
    series = active_share["active_share_pct"]
    return {
        "rows": int(len(series)),
        "date_start": series.index.min().date().isoformat() if not series.empty else None,
        "date_end": series.index.max().date().isoformat() if not series.empty else None,
        "mean_pct": float(series.mean()) if not series.empty else None,
        "median_pct": float(series.median()) if not series.empty else None,
        "min_pct": float(series.min()) if not series.empty else None,
        "max_pct": float(series.max()) if not series.empty else None,
    }


def active_share_payload(weights_csv: Path, *, backtest_output_dir: Path | None = None) -> dict[str, object]:
    paths = write_active_share(default_active_weights_path(weights_csv))
    summary = active_share_summary(Path(paths["active_share_parquet"]))
    payload: dict[str, object] = {**paths, "summary": summary}
    if backtest_output_dir is not None:
        backtest_paths = write_active_share(
            default_active_weights_path(weights_csv),
            output_dir=backtest_output_dir / "series",
        )
        payload["backtest_active_share_parquet"] = backtest_paths["active_share_parquet"]
        payload["backtest_active_share_csv"] = backtest_paths["active_share_csv"]
    return payload


def resolve_run_output_dirs(
    *,
    output_root: Path,
    name: str,
    backtests_root: Path | None,
    reports_root: Path | None,
) -> tuple[Path, Path, Path]:
    run_root = output_root / name
    return (
        run_root,
        backtests_root or run_root / "backtests",
        reports_root or run_root / "reports",
    )


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    weights_csv = args.weights_csv or default_weights_csv(args.output_root, args.weights_name)
    dates = load_weight_dates(weights_csv)
    filtered_dates = filter_dates(dates, start=args.start, end=args.end)
    end = args.end or filtered_dates[-1]

    run_root, backtests_root, reports_root = resolve_run_output_dirs(
        output_root=args.output_root,
        name=args.name,
        backtests_root=args.backtests_root,
        reports_root=args.reports_root,
    )
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or logs_dir / f"backtest_{timestamp()}.log"
    logger = configure_logging(log_path, logger_name="mfbt_emp008_backtest")

    logger.info("MFBT EMP008 backtest run started")
    logger.info("weights_csv=%s", weights_csv)
    logger.info("start=%s end=%s rebalance_count=%s", filtered_dates[0], end, len(filtered_dates))
    logger.info(
        "conditions capital=%s fill_mode=%s fee=%s sell_tax=%s slippage=%s allow_fractional=%s",
        args.capital,
        args.fill_mode,
        args.fee,
        args.sell_tax,
        args.slippage,
        not args.no_fractional,
    )

    summary = {
        "name": args.name,
        "weights_csv": str(weights_csv),
        "start": filtered_dates[0],
        "end": end,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "log_file": str(log_path),
        "conditions": {
            "capital": args.capital,
            "fill_mode": args.fill_mode,
            "fee": args.fee,
            "sell_tax": args.sell_tax,
            "slippage": args.slippage,
            "allow_fractional": not args.no_fractional,
        },
    }

    try:
        with timed(logger, "backtest"):
            spec = build_target_weight_spec(
                name=args.name,
                weights_csv=weights_csv,
                dates=filtered_dates,
                end=end,
                fill_mode=args.fill_mode,
                capital=args.capital,
                fee=args.fee,
                sell_tax=args.sell_tax,
                slippage=args.slippage,
                allow_fractional=not args.no_fractional,
            )
            runner = BacktestRunner(result_dir=backtests_root, write_report_assets=False, profile=True)
            report = runner.run_spec(runner.resolve_spec(spec))
            summary["backtest"] = backtest_summary(report)
            logger.info("backtest_summary=%s", json.dumps(summary["backtest"], ensure_ascii=False))
            summary["active_share"] = active_share_payload(weights_csv, backtest_output_dir=Path(str(report.output_dir)))
            logger.info("active_share=%s", json.dumps(summary["active_share"], ensure_ascii=False))

        if not args.no_report:
            with timed(logger, "report"):
                run_id = Path(str(report.output_dir)).name
                payload = ReportCli(runs_root=backtests_root, reports_root=reports_root).run(
                    [
                        "--runs",
                        run_id,
                        "--name",
                        args.name,
                        "--kind",
                        "tearsheet",
                        "--title",
                        args.title or args.name,
                    ]
                )
                summary["report"] = payload
                logger.info("report_payload=%s", json.dumps(payload, ensure_ascii=False))

        summary_path = run_root / "backtest_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("summary_path=%s", summary_path)
        logger.info("MFBT EMP008 backtest run completed")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception:
        logger.exception("MFBT EMP008 backtest run failed")
        raise


def default_weights_csv(output_root: Path, weights_name: str) -> Path:
    return output_root / weights_name / "weights" / "target_weights.csv"


def load_weight_dates(weights_csv: Path) -> tuple[str, ...]:
    if not weights_csv.exists():
        raise FileNotFoundError(f"missing weights CSV: {weights_csv}")
    frame = pd.read_csv(weights_csv, index_col=0, nrows=0)
    if frame.empty and len(frame.columns) == 0:
        raise ValueError(f"target weights CSV has no symbol columns: {weights_csv}")
    index = pd.read_csv(weights_csv, index_col=0, usecols=[0]).index
    dates = tuple(pd.to_datetime(index).strftime("%Y-%m-%d"))
    if not dates:
        raise ValueError(f"target weights CSV has no rebalance rows: {weights_csv}")
    return dates


def filter_dates(dates: tuple[str, ...], *, start: str | None, end: str | None) -> tuple[str, ...]:
    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None
    filtered = tuple(
        date
        for date in dates
        if (start_ts is None or pd.Timestamp(date) >= start_ts)
        and (end_ts is None or pd.Timestamp(date) <= end_ts)
    )
    if not filtered:
        raise ValueError("no target weight dates remain after applying start/end filters")
    return filtered


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest/report an existing MFBT EMP008 target_weights.csv.")
    parser.add_argument("--weights-csv", type=Path, help="Existing target_weights.csv. Overrides --weights-name.")
    parser.add_argument("--weights-name", default=DEFAULT_WEIGHTS_NAME, help=f"Weights run under --output-root. Default: {DEFAULT_WEIGHTS_NAME}")
    parser.add_argument("--name", default=DEFAULT_NAME, help=f"Backtest/report run name. Default: {DEFAULT_NAME}")
    parser.add_argument("--title", help="Report title. Default: run name.")
    parser.add_argument("--start", help="Optional backtest start date. Default: first target weight date.")
    parser.add_argument("--end", help="Optional backtest end date. Default: last target weight date.")
    parser.add_argument("--capital", type=float, default=100_000_000.0)
    parser.add_argument("--fill-mode", choices=("close", "next_open"), default="close")
    parser.add_argument("--fee", type=float, default=0.0)
    parser.add_argument("--sell-tax", type=float, default=0.0)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--no-fractional", action="store_true")
    parser.add_argument("--no-report", action="store_true", help="Skip report generation.")
    parser.add_argument("--output-root", type=Path, default=Path("results") / "emp008_runs")
    parser.add_argument("--backtests-root", type=Path, help="Backtest output root. Default: <output-root>/<name>/backtests.")
    parser.add_argument("--reports-root", type=Path, help="Report output root. Default: <output-root>/<name>/reports.")
    parser.add_argument("--log-file", type=Path)
    return parser


if __name__ == "__main__":
    main()
