from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtesting.run import BacktestRunner
from backtesting.strategies.emp008.reports.model_comparison import (
    ModelReportInput,
    build_emp008_model_comparison_report,
)
from backtesting.strategies.emp008.run_backtest import build_target_weight_spec


def common_weight_dates(modified_weights_csv: Path, original_weights_csv: Path) -> tuple[str, ...]:
    modified = _weight_dates(modified_weights_csv)
    original = _weight_dates(original_weights_csv)
    common = modified.intersection(original).sort_values()
    if len(common) < 2:
        raise ValueError("mfbt and origin need at least two common rebalance dates")
    return tuple(common.strftime("%Y-%m-%d"))


def build_comparison_from_run_roots(
    *,
    modified_run_root: Path,
    original_run_root: Path,
    output_dir: Path,
    adjusted_close_path: Path = Path("parquet/qw_adj_c.parquet"),
    sector_path: Path = Path("parquet/qw_wics_sec_big.parquet"),
    capital: float = 100_000_000.0,
    fee: float = 0.0002,
    sell_tax: float = 0.0015,
    slippage: float = 0.0005,
) -> dict[str, object]:
    modified_summary = _read_run_summary(modified_run_root, expected_factor_set="mfbt")
    original_summary = _read_run_summary(original_run_root, expected_factor_set="origin")
    modified_weights = modified_run_root / "weights" / "target_weights.csv"
    original_weights = original_run_root / "weights" / "target_weights.csv"
    dates = common_weight_dates(modified_weights, original_weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    backtests_root = output_dir / "backtests"

    modified_gross = _run_backtest(
        name="modified_emp008_gross",
        weights_csv=modified_weights,
        dates=dates,
        output_root=backtests_root,
        capital=capital,
        fee=0.0,
        sell_tax=0.0,
        slippage=0.0,
    )
    original_gross = _run_backtest(
        name="original_emp008_gross",
        weights_csv=original_weights,
        dates=dates,
        output_root=backtests_root,
        capital=capital,
        fee=0.0,
        sell_tax=0.0,
        slippage=0.0,
    )
    modified_net = _run_backtest(
        name="modified_emp008_net",
        weights_csv=modified_weights,
        dates=dates,
        output_root=backtests_root,
        capital=capital,
        fee=fee,
        sell_tax=sell_tax,
        slippage=slippage,
    )
    original_net = _run_backtest(
        name="original_emp008_net",
        weights_csv=original_weights,
        dates=dates,
        output_root=backtests_root,
        capital=capital,
        fee=fee,
        sell_tax=sell_tax,
        slippage=slippage,
    )

    payload = build_emp008_model_comparison_report(
        modified=_model_input("수정EMP008", modified_summary, modified_gross, modified_net),
        original=_model_input("기존EMP008", original_summary, original_gross, original_net),
        adjusted_close_path=adjusted_close_path,
        sector_path=sector_path,
        output_dir=output_dir,
        cost_assumptions={"fee": fee, "sell_tax": sell_tax, "slippage": slippage},
    )
    payload["backtests"] = {
        "modified_gross": str(modified_gross),
        "modified_net": str(modified_net),
        "original_gross": str(original_gross),
        "original_net": str(original_net),
    }
    return payload


def _weight_dates(path: Path) -> pd.DatetimeIndex:
    if not path.exists():
        raise FileNotFoundError(f"missing target weights: {path}")
    index = pd.to_datetime(pd.read_csv(path, index_col=0, usecols=[0]).index)
    if index.has_duplicates:
        raise ValueError(f"duplicate target-weight dates: {path}")
    return pd.DatetimeIndex(index)


def _read_run_summary(run_root: Path, *, expected_factor_set: str) -> dict[str, object]:
    path = run_root / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing EMP008 run summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = payload.get("factor_set")
    if actual != expected_factor_set:
        raise ValueError(f"expected factor_set={expected_factor_set!r}, got {actual!r} in {path}")
    return payload


def _run_backtest(
    *,
    name: str,
    weights_csv: Path,
    dates: tuple[str, ...],
    output_root: Path,
    capital: float,
    fee: float,
    sell_tax: float,
    slippage: float,
) -> Path:
    spec = build_target_weight_spec(
        name=name,
        weights_csv=weights_csv,
        dates=dates,
        end=dates[-1],
        fill_mode="close",
        capital=capital,
        fee=fee,
        sell_tax=sell_tax,
        slippage=slippage,
        allow_fractional=True,
    )
    runner = BacktestRunner(result_dir=output_root, write_report_assets=False, profile=True)
    report = runner.run_spec(runner.resolve_spec(spec))
    if report.output_dir is None:
        raise RuntimeError(f"backtest did not produce an output directory: {name}")
    return Path(str(report.output_dir))


def _model_input(
    label: str,
    summary: dict[str, object],
    gross_run_dir: Path,
    net_run_dir: Path,
) -> ModelReportInput:
    return ModelReportInput(
        label=label,
        factor_set=str(summary["factor_set"]),
        risk_model=str(summary["risk_model"]),
        tracking_error_annual=float(summary["tracking_error_annual"]),
        gross_run_dir=gross_run_dir,
        net_run_dir=net_run_dir,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    payload = build_comparison_from_run_roots(
        modified_run_root=args.modified_run_root,
        original_run_root=args.original_run_root,
        output_dir=args.output_dir,
        adjusted_close_path=args.adjusted_close,
        sector_path=args.sector,
        capital=args.capital,
        fee=args.fee,
        sell_tax=args.sell_tax,
        slippage=args.slippage,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the cost-aligned mfbt/origin EMP008 comparison report.")
    parser.add_argument("--modified-run-root", type=Path, required=True)
    parser.add_argument("--original-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adjusted-close", type=Path, default=Path("parquet/qw_adj_c.parquet"))
    parser.add_argument("--sector", type=Path, default=Path("parquet/qw_wics_sec_big.parquet"))
    parser.add_argument("--capital", type=float, default=100_000_000.0)
    parser.add_argument("--fee", type=float, default=0.0002)
    parser.add_argument("--sell-tax", type=float, default=0.0015)
    parser.add_argument("--slippage", type=float, default=0.0005)
    return parser


if __name__ == "__main__":
    main()


__all__ = ["build_comparison_from_run_roots", "common_weight_dates", "main"]
