"""Single-file operator entrypoint for the EMP008 research handoff.

Edit the ``SETTINGS`` block at the bottom, then run ``python run.py``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from backtest.run_backtest import run_backtest
from backtest.spec import BacktestSpec
from data.convert import convert_required
from emp008.data import Emp008Config
from emp008.run_weights import build_emp008_config, write_target_weights_csv
from emp008.strategy import run_emp008


@dataclass(frozen=True, slots=True)
class RunSettings:
    start: str = "2020-01-31"
    end: str | None = "2020-03-31"
    factor_set: str = "production_core"
    sector_neutral_dataset: str = "wi26"
    factor_timing: str = "none"
    tracking_error_annual: float | None = None
    convert_raw_to_parquet: bool = False
    run_backtest: bool = True
    fill_mode: str = "close"
    capital: float = 100_000_000.0
    fee: float = 0.0
    sell_tax: float = 0.0
    slippage: float = 0.0
    borrow_fee_annual: float = 0.0
    short_cash_collateral_ratio: float = 1.0
    allow_fractional: bool = True
    run_name: str = "emp008_run"


def run_settings(settings: RunSettings, *, project_dir: Path | None = None) -> dict[str, object]:
    project = Path(project_dir or Path(__file__).resolve().parent)
    _validate(settings)
    raw_dir = project / "data" / "raw"
    parquet_dir = project / "data" / "parquet"
    run_dir = project / "results" / settings.run_name
    weights_dir = run_dir / "weights"
    backtest_dir = run_dir / "backtest"

    config: Emp008Config = build_emp008_config(
        tracking_error_annual=settings.tracking_error_annual,
        factor_set=settings.factor_set,
        sector_neutral_dataset=settings.sector_neutral_dataset,
        factor_timing=settings.factor_timing,
    )
    converted: dict[str, str] = {}
    if settings.convert_raw_to_parquet:
        converted = convert_required(raw_dir=raw_dir, parquet_dir=parquet_dir, config=config)

    end = settings.end or _latest_end(parquet_dir, config)
    result = run_emp008(
        parquet_dir=parquet_dir,
        start=settings.start,
        end=end,
        config=config,
        output_dir=weights_dir,
    )
    weights_csv = write_target_weights_csv(result.target_weights, weights_dir / "target_weights.csv")

    backtest_summary: dict[str, object] | None = None
    if settings.run_backtest:
        report = run_backtest(
            BacktestSpec(
                name=settings.run_name,
                data_dir=parquet_dir,
                output_dir=backtest_dir,
                weights_csv=weights_csv,
                start=settings.start,
                end=end,
                capital=settings.capital,
                fill_mode=settings.fill_mode,
                allow_fractional=settings.allow_fractional,
                fee=settings.fee,
                sell_tax=settings.sell_tax,
                slippage=settings.slippage,
                borrow_fee_annual=settings.borrow_fee_annual,
                short_cash_collateral_ratio=settings.short_cash_collateral_ratio,
            )
        )
        backtest_summary = report.summary

    summary = {
        "settings": asdict(settings),
        "paths": {
            "raw_dir": str(raw_dir),
            "parquet_dir": str(parquet_dir),
            "weights_dir": str(weights_dir),
            "weights_csv": str(weights_csv),
            "backtest_dir": str(backtest_dir) if settings.run_backtest else None,
        },
        "converted": converted,
        "weights": {
            "rows": int(result.target_weights.shape[0]),
            "symbols": int(result.target_weights.shape[1]),
            "diagnostics_success_all": bool(result.diagnostics["success"].all()),
        },
        "backtest": backtest_summary,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _validate(settings: RunSettings) -> None:
    start = pd.Timestamp(settings.start)
    if settings.end is not None and pd.Timestamp(settings.end) < start:
        raise ValueError("end must be on or after start")
    if settings.run_name.strip() == "":
        raise ValueError("run_name must not be empty")
    if settings.sector_neutral_dataset not in {"wi26", "wics"}:
        raise ValueError("sector_neutral_dataset must be 'wi26' or 'wics'")


def _latest_end(parquet_dir: Path, config: Emp008Config) -> str:
    from emp008.run_weights import latest_common_end

    return latest_common_end(parquet_dir, config)


# Operator settings: edit these values, then run `uv run python run.py`.
SETTINGS = RunSettings(
    start="2020-01-31",
    end="2020-03-31",
    factor_set="production_core",
    sector_neutral_dataset="wi26",
    factor_timing="none",
    convert_raw_to_parquet=False,
    run_backtest=True,
    fill_mode="close",
    capital=100_000_000.0,
    fee=0.0,
    sell_tax=0.0,
    slippage=0.0,
    run_name="emp008_run",
)


if __name__ == "__main__":
    run_settings(SETTINGS)
