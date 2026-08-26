"""Run every canonical EMP008 factor set under one shared backtest contract."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from backtest.run_backtest import run_backtest
from backtest.spec import BacktestSpec
from data.catalog import DataCatalog
from data.catalog import DatasetId
from data.convert import convert_dataset
from emp008.data import required_datasets
from emp008.factor_registry import get_factor_set_definition, strategy_factor_set_values
from emp008.run_weights import build_emp008_config, write_target_weights_csv
from emp008.strategy import run_emp008
from emp008.data import Emp008Config


START = "2019-12-30"
END = "2026-06-30"
CAPITAL = 100_000_000.0
FILL_MODE = "close"
FEE = 0.0002
SELL_TAX = 0.0015
SLIPPAGE = 0.0005
ALLOW_FRACTIONAL = True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sector-neutral-dataset",
        choices=("wi26", "wics"),
        default="wi26",
    )
    return parser


def resolve_sector_run(project: Path, sector_neutral_dataset: str) -> tuple[DatasetId | None, Path]:
    config = build_emp008_config(sector_neutral_dataset=sector_neutral_dataset)
    label = sector_neutral_dataset.strip().upper()
    return config.sector_neutral_dataset, project / "results" / label / "all_models"


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    project = PACKAGE_ROOT
    data_dir = project / "data" / "parquet"
    raw_dir = project / "data" / "raw"
    sector_neutral_dataset, results_dir = resolve_sector_run(project, args.sector_neutral_dataset)
    results_dir.mkdir(parents=True, exist_ok=True)
    factor_sets = list(strategy_factor_set_values())
    direction_constraints = {
        factor_set: get_factor_set_definition(factor_set).constrain_expected_alpha_to_direction
        for factor_set in factor_sets
    }
    if not all(direction_constraints.values()):
        missing = [name for name, enabled in direction_constraints.items() if not enabled]
        raise RuntimeError(f"direction constraints must be enabled for every portfolio: {missing}")
    ensure_all_model_data(
        raw_dir=raw_dir,
        data_dir=data_dir,
        factor_sets=factor_sets,
        sector_neutral_dataset=sector_neutral_dataset,
    )
    manifest = {
        "start": START,
        "end": END,
        "sector_neutral_dataset": args.sector_neutral_dataset,
        "capital": CAPITAL,
        "fill_mode": FILL_MODE,
        "fee": FEE,
        "sell_tax": SELL_TAX,
        "slippage": SLIPPAGE,
        "allow_fractional": ALLOW_FRACTIONAL,
        "weight_policy": "equal_weight_within_factor_set",
        "expected_alpha_direction_constraint": True,
        "factor_sets": factor_sets,
    }
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summaries: list[dict[str, object]] = []
    for index, factor_set in enumerate(factor_sets, start=1):
        print(f"[{index}/{len(factor_sets)}] {factor_set}", flush=True)
        summaries.append(
            run_one(
                factor_set,
                data_dir=data_dir,
                results_dir=results_dir,
                sector_neutral_dataset=sector_neutral_dataset,
            )
        )
        pd.DataFrame(summaries).to_csv(results_dir / "comparison_summary.csv", index=False)

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def ensure_all_model_data(
    *,
    raw_dir: Path,
    data_dir: Path,
    factor_sets: list[str],
    sector_neutral_dataset: DatasetId | None,
) -> None:
    dataset_ids = set()
    for factor_set in factor_sets:
        dataset_ids.update(
            required_datasets(
                Emp008Config(
                    factor_set=factor_set,
                    sector_neutral_dataset=sector_neutral_dataset,
                )
            )
        )
    catalog = DataCatalog.default()
    for dataset_id in sorted(dataset_ids, key=str):
        path = data_dir / f"{catalog.get(dataset_id).stem}.parquet"
        if not path.exists():
            print(f"converting {dataset_id.value}", flush=True)
            convert_dataset(raw_dir=raw_dir, parquet_dir=data_dir, dataset_id=dataset_id)


def run_one(
    factor_set: str,
    *,
    data_dir: Path,
    results_dir: Path,
    sector_neutral_dataset: DatasetId | None,
) -> dict[str, object]:
    run_dir = results_dir / factor_set
    weights_dir = run_dir / "weights"
    backtest_dir = run_dir / "backtest"
    weights_dir.mkdir(parents=True, exist_ok=True)

    config = Emp008Config(
        factor_set=factor_set,
        sector_neutral_dataset=sector_neutral_dataset,
    )
    result = run_emp008(
        parquet_dir=data_dir,
        start=START,
        end=END,
        config=config,
        output_dir=weights_dir,
    )
    weights_csv = write_target_weights_csv(result.target_weights, weights_dir / "target_weights.csv")
    report = run_backtest(
        BacktestSpec(
            name=factor_set,
            data_dir=data_dir,
            output_dir=backtest_dir,
            weights_csv=weights_csv,
            start=START,
            end=END,
            capital=CAPITAL,
            fill_mode=FILL_MODE,
            allow_fractional=ALLOW_FRACTIONAL,
            fee=FEE,
            sell_tax=SELL_TAX,
            slippage=SLIPPAGE,
        )
    )
    equity = report.result.equity.astype(float)
    drawdown = equity.div(equity.cummax()).sub(1.0)
    plots_dir = run_dir / "plots"
    save_plot(
        equity.div(equity.iloc[0]).sub(1.0),
        plots_dir / "cumulative_return.png",
        title=f"EMP008 {factor_set} cumulative return",
        ylabel="Cumulative return",
    )
    save_plot(
        drawdown,
        plots_dir / "drawdown.png",
        title=f"EMP008 {factor_set} drawdown",
        ylabel="Drawdown",
    )
    metrics = calculate_metrics(report.result.equity, report.result.returns)
    summary = {
        "factor_set": factor_set,
        "factor_count": len(get_factor_set_definition(factor_set).factors),
        "weight_rows": int(result.target_weights.shape[0]),
        "weight_symbols": int(result.target_weights.shape[1]),
        "weights_dir": str(weights_dir),
        "backtest_dir": str(backtest_dir),
        "plots_dir": str(plots_dir),
        **metrics,
    }
    (run_dir / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def calculate_metrics(equity: pd.Series, returns: pd.Series) -> dict[str, float]:
    equity = equity.astype(float)
    returns = returns.astype(float).dropna()
    cumulative = (1.0 + returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0) if not cumulative.empty else 0.0
    years = len(returns) / 252.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    volatility = float(returns.std(ddof=0) * np.sqrt(252.0)) if not returns.empty else 0.0
    sharpe = float((returns.mean() * 252.0) / volatility) if volatility > 0 else 0.0
    drawdown = cumulative.div(cumulative.cummax()).sub(1.0) if not cumulative.empty else pd.Series(dtype=float)
    return {
        "final_equity": float(equity.iloc[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
    }


def save_plot(series: pd.Series, path: Path, *, title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 6), dpi=160)
    axis.plot(series.index, series.to_numpy(), linewidth=1.4)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, format="png")
    plt.close(figure)


if __name__ == "__main__":
    main()
