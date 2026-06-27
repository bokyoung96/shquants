from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from root import ROOT
from scripts.run_tech_gamma_breakout_grid import _dated_config, _load_grid_inputs
from scripts.run_tech_gamma_long_only import TechGammaConfig
from scripts.tech_gamma_breakout_grid_simulation import base_entry_candidates, daily_frame, simulate_grid_continuation
from scripts.tech_gamma_is_oos_report import (
    daily_returns_from_trades,
    metrics_table,
    monthly_heatmap,
    rolling_metrics,
    write_report_plots,
)
from scripts.tech_gamma_research_filters import apply_research_features


DEFAULT_BEST = ROOT.results_path / "tech_gamma_breakout_grid_full" / "20260627_124825" / "best_strategy.json"
RESULT_DIR = ROOT.results_path / "tech_gamma_best_is_oos"


def config_from_best(path: Path) -> TechGammaConfig:
    row = json.loads(path.read_text(encoding="utf-8"))
    return TechGammaConfig(
        scheme="52w_high_breakout",
        universe="kospi200_historical",
        start="2021-01-01",
        end="2026-12-31 23:59:59",
        range_end_hhmm=str(row["range_end_hhmm"]).zfill(4),
        exit_hhmm=str(row["exit_hhmm"]).zfill(4),
        stop_bps=float(row["stop_bps"]),
        trailing_bps=float(row["trailing_bps"]),
        holding_mode="continuation",
        min_holding_days=int(row["min_holding_days"]),
        use_positivity=True,
        positivity_lookback_days=int(row["positivity_lookback_days"]),
        min_daily_positivity=0.0,
        positivity_benchmark=str(row["positivity_benchmark"]),
        positivity_margin=float(row["positivity_margin"]),
        factor_filter=str(row["factor_filter"]),
        factor_lookback_days=int(row["factor_lookback_days"]),
        atr_stop_multiplier=float(row["atr_stop_multiplier"]),
        range_buffer_bps=float(row["range_buffer_bps"]),
        overnight_enabled=False,
        high_lookback_days=420,
    )


def run_best_is_oos(
    *,
    best_path: Path = DEFAULT_BEST,
    output_dir: Path = RESULT_DIR,
    start: str = "2021-01-01",
    end: str = "2026-12-31 23:59:59",
) -> Path:
    config = _dated_config(config_from_best(best_path), start, end)
    output = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    frame, data = _load_grid_inputs(specs=[_spec(config)], start=start, end=end)
    enriched = apply_research_features(frame, config, data)
    trades = simulate_grid_continuation(base_entry_candidates(enriched), daily_frame(enriched), config)
    daily_returns = daily_returns_from_trades(trades)
    equity = (1.0 + daily_returns.fillna(0.0)).cumprod().rename("equity")
    splits = {"pre_grid_2021_2023": ("2021-01-01", "2023-12-31"), "selected_window_2024_2026": ("2024-01-01", "2026-12-31")}
    metrics = metrics_table(daily_returns, splits)
    rolling = rolling_metrics(daily_returns)
    heatmap = monthly_heatmap(daily_returns)
    _write_outputs(output, config, trades, daily_returns, equity, metrics, rolling, heatmap, best_path)
    return output


def _write_outputs(
    output: Path,
    config: TechGammaConfig,
    trades: pd.DataFrame,
    daily_returns: pd.Series,
    equity: pd.Series,
    metrics: pd.DataFrame,
    rolling: pd.DataFrame,
    heatmap: pd.DataFrame,
    best_path: Path,
) -> None:
    trades.to_csv(output / "trades.csv", index=False)
    daily_returns.to_csv(output / "daily_returns.csv", index_label="date")
    equity.to_csv(output / "equity_curve.csv", index_label="date")
    metrics.to_csv(output / "is_oos_metrics.csv", index=False)
    rolling.to_csv(output / "rolling_metrics.csv", index=False)
    heatmap.to_csv(output / "monthly_return_heatmap.csv", index_label="year")
    write_report_plots(daily_returns, metrics, rolling, heatmap, output)
    payload = {"source_best_strategy": str(best_path), "config": asdict(config)}
    (output / "config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _spec(config: TechGammaConfig) -> object:
    from scripts.tech_gamma_breakout_grid_specs import BreakoutStrategySpec

    return BreakoutStrategySpec(name="best_strategy_is_oos", config=config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed best breakout strategy over extended IS/OOS windows.")
    parser.add_argument("--best-path", type=Path, default=DEFAULT_BEST)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-12-31 23:59:59")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_best_is_oos(best_path=args.best_path, output_dir=args.output_dir, start=args.start, end=args.end)
    print(output)


if __name__ == "__main__":
    main()
