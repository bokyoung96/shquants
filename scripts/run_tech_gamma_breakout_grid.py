from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from root import ROOT
from backtesting.data.kr_stock_5m import KrStock5mDataset, read_tickers_bars
from scripts.run_tech_gamma_long_only import TechGammaConfig, build_features
from scripts.tech_gamma_breakout_grid_simulation import (
    base_entry_candidates,
    daily_frame,
    rank_grid_summary,
    simulate_grid_continuation,
    strategy_summary,
)
from scripts.tech_gamma_breakout_grid_specs import BreakoutStrategySpec, FeatureKey, build_strategy_specs, feature_key
from scripts.tech_gamma_research_filters import apply_research_features, load_research_feature_data
from scripts.tech_gamma_universe import filter_kospi200_historical_members, kospi200_tickers


RESULT_DIR = ROOT.results_path / "tech_gamma_breakout_grid"
_base_entry_candidates = base_entry_candidates
_daily_frame = daily_frame
_simulate_continuation = simulate_grid_continuation


def run_grid(
    *,
    output_dir: Path = RESULT_DIR,
    max_strategies: int = 5000,
    start: str = "2024-01-01",
    end: str = "2026-12-31 23:59:59",
) -> pd.DataFrame:
    run_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, int | float | str]] = []
    specs = build_strategy_specs(max_strategies=max_strategies)
    frame, data = _load_grid_inputs(specs=specs, start=start, end=end)
    groups = _group_by_features(specs)
    for group_index, group in enumerate(groups.values(), start=1):
        feature_config = _dated_config(group[0].config, start, end)
        enriched = apply_research_features(frame, feature_config, data)
        daily = daily_frame(enriched)
        base_candidates = base_entry_candidates(enriched)
        for spec in group:
            config = _dated_config(spec.config, start, end)
            trades = simulate_grid_continuation(base_candidates, daily, config)
            summary_rows.append(strategy_summary(spec.name, config, trades))
        pd.DataFrame(summary_rows).to_csv(run_dir / "grid_summary_partial.csv", index=False)
        if group_index % 10 == 0 or group_index == len(groups):
            print(f"completed_feature_groups={group_index}/{len(groups)} strategies={len(summary_rows)}/{len(specs)}", flush=True)
    summary_frame = rank_grid_summary(pd.DataFrame(summary_rows))
    summary_frame.to_csv(run_dir / "grid_summary.csv", index=False)
    summary_frame.head(50).to_csv(run_dir / "top_strategies.csv", index=False)
    (run_dir / "best_strategy.json").write_text(
        json.dumps(summary_frame.iloc[0].to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "grid_config.json").write_text(
        json.dumps({"max_strategies": max_strategies, "strategies": [asdict(spec) for spec in specs]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_frame


def _load_grid_inputs(
    *,
    specs: list[BreakoutStrategySpec],
    start: str,
    end: str,
) -> tuple[pd.DataFrame, object]:
    dataset = KrStock5mDataset(ROOT.parquet_path / "KR_STOCK_5m")
    base_config = _base_config(specs, start, end)
    tickers = kospi200_tickers(dataset.root.parent, base_config)
    raw = read_tickers_bars(dataset, tickers, start=_load_start(base_config), end=end)
    usable = raw.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
    frame = build_features(usable, base_config)
    if base_config.universe == "kospi200_historical":
        frame = filter_kospi200_historical_members(frame, dataset.root.parent)
    frame = frame.loc[frame["ts"].ge(pd.Timestamp(start))].reset_index(drop=True)
    data = load_research_feature_data(dataset.root.parent, tickers)
    return frame, data


def _base_config(specs: list[BreakoutStrategySpec], start: str, end: str) -> TechGammaConfig:
    values = asdict(specs[0].config)
    values["start"] = start
    values["end"] = end
    values["use_positivity"] = False
    values["factor_filter"] = "none"
    values["holding_mode"] = "intraday"
    values["high_lookback_days"] = max(spec.config.high_lookback_days for spec in specs)
    return TechGammaConfig(**values)


def _load_start(config: TechGammaConfig) -> pd.Timestamp:
    return pd.Timestamp(config.start) - pd.Timedelta(days=config.high_lookback_days)


def _group_by_features(specs: Iterable[BreakoutStrategySpec]) -> dict[FeatureKey, list[BreakoutStrategySpec]]:
    groups: dict[FeatureKey, list[BreakoutStrategySpec]] = {}
    for spec in specs:
        groups.setdefault(feature_key(spec.config), []).append(spec)
    return groups


def _dated_config(config: TechGammaConfig, start: str, end: str) -> TechGammaConfig:
    values = asdict(config)
    values["start"] = start
    values["end"] = end
    return TechGammaConfig(**values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structural 52-week breakout strategy grid without parameter fitting.")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-strategies", type=int, default=5000)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-12-31 23:59:59")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_grid(output_dir=args.output_dir, max_strategies=args.max_strategies, start=args.start, end=args.end)
    print(summary)


if __name__ == "__main__":
    main()
