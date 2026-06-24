from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from root import ROOT

from backtesting.data import ParquetStore
from backtesting.strategies.positivity_event import (
    EventQueueConfig,
    build_positivity_event_queue_strategy,
)
from scripts.run_pos_research import (
    DEFAULT_START,
    _next_day_benchmark_returns,
    _weighted_next_day_returns,
    summarize_quintile_returns,
)


RESULT_DIR = ROOT.results_path / "pos_research" / "event_alpha"


@dataclass(frozen=True, slots=True)
class EventStrategySpec:
    name: str
    params: dict[str, Any]


def build_event_strategy_specs() -> list[EventStrategySpec]:
    specs: list[EventStrategySpec] = []
    for max_positions in (1, 3, 5):
        for atr_multiplier in (2.0, 2.5, 3.0):
            for entry_mode in ("near_high", "breakout"):
                specs.append(
                    EventStrategySpec(
                        name=f"event_{entry_mode}_n{max_positions}_atr{str(atr_multiplier).replace('.', '')}",
                        params={
                            "max_positions": max_positions,
                            "positivity_lookback": 60,
                            "min_periods": 60,
                            "high_lookback": 252,
                            "atr_lookback": 20,
                            "atr_multiplier": atr_multiplier,
                            "relative_signal_groups": 3,
                            "entry_high_ratio": 0.95,
                            "exit_high_ratio": 0.90,
                            "replacement_margin": 0.25,
                            "entry_mode": entry_mode,
                            "exit_rank_group_count": 2,
                        },
                    )
                )
    return specs


def rank_event_summary(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    ranked["event_viable"] = ranked["event_count"].gt(0).astype(int)
    ranked["validation_pass"] = (
        ranked["late_active_alpha_cagr"].gt(0.0)
        & ranked["active_alpha_cagr"].gt(0.0)
        & ranked["event_viable"].eq(1)
    ).astype(int)
    ranked["robust_return"] = pd.concat(
        [ranked["active_alpha_cagr"], ranked["late_active_alpha_cagr"]],
        axis=1,
    ).min(axis=1)
    ranked["worst_mdd"] = pd.concat(
        [ranked["active_alpha_mdd"], ranked["late_active_alpha_mdd"]],
        axis=1,
    ).min(axis=1)
    drawdown_risk = ranked["worst_mdd"].abs().where(ranked["worst_mdd"].abs().gt(0.0))
    ranked["robust_score"] = ranked["robust_return"].div(drawdown_risk).replace(
        [float("inf"), -float("inf")],
        pd.NA,
    ).fillna(0.0)
    ranked["selection_score"] = ranked["robust_score"]
    return ranked.sort_values(
        [
            "event_viable",
            "validation_pass",
            "selection_score",
            "robust_return",
            "worst_mdd",
            "active_day_ratio",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)


def run_event_alpha_grid(
    *,
    start: str = DEFAULT_START,
    end: str | None = None,
    output_dir: Path = RESULT_DIR,
    data_root: Path | None = None,
) -> pd.DataFrame:
    specs = build_event_strategy_specs()
    store = ParquetStore(ROOT.parquet_path if data_root is None else data_root)
    start_ts = pd.Timestamp(start)
    warmup_start = _warmup_start_for_specs(start=start_ts, specs=specs)
    close = store.read("qw_adj_c").astype(float)
    high = store.read("qw_adj_h").astype(float)
    low = store.read("qw_adj_l").astype(float)
    membership = store.read("qw_k200_yn").reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    benchmark = store.read("qw_BM")

    end_ts = pd.Timestamp(end) if end is not None else close.index.max()
    close = close.loc[warmup_start:end_ts]
    common_index = close.index.intersection(high.index).intersection(low.index)
    common_columns = close.columns.intersection(high.columns).intersection(low.columns)
    close = close.loc[common_index, common_columns]
    high = high.reindex(index=common_index, columns=common_columns)
    low = low.reindex(index=common_index, columns=common_columns)
    membership = membership.reindex(index=common_index, columns=common_columns).fillna(False).astype(bool)
    entry_membership = _entry_membership_window(membership=membership, start=start_ts)

    stock_returns = close.pct_change(fill_method=None)
    benchmark_returns = _next_day_benchmark_returns(benchmark=benchmark, index=close.index).reindex(close.index)
    summary_rows: list[dict[str, Any]] = []
    all_trades: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []

    for spec in specs:
        config = EventQueueConfig(**spec.params)
        result = build_positivity_event_queue_strategy(
            close=close,
            high=high,
            low=low,
            membership=entry_membership,
            config=config,
        )
        strategy_returns = _weighted_next_day_returns(weights=result.weights, stock_returns=stock_returns).loc[start_ts:end_ts]
        active = result.weights.reindex(strategy_returns.index).fillna(0.0).sum(axis=1).gt(0.0)
        active_alpha = strategy_returns.sub(benchmark_returns.reindex(strategy_returns.index).where(active, 0.0))
        sleeve_perf = _single_summary(strategy_returns, prefix="sleeve")
        active_alpha_perf = _single_summary(active_alpha, prefix="active_alpha")
        split = _split_validation_metrics(active_alpha)
        names = result.weights.reindex(strategy_returns.index).fillna(0.0).gt(0.0).sum(axis=1)
        trades = _filter_trades(trades=result.trades, start=start_ts, end=end_ts)
        events = _filter_events(events=result.entry_events, start=start_ts, end=end_ts)
        holding_days = _average_holding_days(trades)
        row = {
            "strategy": spec.name,
            "event_count": int(len(events)),
            "trade_count": int(len(trades)),
            "trade_win_rate": float(trades["return"].gt(0.0).mean()) if not trades.empty else 0.0,
            "avg_trade_return": float(trades["return"].mean()) if not trades.empty else 0.0,
            "avg_holding_days": holding_days,
            "avg_names": float(names.mean()) if not names.empty else 0.0,
            "max_names": int(names.max()) if not names.empty else 0,
            "active_day_ratio": float(names.gt(0).mean()) if not names.empty else 0.0,
            **spec.params,
            **sleeve_perf,
            **active_alpha_perf,
            **split,
        }
        summary_rows.append(row)
        if not trades.empty:
            all_trades.append(trades.assign(strategy=spec.name))
        if not events.empty:
            all_events.append(events.assign(strategy=spec.name))

    summary = rank_event_summary(pd.DataFrame(summary_rows))
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    _write_event_outputs(
        output_dir=output_dir,
        summary=summary,
        start=pd.Timestamp(start),
        end_ts=end_ts,
        specs=specs,
        trades=trades,
        events=events,
    )
    return summary


def _entry_membership_window(*, membership: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    eligible = membership.copy()
    eligible.loc[eligible.index < start] = False
    return eligible


def _warmup_start_for_specs(*, start: pd.Timestamp, specs: list[EventStrategySpec]) -> pd.Timestamp:
    max_lookback = 0
    for spec in specs:
        params = spec.params
        max_lookback = max(
            max_lookback,
            int(params["positivity_lookback"]),
            int(params["high_lookback"]),
            int(params["atr_lookback"]),
        )
    return start - pd.offsets.BDay(max_lookback + 80)


def _single_summary(returns: pd.Series, *, prefix: str) -> dict[str, float | int]:
    clean = returns.dropna()
    if clean.empty:
        return {
            f"{prefix}_observations": 0,
            f"{prefix}_total_return": float("nan"),
            f"{prefix}_cagr": float("nan"),
            f"{prefix}_mdd": float("nan"),
            f"{prefix}_sharpe": float("nan"),
            f"{prefix}_daily_win_rate": float("nan"),
        }
    row = summarize_quintile_returns(pd.DataFrame({prefix: clean})).iloc[0]
    return {
        f"{prefix}_observations": int(row["observations"]),
        f"{prefix}_total_return": float(row["total_return"]),
        f"{prefix}_cagr": float(row["cagr"]),
        f"{prefix}_mdd": float(row["mdd"]),
        f"{prefix}_sharpe": float(row["sharpe"]),
        f"{prefix}_daily_win_rate": float(row["daily_win_rate"]),
    }


def _split_validation_metrics(returns: pd.Series) -> dict[str, float | int]:
    clean = returns.dropna()
    midpoint = len(clean) // 2
    early = clean.iloc[:midpoint]
    late = clean.iloc[midpoint:]
    return {
        **_single_summary(early, prefix="early_active_alpha"),
        **_single_summary(late, prefix="late_active_alpha"),
    }


def _average_holding_days(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    days = pd.to_datetime(trades["exit_date"]).sub(pd.to_datetime(trades["entry_date"])).dt.days
    return float(days.mean())


def _filter_trades(*, trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if trades.empty:
        return trades
    exit_dates = pd.to_datetime(trades["exit_date"])
    return trades.loc[exit_dates.between(start, end)].reset_index(drop=True)


def _filter_events(*, events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if events.empty:
        return events
    event_dates = pd.to_datetime(events["date"])
    return events.loc[event_dates.between(start, end)].reset_index(drop=True)


def _write_event_outputs(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    start: pd.Timestamp,
    end_ts: pd.Timestamp,
    specs: list[EventStrategySpec],
    trades: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "event_alpha_summary.csv", index=False)
    summary.head(10).to_csv(output_dir / "top10_event_alpha_summary.csv", index=False)
    if trades is not None and not trades.empty:
        trades.to_csv(output_dir / "event_trades.csv", index=False)
    if events is not None and not events.empty:
        events.to_csv(output_dir / "entry_events.csv", index=False)
    selected = {key: _json_safe(value) for key, value in summary.iloc[0].to_dict().items()}
    (output_dir / "selected_event_alpha_strategy.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    (output_dir / "event_alpha_config.json").write_text(
        json.dumps(
            {
                "analysis": "positivity event-driven long alpha grid",
                "start": str(start.date()),
                "end": str(pd.Timestamp(end_ts).date()),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "spec_count": len(specs),
                "specs": [asdict(spec) for spec in specs],
                "return_alignment": "signal date t uses close-to-close return from t to next trading day",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run positivity event-driven alpha grid.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default=str(RESULT_DIR))
    parser.add_argument("--data-root", default=None, help="Directory containing parquet input files.")
    args = parser.parse_args()

    data_root = None if args.data_root is None else Path(args.data_root)
    summary = run_event_alpha_grid(start=args.start, end=args.end, output_dir=Path(args.output), data_root=data_root)
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
