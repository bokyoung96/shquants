from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_pos_research import (
    MomentumComparisonResult,
    PosResearchResult,
    _latest_weights,
    summarize_quintile_returns,
    _weighted_next_day_returns,
    write_momentum_comparison_outputs,
    write_outputs,
    _ordered_plot_columns,
)


def test_summarize_quintile_returns_includes_q5_minus_q1_spread() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    returns = pd.DataFrame(
        {
            "q1": [0.0, 0.01, -0.01],
            "q5": [0.02, 0.01, 0.03],
        },
        index=idx,
    )

    summary = summarize_quintile_returns(returns)

    assert summary["portfolio"].tolist() == ["q1", "q5", "q5_minus_q1"]
    assert summary.loc[summary["portfolio"].eq("q5_minus_q1"), "total_return"].iloc[0] > 0.0


def test_weighted_next_day_returns_keeps_inactive_days_as_cash_returns() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    weights = pd.DataFrame({"A": [0.0, 1.0, 0.0]}, index=idx)
    stock_returns = pd.DataFrame({"A": [0.00, 0.05, -0.10]}, index=idx)

    returns = _weighted_next_day_returns(weights=weights, stock_returns=stock_returns)

    assert returns.index.tolist() == idx[:-1].tolist()
    assert returns.tolist() == pytest.approx([0.0, -0.10])


def test_write_outputs_creates_pos_research_artifacts(tmp_path: Path) -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    returns = pd.DataFrame({"q1": [0.01, -0.01], "q5": [0.02, 0.03]}, index=idx)
    equity = (1.0 + returns).cumprod()
    weights = {
        "q1": pd.DataFrame({"A": [1.0, 1.0], "B": [0.0, 0.0]}, index=idx),
        "q5": pd.DataFrame({"A": [0.0, 0.0], "B": [1.0, 1.0]}, index=idx),
    }
    summary = summarize_quintile_returns(returns)
    sponsorship_returns = pd.DataFrame(
        {
            "foreign_persistent": [0.01, 0.02],
            "dual_sponsorship": [0.02, 0.01],
            "reacceleration": [0.03, -0.01],
        },
        index=idx,
    )
    sponsorship_summary = summarize_quintile_returns(sponsorship_returns)
    reacceleration_weights = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=idx)
    band_holding_returns = pd.DataFrame({"band_holding": [0.01, 0.02]}, index=idx)
    band_holding_weights = pd.DataFrame({"A": [1.0, 0.5], "B": [0.0, 0.5]}, index=idx)
    signal_band_returns = pd.DataFrame({"signal_band_v1": [0.03, -0.01]}, index=idx)
    signal_band_weights = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=idx)
    signal_band_trades = pd.DataFrame(
        [{"symbol": "A", "entry_date": idx[0], "exit_date": idx[1], "return": 0.02, "exit_reason": "stop"}]
    )
    pure_tilt_returns = pd.DataFrame({"pure_tilt_v1": [0.02, 0.01]}, index=idx)
    pure_tilt_weights = pd.DataFrame({"A": [0.5, 1.0], "B": [0.5, 0.0]}, index=idx)
    pure_tilt_trades = pd.DataFrame(
        [{"symbol": "B", "entry_date": idx[0], "exit_date": idx[1], "return": 0.01, "exit_reason": "band_exit"}]
    )
    sector_breakout_returns = pd.DataFrame({"sector_breakout_v1": [0.01, 0.02]}, index=idx)
    sector_breakout_weights = pd.DataFrame({"A": [1.0, 0.5], "B": [0.0, 0.5]}, index=idx)
    sector_breakout_trades = pd.DataFrame(
        [{"symbol": "A", "entry_date": idx[0], "exit_date": idx[1], "return": 0.03, "exit_reason": "stop"}]
    )
    sector_state = pd.DataFrame(
        {"sector_weighted_pos": [0.5], "sector_equal_pos": [0.4]},
        index=pd.MultiIndex.from_tuples([(idx[0], "Tech")], names=["date", "sector"]),
    )
    market_state = pd.DataFrame({"market_weighted_pos": [0.5], "market_equal_pos": [0.4]}, index=[idx[0]])
    entry_candidates = pd.DataFrame([{"date": idx[0], "symbol": "A", "mode": "sector_expansion"}])
    sector_event_returns = pd.DataFrame({"sector_event_core_v2": [0.02, 0.03]}, index=idx)
    sector_event_weights = pd.DataFrame({"A": [1.0, 1.0], "B": [0.0, 0.0]}, index=idx)
    sector_event_trades = pd.DataFrame(
        [{"symbol": "A", "entry_date": idx[0], "exit_date": idx[1], "return": 0.04, "exit_reason": "stop"}]
    )

    written = write_outputs(
        tmp_path,
        PosResearchResult(
            returns=returns,
            equity=equity,
            weights=weights,
            summary=summary,
            sponsorship_returns=sponsorship_returns,
            sponsorship_summary=sponsorship_summary,
            reacceleration_weights=reacceleration_weights,
            band_holding_returns=band_holding_returns,
            band_holding_summary=summarize_quintile_returns(band_holding_returns),
            band_holding_weights=band_holding_weights,
            signal_band_returns=signal_band_returns,
            signal_band_summary=summarize_quintile_returns(signal_band_returns),
            signal_band_weights=signal_band_weights,
            signal_band_trades=signal_band_trades,
            pure_tilt_returns=pure_tilt_returns,
            pure_tilt_summary=summarize_quintile_returns(pure_tilt_returns),
            pure_tilt_weights=pure_tilt_weights,
            pure_tilt_trades=pure_tilt_trades,
            sector_breakout_returns=sector_breakout_returns,
            sector_breakout_summary=summarize_quintile_returns(sector_breakout_returns),
            sector_breakout_weights=sector_breakout_weights,
            sector_breakout_trades=sector_breakout_trades,
            sector_breakout_sector_state=sector_state,
            sector_breakout_market_state=market_state,
            sector_breakout_entry_candidates=entry_candidates,
            sector_event_returns=sector_event_returns,
            sector_event_summary=summarize_quintile_returns(sector_event_returns),
            sector_event_weights=sector_event_weights,
            sector_event_trades=sector_event_trades,
            sector_event_sector_state=sector_state,
            sector_event_market_state=market_state,
            sector_event_entry_candidates=entry_candidates,
            metadata={"lookback": 252, "universe": "KOSPI200"},
        ),
    )

    assert (tmp_path / "daily_returns.csv").exists()
    assert (tmp_path / "equity.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "plots" / "cumulative_performance.png").stat().st_size > 0
    assert (tmp_path / "plots" / "summary_metrics.png").stat().st_size > 0
    assert (tmp_path / "positions" / "weights_q1.parquet").exists()
    assert (tmp_path / "positions" / "latest_q5.csv").exists()
    assert (tmp_path / "sponsorship" / "daily_returns.csv").exists()
    assert (tmp_path / "sponsorship" / "summary.csv").exists()
    assert (tmp_path / "sponsorship" / "summary.json").exists()
    assert (tmp_path / "positions" / "weights_reacceleration.parquet").exists()
    assert (tmp_path / "positions" / "latest_reacceleration.csv").exists()
    assert (tmp_path / "band_holding" / "daily_returns.csv").exists()
    assert (tmp_path / "band_holding" / "summary.csv").exists()
    assert (tmp_path / "positions" / "weights_band_holding.parquet").exists()
    assert (tmp_path / "positions" / "latest_band_holding.csv").exists()
    assert (tmp_path / "plots" / "band_holding.png").stat().st_size > 0
    assert (tmp_path / "signal_band_strategy" / "daily_returns.csv").exists()
    assert (tmp_path / "signal_band_strategy" / "summary.csv").exists()
    assert (tmp_path / "signal_band_strategy" / "trades.csv").exists()
    assert (tmp_path / "positions" / "weights_signal_band_v1.parquet").exists()
    assert (tmp_path / "positions" / "latest_signal_band_v1.csv").exists()
    assert (tmp_path / "plots" / "signal_band_v1.png").stat().st_size > 0
    assert (tmp_path / "pure_tilt_strategy" / "daily_returns.csv").exists()
    assert (tmp_path / "pure_tilt_strategy" / "summary.csv").exists()
    assert (tmp_path / "pure_tilt_strategy" / "trades.csv").exists()
    assert (tmp_path / "positions" / "weights_pure_tilt_v1.parquet").exists()
    assert (tmp_path / "positions" / "latest_pure_tilt_v1.csv").exists()
    assert (tmp_path / "plots" / "pure_tilt_v1.png").stat().st_size > 0
    assert (tmp_path / "sector_positivity_breakout" / "daily_returns.csv").exists()
    assert (tmp_path / "sector_positivity_breakout" / "summary.csv").exists()
    assert (tmp_path / "sector_positivity_breakout" / "trades.csv").exists()
    assert (tmp_path / "sector_positivity_breakout" / "sector_state.csv").exists()
    assert (tmp_path / "sector_positivity_breakout" / "market_state.csv").exists()
    assert (tmp_path / "sector_positivity_breakout" / "entry_candidates.csv").exists()
    assert (tmp_path / "positions" / "weights_sector_breakout_v1.parquet").exists()
    assert (tmp_path / "positions" / "latest_sector_breakout_v1.csv").exists()
    assert (tmp_path / "plots" / "sector_breakout_v1.png").stat().st_size > 0
    assert (tmp_path / "sector_event_core" / "daily_returns.csv").exists()
    assert (tmp_path / "sector_event_core" / "summary.csv").exists()
    assert (tmp_path / "sector_event_core" / "trades.csv").exists()
    assert (tmp_path / "sector_event_core" / "sector_state.csv").exists()
    assert (tmp_path / "sector_event_core" / "market_state.csv").exists()
    assert (tmp_path / "sector_event_core" / "entry_candidates.csv").exists()
    assert (tmp_path / "positions" / "weights_sector_event_core_v2.parquet").exists()
    assert (tmp_path / "positions" / "latest_sector_event_core_v2.csv").exists()
    assert (tmp_path / "plots" / "sector_event_core_v2.png").stat().st_size > 0
    assert (tmp_path / "plots" / "sponsorship_groups.png").stat().st_size > 0
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["lookback"] == 252
    assert written["output_dir"] == str(tmp_path)
    assert written["cumulative_plot"] == str(tmp_path / "plots" / "cumulative_performance.png")
    assert written["summary_plot"] == str(tmp_path / "plots" / "summary_metrics.png")
    assert written["sponsorship_returns"] == str(tmp_path / "sponsorship" / "daily_returns.csv")
    assert written["band_holding_returns"] == str(tmp_path / "band_holding" / "daily_returns.csv")
    assert written["signal_band_returns"] == str(tmp_path / "signal_band_strategy" / "daily_returns.csv")
    assert written["pure_tilt_returns"] == str(tmp_path / "pure_tilt_strategy" / "daily_returns.csv")
    assert written["sector_breakout_returns"] == str(tmp_path / "sector_positivity_breakout" / "daily_returns.csv")
    assert written["sector_event_returns"] == str(tmp_path / "sector_event_core" / "daily_returns.csv")


def test_latest_weights_uses_last_active_row_for_sparse_entry_signals() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    weights = pd.DataFrame({"A": [0.0, 1.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=idx)

    latest = _latest_weights(weights)

    assert latest.to_dict(orient="records") == [{"symbol": "A", "weight": 1.0}]


def test_write_momentum_comparison_outputs_creates_artifacts(tmp_path: Path) -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    returns = pd.DataFrame(
        {
            "positivity_q5": [0.01, 0.02],
            "return_momentum_q5": [0.02, -0.01],
        },
        index=idx,
    )
    summary = summarize_quintile_returns(returns)

    written = write_momentum_comparison_outputs(
        tmp_path,
        MomentumComparisonResult(
            returns=returns,
            equity=(1.0 + returns).cumprod(),
            summary=summary,
            metadata={"lookback": 252},
        ),
    )

    assert (tmp_path / "daily_returns.csv").exists()
    assert (tmp_path / "equity.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "comparison.png").stat().st_size > 0
    assert written["comparison_plot"] == str(tmp_path / "comparison.png")


def test_ordered_plot_columns_keeps_custom_comparison_columns_before_benchmark() -> None:
    frame = pd.DataFrame(
        columns=[
            "positivity_q5",
            "return_momentum_q5",
            "positivity_q5_minus_q1",
            "return_momentum_q5_minus_q1",
            "KOSPI200",
        ]
    )

    assert _ordered_plot_columns(frame) == [
        "positivity_q5",
        "return_momentum_q5",
        "positivity_q5_minus_q1",
        "return_momentum_q5_minus_q1",
        "KOSPI200",
    ]
