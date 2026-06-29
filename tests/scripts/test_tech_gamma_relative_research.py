from __future__ import annotations

import pandas as pd

from scripts.run_tech_gamma_breakout_grid import (
    _base_entry_candidates,
    _daily_frame,
    _simulate_continuation,
    build_strategy_specs,
    rank_grid_summary,
)
from scripts.run_tech_gamma_long_only import TechGammaConfig, build_features, simulate_intraday
from scripts.tech_gamma_holding import simulate_continuation_holding
from scripts.tech_gamma_research_filters import ResearchFeatureData, apply_research_features


def _ticker_frame(ticker: str, closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-02 09:05", periods=len(closes), freq="5min")
    return pd.DataFrame(
        {
            "ts": index,
            "ticker": ticker,
            "open": [closes[0], *closes[:-1]],
            "high": [value + 1.0 for value in closes],
            "low": [value - 1.0 for value in closes],
            "close": closes,
            "volume": [100.0] * len(closes),
        }
    )


def _dated_day(ticker: str, close: float, offset_days: int) -> pd.DataFrame:
    day = _ticker_frame(ticker, [close] * 8)
    day["ts"] = day["ts"] - pd.Timedelta(days=offset_days)
    return day


def test_sector_cap_weighted_positivity_selects_above_sector_average() -> None:
    config = TechGammaConfig(
        scheme="52w_high_breakout",
        range_end_hhmm="0920",
        use_positivity=True,
        positivity_lookback_days=3,
        positivity_benchmark="sector_cap_weighted",
    )
    good_warmup = [_dated_day("A005930", close, offset) for offset, close in zip([7, 6, 5, 4], [100.0, 101.0, 102.0, 103.0], strict=True)]
    bad_warmup = [_dated_day("A000660", close, offset) for offset, close in zip([7, 6, 5, 4], [100.0, 99.0, 100.0, 99.0], strict=True)]
    breakout = [103, 103, 103, 103, 104, 104.5, 105, 105]
    raw = pd.concat(
        [*good_warmup, *bad_warmup, _ticker_frame("A005930", breakout), _ticker_frame("A000660", breakout)],
        ignore_index=True,
    )
    frame = build_features(raw, config)
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    data = ResearchFeatureData(
        sector=pd.DataFrame("IT", index=dates, columns=["A005930", "A000660"]),
        market_cap=pd.DataFrame({"A005930": [3.0] * len(dates), "A000660": [1.0] * len(dates)}, index=dates),
    )

    enriched = apply_research_features(frame, config, data)
    trades = simulate_intraday(enriched, config)

    assert "positivity_benchmark" in enriched.columns
    assert "positivity_spread" in enriched.columns
    assert trades["ticker"].tolist() == ["A005930"]


def test_op_revision_filter_uses_prior_day_revision() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["A005930", "A005930", "A005930"],
            "close": [100.0, 101.0, 102.0],
        }
    )
    data = ResearchFeatureData(
        op_fwd_12m=pd.DataFrame(
            {"A005930": [100.0, 150.0, 150.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )
    )
    config = TechGammaConfig(
        factor_filter="op_revision_positive",
        factor_lookback_days=1,
    )

    enriched = apply_research_features(frame, config, data)

    by_date = enriched.set_index("date")
    assert pd.isna(by_date.loc[pd.Timestamp("2024-01-03"), "op_revision"])
    assert not bool(by_date.loc[pd.Timestamp("2024-01-03"), "factor_filter_ok"])
    assert by_date.loc[pd.Timestamp("2024-01-04"), "op_revision"] == 0.5
    assert bool(by_date.loc[pd.Timestamp("2024-01-04"), "factor_filter_ok"])


def test_flow_filters_use_prior_trailing_flow_to_cap() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["A005930", "A005930", "A005930"],
            "close": [100.0, 101.0, 102.0],
        }
    )
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    data = ResearchFeatureData(
        market_cap=pd.DataFrame({"A005930": [100.0, 100.0, 100.0]}, index=dates),
        foreign_flow=pd.DataFrame({"A005930": [-2.0, 5.0, 7.0]}, index=dates),
        institution_flow=pd.DataFrame({"A005930": [-1.0, -1.0, 3.0]}, index=dates),
    )

    foreign = apply_research_features(
        frame,
        TechGammaConfig(factor_filter="foreign_positive", factor_lookback_days=1),
        data,
    ).set_index("date")
    either = apply_research_features(
        frame,
        TechGammaConfig(factor_filter="foreign_or_institution_positive", factor_lookback_days=1),
        data,
    ).set_index("date")
    both = apply_research_features(
        frame,
        TechGammaConfig(factor_filter="foreign_and_institution_positive", factor_lookback_days=1),
        data,
    ).set_index("date")

    assert not bool(foreign.loc[pd.Timestamp("2024-01-03"), "factor_filter_ok"])
    assert bool(foreign.loc[pd.Timestamp("2024-01-04"), "factor_filter_ok"])
    assert bool(either.loc[pd.Timestamp("2024-01-04"), "factor_filter_ok"])
    assert not bool(both.loc[pd.Timestamp("2024-01-04"), "factor_filter_ok"])


def test_fast_grid_continuation_matches_reference_simulator() -> None:
    day1 = _ticker_frame("A005930", [100, 101, 102, 103, 104, 105, 106, 106])
    day2 = _ticker_frame("A005930", [106, 106, 106, 106, 107, 107.5, 108, 108])
    day3 = _ticker_frame("A005930", [108, 108.5, 109, 109.5, 110, 110.5, 111, 111])
    day4 = _ticker_frame("A005930", [111, 110.5, 110, 109.5, 109, 108.5, 108, 108])
    day2["ts"] = day2["ts"] + pd.Timedelta(days=1)
    day3["ts"] = day3["ts"] + pd.Timedelta(days=2)
    day4["ts"] = day4["ts"] + pd.Timedelta(days=3)
    config = TechGammaConfig(
        scheme="52w_high_breakout",
        range_end_hhmm="0920",
        holding_mode="continuation",
        min_holding_days=1,
        atr_stop_multiplier=10.0,
    )
    frame = build_features(pd.concat([day1, day2, day3, day4], ignore_index=True), config)

    fast = _simulate_continuation(_base_entry_candidates(frame), _daily_frame(frame), config)
    reference = simulate_continuation_holding(frame, config)

    assert fast[["ticker", "signal_time", "entry_time", "exit_time", "exit_reason"]].to_dict("records") == reference[
        ["ticker", "signal_time", "entry_time", "exit_time", "exit_reason"]
    ].to_dict("records")


def test_fast_grid_continuation_drops_open_position_before_min_holding_at_data_end() -> None:
    day1 = _ticker_frame("A005930", [100, 101, 102, 103, 104, 105, 106, 106])
    day2 = _ticker_frame("A005930", [106, 106, 106, 106, 107, 107.5, 108, 108])
    day2["ts"] = day2["ts"] + pd.Timedelta(days=1)
    config = TechGammaConfig(
        scheme="52w_high_breakout",
        range_end_hhmm="0920",
        holding_mode="continuation",
        min_holding_days=2,
        atr_stop_multiplier=10.0,
    )
    frame = build_features(pd.concat([day1, day2], ignore_index=True), config)

    fast = _simulate_continuation(_base_entry_candidates(frame), _daily_frame(frame), config)

    assert fast.empty


def test_breakout_grid_is_flow_filtered_without_op_filters() -> None:
    specs = build_strategy_specs(max_strategies=5000)
    filters = {spec.config.factor_filter for spec in specs}
    lookbacks = {spec.config.factor_lookback_days for spec in specs}
    positivity_lookbacks = {spec.config.positivity_lookback_days for spec in specs}
    benchmarks = {spec.config.positivity_benchmark for spec in specs}

    assert len(specs) == 5000
    assert len({spec.name for spec in specs}) == 5000
    assert filters == {
        "none",
        "foreign_positive",
        "institution_positive",
        "foreign_or_institution_positive",
        "foreign_and_institution_positive",
    }
    assert not any("op_" in factor_filter for factor_filter in filters)
    assert lookbacks == {20, 40, 60}
    assert positivity_lookbacks == {60, 90, 126}
    assert benchmarks == {"sector_cap_weighted", "index_cap_weighted"}
    assert all(spec.config.min_daily_positivity == 0.0 for spec in specs)


def test_rank_grid_summary_prefers_trade_viable_late_robust_strategy() -> None:
    summary = pd.DataFrame(
        {
            "strategy": ["high_avg_overfit", "lower_avg_robust"],
            "trades": [10, 20],
            "avg_net_bps": [300.0, 120.0],
            "hit_rate": [0.9, 0.55],
            "early_avg_net_bps": [500.0, 100.0],
            "late_avg_net_bps": [-50.0, 110.0],
            "max_drawdown": [-0.4, -0.08],
        }
    )

    ranked = rank_grid_summary(summary)

    assert ranked.iloc[0]["strategy"] == "lower_avg_robust"
    assert ranked.iloc[0]["selection_rank"] == 1
