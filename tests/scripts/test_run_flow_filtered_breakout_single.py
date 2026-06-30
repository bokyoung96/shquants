from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_tech_gamma_long_only import TechGammaConfig


def test_batched_single_runner_uses_warmup_for_loading_but_not_performance(tmp_path, monkeypatch) -> None:
    from scripts import run_flow_filtered_breakout_single as runner

    calls: list[tuple[tuple[str, ...], str, str]] = []
    frame_starts: list[pd.Timestamp] = []

    monkeypatch.setattr(runner, "kospi200_tickers", lambda _root, _config: ("A001", "A002", "A003"))
    monkeypatch.setattr(runner, "load_research_feature_data", lambda _root, _tickers: object())
    monkeypatch.setattr(
        runner,
        "build_daily_research_features",
        lambda **_kwargs: pd.DataFrame(
            {
                "date": pd.to_datetime(["2018-12-31", "2019-01-02"] * 3),
                "ticker": ["A001", "A001", "A002", "A002", "A003", "A003"],
                "daily_positivity": [1.0] * 6,
                "positivity_benchmark": [0.0] * 6,
                "positivity_spread": [1.0] * 6,
                "positivity_filter_ok": [True] * 6,
                "factor_filter_ok": [True] * 6,
            }
        ),
    )

    def fake_read_tickers_bars(_dataset, tickers, *, start, end):
        calls.append((tickers, str(pd.Timestamp(start).date()), str(pd.Timestamp(end))))
        rows = []
        for ticker in tickers:
            rows.extend(
                [
                    {"ts": pd.Timestamp("2018-12-31 09:00"), "ticker": ticker},
                    {"ts": pd.Timestamp("2019-01-02 09:00"), "ticker": ticker},
                ]
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(runner, "read_tickers_bars", fake_read_tickers_bars)
    monkeypatch.setattr(runner, "build_features", lambda raw, _config: raw.assign(date=raw["ts"].dt.normalize()))
    monkeypatch.setattr(runner, "filter_kospi200_historical_members", lambda frame, _root: frame)

    def fake_simulate(frame, _config):
        frame_starts.append(frame["ts"].min())
        return pd.DataFrame(
            {
                "ticker": [frame["ticker"].iloc[0]],
                "signal_date": ["2019-01-02"],
                "entry_time": [pd.Timestamp("2019-01-02 09:05")],
                "exit_time": [pd.Timestamp("2019-01-02 15:30")],
                "entry_price": [100.0],
                "exit_price": [101.0],
                "signal_score": [1.0],
                "gross_return": [0.01],
                "net_return": [0.0076],
                "exit_reason": ["test"],
            }
        )

    monkeypatch.setattr(runner, "simulate_continuation_holding", fake_simulate)
    monkeypatch.setattr(runner, "write_performance_outputs", lambda _intraday, _overnight, output, _title: (output / "performance_subplots.png").write_bytes(b"png"))

    output = runner.run_batched_single_strategy(
        TechGammaConfig(start="2019-01-01", end="2019-12-31", high_lookback_days=420, holding_mode="continuation"),
        output_dir=tmp_path,
        batch_size=2,
        dataset=object(),
    )

    assert output == tmp_path
    assert [call[0] for call in calls] == [("A001", "A002"), ("A003",)]
    assert {call[1] for call in calls} == {"2017-11-07"}
    assert all(start >= pd.Timestamp("2019-01-01") for start in frame_starts)
    assert (tmp_path / "base" / "summary.csv").exists()
    assert (tmp_path / "base" / "intraday_trades.csv").exists()
    assert not (tmp_path / "signals.csv").exists()


def test_daily_prefilter_keeps_only_possible_breakout_dates() -> None:
    from scripts.run_flow_filtered_breakout_single import prefilter_breakout_candidates

    dates = pd.to_datetime(["2018-12-28", "2019-01-02", "2019-01-03"])
    close = pd.DataFrame({"A001": [100.0, 101.0, 99.0], "A002": [50.0, 49.0, 48.0]}, index=dates)
    high = pd.DataFrame({"A001": [100.0, 103.0, 99.5], "A002": [50.0, 49.5, 48.5]}, index=dates)
    low = pd.DataFrame({"A001": [99.0, 100.5, 98.0], "A002": [49.0, 48.0, 47.0]}, index=dates)
    filters = pd.DataFrame(
        {
            "date": [pd.Timestamp("2019-01-02"), pd.Timestamp("2019-01-03")],
            "ticker": ["A001", "A001"],
            "positivity_filter_ok": [True, True],
            "factor_filter_ok": [True, True],
        }
    )

    candidates = prefilter_breakout_candidates(
        close=close,
        high=high,
        low=low,
        daily_features=filters,
        config=TechGammaConfig(start="2019-01-01", range_buffer_bps=8.0),
    )

    assert candidates[["date", "ticker"]].to_dict("records") == [
        {"date": pd.Timestamp("2019-01-02"), "ticker": "A001"}
    ]
    assert candidates.iloc[0]["prior_52w_close_high"] == 100.0


def test_daily_prefilter_respects_disabled_research_filters() -> None:
    from scripts.run_flow_filtered_breakout_single import prefilter_breakout_candidates

    dates = pd.to_datetime(["2018-12-28", "2019-01-02"])
    close = pd.DataFrame({"A001": [100.0, 101.0]}, index=dates)
    high = pd.DataFrame({"A001": [100.0, 103.0]}, index=dates)
    low = pd.DataFrame({"A001": [99.0, 100.5]}, index=dates)
    filters = pd.DataFrame(
        {
            "date": [pd.Timestamp("2019-01-02")],
            "ticker": ["A001"],
            "positivity_filter_ok": [False],
            "factor_filter_ok": [False],
        }
    )

    candidates = prefilter_breakout_candidates(
        close=close,
        high=high,
        low=low,
        daily_features=filters,
        config=TechGammaConfig(start="2019-01-01", range_buffer_bps=8.0, use_positivity=False, factor_filter="none"),
    )

    assert candidates[["date", "ticker"]].to_dict("records") == [
        {"date": pd.Timestamp("2019-01-02"), "ticker": "A001"}
    ]


def test_entry_candidates_can_wait_for_next_close_confirmation() -> None:
    from scripts.run_flow_filtered_breakout_single import _entry_candidates

    frame = pd.DataFrame(
        {
            "ticker": ["A001", "A001", "A001", "A001"],
            "date": pd.to_datetime(["2019-01-02"] * 4),
            "ts": pd.to_datetime(["2019-01-02 09:25", "2019-01-02 09:30", "2019-01-02 09:35", "2019-01-02 09:40"]),
            "next_ts": pd.to_datetime(["2019-01-02 09:30", "2019-01-02 09:35", "2019-01-02 09:40", pd.NaT]),
            "open": [99.0, 101.0, 102.0, 103.0],
            "next_open": [101.0, 102.0, 103.0, float("nan")],
            "close": [99.0, 101.0, 102.0, 103.0],
            "previous_intraday_close": [float("nan"), 99.0, 101.0, 102.0],
            "prior_52w_close_high": [100.0] * 4,
            "atr": [1.0] * 4,
            "hhmm": ["0925", "0930", "0935", "0940"],
            "breakout_52w_bps": [-100.0, 100.0, 200.0, 300.0],
            "signal_score": [0.0, 1.0, 2.0, 3.0],
            "positivity_filter_ok": [True] * 4,
            "factor_filter_ok": [True] * 4,
        }
    )

    baseline = _entry_candidates(
        frame,
        TechGammaConfig(start="2019-01-01", range_end_hhmm="0920", exit_hhmm="1455", range_buffer_bps=8.0),
    )
    confirmed = _entry_candidates(
        frame,
        TechGammaConfig(
            start="2019-01-01",
            range_end_hhmm="0920",
            exit_hhmm="1455",
            range_buffer_bps=8.0,
            entry_confirmation="next_close_confirmed",
        ),
    )

    assert baseline.iloc[0]["ts"] == pd.Timestamp("2019-01-02 09:30")
    assert baseline.iloc[0]["next_ts"] == pd.Timestamp("2019-01-02 09:35")
    assert baseline.iloc[0]["next_open"] == 102.0
    assert confirmed.iloc[0]["ts"] == pd.Timestamp("2019-01-02 09:30")
    assert confirmed.iloc[0]["next_ts"] == pd.Timestamp("2019-01-02 09:40")
    assert confirmed.iloc[0]["next_open"] == 103.0


def test_entry_candidates_reject_failed_next_close_confirmation() -> None:
    from scripts.run_flow_filtered_breakout_single import _entry_candidates

    frame = pd.DataFrame(
        {
            "ticker": ["A001", "A001", "A001"],
            "date": pd.to_datetime(["2019-01-02"] * 3),
            "ts": pd.to_datetime(["2019-01-02 09:25", "2019-01-02 09:30", "2019-01-02 09:35"]),
            "next_ts": pd.to_datetime(["2019-01-02 09:30", "2019-01-02 09:35", pd.NaT]),
            "open": [99.0, 101.0, 99.5],
            "next_open": [101.0, 99.5, float("nan")],
            "close": [99.0, 101.0, 99.5],
            "previous_intraday_close": [float("nan"), 99.0, 101.0],
            "prior_52w_close_high": [100.0] * 3,
            "atr": [1.0] * 3,
            "hhmm": ["0925", "0930", "0935"],
            "breakout_52w_bps": [-100.0, 100.0, -50.0],
            "signal_score": [0.0, 1.0, 0.0],
            "positivity_filter_ok": [True] * 3,
            "factor_filter_ok": [True] * 3,
        }
    )

    confirmed = _entry_candidates(
        frame,
        TechGammaConfig(
            start="2019-01-01",
            range_end_hhmm="0920",
            exit_hhmm="1455",
            range_buffer_bps=8.0,
            entry_confirmation="next_close_confirmed",
        ),
    )

    assert confirmed.empty


def test_entry_candidates_reject_floating_point_touch_as_confirmation() -> None:
    from scripts.run_flow_filtered_breakout_single import _entry_candidates

    frame = pd.DataFrame(
        {
            "ticker": ["A001", "A001", "A001"],
            "date": pd.to_datetime(["2019-01-02"] * 3),
            "ts": pd.to_datetime(["2019-01-02 09:25", "2019-01-02 09:30", "2019-01-02 09:35"]),
            "next_ts": pd.to_datetime(["2019-01-02 09:30", "2019-01-02 09:35", "2019-01-02 09:40"]),
            "open": [99.0, 101.0, 100.0],
            "next_open": [101.0, 100.0, 100.2],
            "close": [99.0, 101.0, 100.0],
            "previous_intraday_close": [float("nan"), 99.0, 101.0],
            "prior_52w_close_high": [99.99999999999999] * 3,
            "atr": [1.0] * 3,
            "hhmm": ["0925", "0930", "0935"],
            "breakout_52w_bps": [-100.0, 100.0, 0.0],
            "signal_score": [0.0, 1.0, 0.0],
            "positivity_filter_ok": [True] * 3,
            "factor_filter_ok": [True] * 3,
        }
    )

    confirmed = _entry_candidates(
        frame,
        TechGammaConfig(
            start="2019-01-01",
            range_end_hhmm="0920",
            exit_hhmm="1455",
            range_buffer_bps=8.0,
            entry_confirmation="next_close_confirmed",
        ),
    )

    assert confirmed.empty


def test_remove_overlapping_trades_preserves_cross_month_position_state() -> None:
    from scripts.run_flow_filtered_breakout_single import remove_overlapping_trades

    trades = pd.DataFrame(
        {
            "ticker": ["A001", "A001", "A001", "A002"],
            "signal_time": pd.to_datetime(["2019-01-31 10:00", "2019-02-01 10:00", "2019-02-04 10:00", "2019-02-01 10:00"]),
            "entry_time": pd.to_datetime(["2019-01-31 10:05", "2019-02-01 10:05", "2019-02-04 10:05", "2019-02-01 10:05"]),
            "exit_time": pd.to_datetime(["2019-02-01 15:30", "2019-02-04 15:30", "2019-02-05 15:30", "2019-02-01 15:30"]),
            "net_return": [0.01, 0.02, 0.03, 0.04],
        }
    )

    filtered = remove_overlapping_trades(trades)

    assert filtered.sort_values(["ticker", "signal_time"])[["ticker", "signal_time"]].to_dict("records") == [
        {"ticker": "A001", "signal_time": pd.Timestamp("2019-01-31 10:00")},
        {"ticker": "A001", "signal_time": pd.Timestamp("2019-02-04 10:00")},
        {"ticker": "A002", "signal_time": pd.Timestamp("2019-02-01 10:00")},
    ]


def test_compress_breakout_episodes_requires_daily_close_reset() -> None:
    from scripts.run_flow_filtered_breakout_single import compress_breakout_episodes

    trades = pd.DataFrame(
        {
            "ticker": ["A001", "A001", "A001", "A002"],
            "signal_time": pd.to_datetime(["2019-01-02 10:00", "2019-01-04 10:00", "2019-01-08 10:00", "2019-01-04 10:00"]),
            "entry_time": pd.to_datetime(["2019-01-02 10:10", "2019-01-04 10:10", "2019-01-08 10:10", "2019-01-04 10:10"]),
            "exit_time": pd.to_datetime(["2019-01-03 15:30", "2019-01-04 15:30", "2019-01-09 15:30", "2019-01-04 15:30"]),
            "net_return": [-0.01, 0.02, 0.03, 0.04],
        }
    )
    daily = pd.DataFrame(
        {
            "ticker": ["A001", "A001", "A001", "A001", "A002"],
            "date": pd.to_datetime(["2019-01-02", "2019-01-03", "2019-01-05", "2019-01-08", "2019-01-04"]),
            "close": [101.0, 102.0, 99.0, 103.0, 55.0],
            "daily_low": [100.0, 101.0, 98.0, 102.0, 54.0],
            "prior_52w_close_high": [100.0, 100.0, 100.0, 100.0, 50.0],
        }
    )

    compressed = compress_breakout_episodes(trades, daily)

    assert compressed.sort_values(["ticker", "signal_time"])[["ticker", "signal_time"]].to_dict("records") == [
        {"ticker": "A001", "signal_time": pd.Timestamp("2019-01-02 10:00")},
        {"ticker": "A001", "signal_time": pd.Timestamp("2019-01-08 10:00")},
        {"ticker": "A002", "signal_time": pd.Timestamp("2019-01-04 10:00")},
    ]


def test_config_from_json_ignores_removed_experimental_keys(tmp_path: Path) -> None:
    from scripts.run_flow_filtered_breakout_single import config_from_json

    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "scheme": "52w_high_breakout",
          "entry_confirmation": "next_close_confirmed",
          "episode_compression": true,
          "episode_compression_mode": "prior_high_ratchet",
          "market_regime_filter": "iks200_above_ma",
          "sector_regime_filter": "same_sector_above_ma_breadth"
        }
        """,
        encoding="utf-8",
    )

    config = config_from_json(config_path, start="2019-01-01")

    assert config.entry_confirmation == "next_close_confirmed"
    assert config.episode_compression is True
    assert not hasattr(config, "episode_compression_mode")
