from __future__ import annotations

import pandas as pd

from backtesting.data.kr_stock_5m import KrStock5mDataset
from scripts.run_tech_gamma_long_only import (
    TechGammaConfig,
    build_features,
    run,
    simulate_continuation_holding,
    simulate_intraday,
    simulate_overnight,
)
from scripts.tech_gamma_intraday import TradeSide
from scripts.tech_gamma_schemes import get_scheme, scheme_names
from scripts.tech_gamma_plots import write_performance_outputs


def _ticker_frame(ticker: str, closes: list[float], volumes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-02 09:05", periods=len(closes), freq="5min")
    return pd.DataFrame(
        {
            "ts": index,
            "ticker": ticker,
            "open": [closes[0], *closes[:-1]],
            "high": [value + 1.0 for value in closes],
            "low": [value - 1.0 for value in closes],
            "close": closes,
            "volume": volumes,
        }
    )


def test_simulate_intraday_buys_stronger_opening_range_breakout() -> None:
    weak = _ticker_frame("A005930", [100, 100, 101, 101, 102, 103, 104, 103], [100, 100, 100, 100, 130, 130, 130, 130])
    strong = _ticker_frame("A000660", [100, 100, 101, 102, 106, 109, 112, 111], [100, 100, 100, 100, 500, 520, 540, 520])
    frame = build_features(pd.concat([weak, strong], ignore_index=True), TechGammaConfig(range_end_hhmm="0920", exit_hhmm="0940"))

    trades = simulate_intraday(frame, TechGammaConfig(range_end_hhmm="0920", exit_hhmm="0940"))

    assert not trades.empty
    first = trades.iloc[0]
    assert first["ticker"] == "A000660"
    assert first["entry_time"] > first["signal_time"]
    assert first["exit_reason"] in {"time_exit", "trailing_stop", "vwap_failure"}


def test_build_features_uses_selected_scheme_score() -> None:
    raw = _ticker_frame("A005930", [100, 100, 101, 102, 106, 109, 112, 111], [100, 100, 100, 100, 500, 520, 540, 520])

    frame = build_features(raw, TechGammaConfig(scheme="opening_range_vwap", range_end_hhmm="0920"))

    assert "signal_score" in frame.columns
    assert "opening_range_vwap_score" in frame.columns
    assert frame["signal_score"].max() == frame["opening_range_vwap_score"].max()


def test_get_scheme_rejects_unknown_scheme() -> None:
    try:
        get_scheme("does_not_exist")
    except KeyError as exc:
        assert "does_not_exist" in str(exc)
        assert "tech_gamma" in str(exc)
    else:
        raise AssertionError("unknown scheme should fail")


def test_scheme_names_exposes_builtin_schemes() -> None:
    assert scheme_names() == ("52w_high_breakout", "opening_range_vwap", "tech_gamma")


def test_52w_high_breakout_uses_intraday_new_high_without_positivity() -> None:
    day1 = _ticker_frame("A005930", [100, 101, 102, 103, 104, 105, 106, 106], [100, 100, 100, 100, 100, 100, 100, 100])
    day2 = _ticker_frame("A005930", [106, 106, 106, 106, 107, 106.5, 106.25, 106.1], [100, 100, 100, 100, 350, 360, 370, 380])
    day2["ts"] = day2["ts"] + pd.Timedelta(days=1)
    warmup = []
    for index, close in enumerate([100.0, 101.0, 100.0], start=1):
        day = _ticker_frame("A005930", [close] * 8, [100] * 8)
        day["ts"] = day["ts"] - pd.Timedelta(days=4 - index)
        warmup.append(day)
    raw = pd.concat([*warmup, day1, day2], ignore_index=True)
    config = TechGammaConfig(
        scheme="52w_high_breakout",
        range_end_hhmm="0920",
        min_score=999.0,
    )

    frame = build_features(
        raw,
        config,
    )
    trades = simulate_intraday(
        frame,
        TechGammaConfig(
            scheme="52w_high_breakout",
            range_end_hhmm="0920",
            exit_hhmm="0940",
            min_score=999.0,
        ),
    )

    assert "prior_52w_close_high" in frame.columns
    assert "daily_positivity" not in frame.columns
    assert "intraday_positivity" not in frame.columns
    assert frame.loc[frame["date"].eq(pd.Timestamp("2024-01-03")), "prior_52w_close_high"].dropna().iloc[0] == 106
    assert not trades.empty
    assert trades.iloc[0]["ticker"] == "A005930"
    assert trades.iloc[0]["signal_time"] == pd.Timestamp("2024-01-03 09:25")
    assert trades.iloc[0]["entry_time"] == pd.Timestamp("2024-01-03 09:30")


def test_52w_high_breakout_uses_entry_atr_stop_for_long() -> None:
    day1 = _ticker_frame("A005930", [100, 101, 102, 103, 104, 105, 106, 106], [100, 100, 100, 100, 100, 100, 100, 100])
    day2 = _ticker_frame("A005930", [106, 106, 106, 106, 107, 106.8, 105.2, 105.1], [100, 100, 100, 100, 350, 360, 370, 380])
    day2["ts"] = day2["ts"] + pd.Timedelta(days=1)
    raw = pd.concat([day1, day2], ignore_index=True)
    config = TechGammaConfig(
        scheme="52w_high_breakout",
        range_end_hhmm="0920",
        exit_hhmm="0940",
        atr_lookback_bars=2,
        atr_stop_multiplier=1.0,
    )

    trades = simulate_intraday(build_features(raw, config), config)

    assert not trades.empty
    first = trades.iloc[0]
    assert first["exit_reason"] == "atr_stop"
    assert first["entry_time"] == pd.Timestamp("2024-01-03 09:30")
    assert first["exit_time"] == pd.Timestamp("2024-01-03 09:35")
    assert first["exit_price"] == 105.0


def test_52w_high_breakout_continuation_holds_while_daily_new_high_persists() -> None:
    day1 = _ticker_frame("A005930", [100, 101, 102, 103, 104, 105, 106, 106], [100] * 8)
    day2 = _ticker_frame("A005930", [106, 106, 106, 106, 107, 107.5, 108, 108], [100, 100, 100, 100, 350, 360, 370, 380])
    day3 = _ticker_frame("A005930", [108, 108.5, 109, 109.5, 110, 110.5, 111, 111], [200] * 8)
    day4 = _ticker_frame("A005930", [111, 110.5, 110, 109.5, 109, 108.5, 108, 108], [200] * 8)
    day2["ts"] = day2["ts"] + pd.Timedelta(days=1)
    day3["ts"] = day3["ts"] + pd.Timedelta(days=2)
    day4["ts"] = day4["ts"] + pd.Timedelta(days=3)
    config = TechGammaConfig(
        scheme="52w_high_breakout",
        range_end_hhmm="0920",
        min_holding_days=1,
        atr_stop_multiplier=10.0,
    )
    frame = build_features(pd.concat([day1, day2, day3, day4], ignore_index=True), config)

    trades = simulate_continuation_holding(frame, config)

    assert len(trades) == 1
    first = trades.iloc[0]
    assert first["signal_time"] == pd.Timestamp("2024-01-03 09:25")
    assert first["entry_time"] == pd.Timestamp("2024-01-03 09:30")
    assert first["exit_time"] == pd.Timestamp("2024-01-05 09:40")
    assert first["exit_reason"] == "new_high_lost"
    assert first["gross_return"] > 0.0


def test_simulate_intraday_can_short_breakout_failure() -> None:
    day1 = _ticker_frame("A005930", [100, 101, 102, 103, 104, 105, 106, 106], [100, 100, 100, 100, 100, 100, 100, 100])
    day2 = _ticker_frame("A005930", [106, 106, 106, 106, 107, 106, 105, 104], [100, 100, 100, 100, 350, 360, 370, 380])
    day2["ts"] = day2["ts"] + pd.Timedelta(days=1)
    warmup = []
    for index, close in enumerate([100.0, 101.0, 100.0], start=1):
        day = _ticker_frame("A005930", [close] * 8, [100] * 8)
        day["ts"] = day["ts"] - pd.Timedelta(days=4 - index)
        warmup.append(day)
    raw = pd.concat([*warmup, day1, day2], ignore_index=True)
    config = TechGammaConfig(
        scheme="52w_high_breakout",
        side=TradeSide.SHORT,
        range_end_hhmm="0920",
        exit_hhmm="0940",
    )

    trades = simulate_intraday(build_features(raw, config), config)

    assert not trades.empty
    first = trades.iloc[0]
    assert first["side"] == "short"
    assert first["gross_return"] > 0.0
    assert first["exit_reason"] in {"time_exit", "trailing_stop", "vwap_reclaim", "stop_loss"}


def test_simulate_overnight_uses_close_pressure_and_next_open() -> None:
    day1 = _ticker_frame(
        "A005930",
        [100, 101, 102, 104, 107, 110, 114, 118, 121, 124],
        [100, 110, 120, 140, 300, 350, 400, 450, 500, 550],
    )
    day2 = _ticker_frame("A005930", [126, 127, 128], [200, 200, 200])
    day2["ts"] = day2["ts"] + pd.Timedelta(days=1)
    raw = pd.concat([day1, day2], ignore_index=True)
    frame = build_features(raw, TechGammaConfig(range_end_hhmm="0920", overnight_entry_hhmm="0945"))

    trades = simulate_overnight(frame, TechGammaConfig(range_end_hhmm="0920", overnight_entry_hhmm="0945"))

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["ticker"] == "A005930"
    assert trade["entry_time"].strftime("%Y-%m-%d") == "2024-01-02"
    assert trade["exit_time"].strftime("%Y-%m-%d") == "2024-01-03"
    assert trade["net_return"] > 0.0


def test_simulate_overnight_rejects_missing_opening_range_score() -> None:
    day1 = _ticker_frame("A005930", [100, 102, 104], [300, 350, 400])
    day1["ts"] = day1["ts"] + pd.Timedelta(hours=1)
    day2 = _ticker_frame("A005930", [106, 107], [200, 200])
    day2["ts"] = day2["ts"] + pd.Timedelta(days=1)
    raw = pd.concat([day1, day2], ignore_index=True)
    frame = build_features(raw, TechGammaConfig(range_end_hhmm="0920", overnight_entry_hhmm="1015"))

    trades = simulate_overnight(frame, TechGammaConfig(range_end_hhmm="0920", overnight_entry_hhmm="1015"))

    assert trades.empty


def test_write_performance_outputs_creates_subplot_png_and_equity_csv(tmp_path) -> None:
    intraday = pd.DataFrame(
        {
            "ticker": ["A005930", "A000660"],
            "exit_time": pd.to_datetime(["2024-01-02 14:55", "2024-01-03 14:55"]),
            "net_return": [0.01, -0.005],
        }
    )
    overnight = pd.DataFrame(
        {
            "ticker": ["A005930"],
            "exit_time": pd.to_datetime(["2024-01-04 09:05"]),
            "net_return": [0.02],
        }
    )

    write_performance_outputs(intraday, overnight, tmp_path)

    assert (tmp_path / "performance_subplots.png").stat().st_size > 0
    curves = pd.read_csv(tmp_path / "equity_curves.csv")
    assert list(curves.columns) == ["date", "intraday", "overnight", "combined"]
    assert curves["combined"].iloc[-1] > 1.0


def test_run_persists_selected_scheme_config(tmp_path, monkeypatch) -> None:
    raw = _ticker_frame("A005930", [100, 100, 101, 102, 106, 109, 112, 111], [100, 100, 100, 100, 500, 520, 540, 520])

    def fake_load_strategy_frame(_dataset: KrStock5mDataset, config: TechGammaConfig) -> pd.DataFrame:
        return build_features(raw, config)

    monkeypatch.setattr("scripts.run_tech_gamma_long_only.load_strategy_frame", fake_load_strategy_frame)

    output = run(TechGammaConfig(scheme="opening_range_vwap", range_end_hhmm="0920", exit_hhmm="0940"), tmp_path)

    config = pd.read_json(output / "config.json", typ="series")
    signals = pd.read_csv(output / "signals.csv")
    universe = pd.read_csv(output / "universe_tickers.csv")
    assert config["scheme"] == "opening_range_vwap"
    assert "signal_score" in signals.columns
    assert universe["ticker"].tolist() == ["A005930", "A000660"]


def test_run_intraday_only_skips_overnight_trades(tmp_path, monkeypatch) -> None:
    day1 = _ticker_frame("A005930", [100, 101, 102, 104, 107, 110, 114, 118], [100, 110, 120, 140, 300, 350, 400, 450])
    day2 = _ticker_frame("A005930", [126, 127, 128], [200, 200, 200])
    day2["ts"] = day2["ts"] + pd.Timedelta(days=1)
    raw = pd.concat([day1, day2], ignore_index=True)

    def fake_load_strategy_frame(_dataset: KrStock5mDataset, config: TechGammaConfig) -> pd.DataFrame:
        return build_features(raw, config)

    monkeypatch.setattr("scripts.run_tech_gamma_long_only.load_strategy_frame", fake_load_strategy_frame)

    output = run(
        TechGammaConfig(range_end_hhmm="0920", exit_hhmm="0940", overnight_enabled=False),
        tmp_path,
    )

    overnight = pd.read_csv(output / "overnight_trades.csv")
    assert overnight.empty


def test_parse_args_accepts_flow_only_filter(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_tech_gamma_long_only.py",
            "--factor-filter",
            "foreign_or_institution_positive",
        ],
    )

    from scripts.run_tech_gamma_long_only import parse_args

    args = parse_args()

    assert args.factor_filter == "foreign_or_institution_positive"
