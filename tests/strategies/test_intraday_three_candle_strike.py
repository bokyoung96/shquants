import pandas as pd
import pytest

from backtesting.strategies.intraday_three_candle_strike import (
    BacktestConfig,
    compute_indicators,
    detect_three_candle_strike,
    run_backtest,
    run_backtest_from_signals,
)


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows).assign(
        ts=lambda df: pd.to_datetime(df["ts"], utc=True),
        trade_date_kst=lambda df: pd.to_datetime(df["trade_date_kst"]).dt.date,
    )


def test_detect_three_candle_strike_flags_long_and_short_reversals() -> None:
    df = _frame(
        [
            {"ts": "2024-01-02 00:00Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0900", "open": 104.0, "high": 105.0, "low": 102.0, "close": 103.0},
            {"ts": "2024-01-02 00:01Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0901", "open": 103.0, "high": 104.0, "low": 101.0, "close": 102.0},
            {"ts": "2024-01-02 00:02Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0902", "open": 102.0, "high": 103.0, "low": 100.0, "close": 101.0},
            {"ts": "2024-01-02 00:03Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0903", "open": 100.5, "high": 105.5, "low": 100.0, "close": 104.5},
            {"ts": "2024-01-02 00:04Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0904", "open": 101.0, "high": 102.0, "low": 100.0, "close": 102.0},
            {"ts": "2024-01-02 00:05Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0905", "open": 102.0, "high": 103.0, "low": 101.0, "close": 103.0},
            {"ts": "2024-01-02 00:06Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0906", "open": 103.0, "high": 104.0, "low": 102.0, "close": 104.0},
            {"ts": "2024-01-02 00:07Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0907", "open": 104.5, "high": 105.0, "low": 99.0, "close": 100.5},
        ]
    )

    signal = detect_three_candle_strike(df)

    assert signal.tolist() == [0, 0, 0, 1, 0, 0, 0, -1]


def test_compute_indicators_applies_trend_and_range_filters() -> None:
    close = [100.0 + i * 0.2 for i in range(70)]
    start = pd.Timestamp("2024-01-02 00:00Z")
    df = _frame(
        [
            {
                "ts": start + pd.Timedelta(minutes=i),
                "trade_date_kst": "2024-01-02",
                "hhmm_kst": f"{900 + i:04d}",
                "open": value - 0.1,
                "high": value + 0.2,
                "low": value - 0.2,
                "close": value,
            }
            for i, value in enumerate(close)
        ]
    )

    out = compute_indicators(df, slope_lookback=20, slope_threshold=0.00001)

    assert out["allow_long"].iloc[-1]
    assert not out["allow_short"].iloc[-1]
    assert out["is_trending"].iloc[-1]


def test_run_backtest_enters_next_bar_and_exits_on_fixed_target() -> None:
    rows = [
        {"ts": "2024-01-02 00:00Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0900", "open": 104.0, "high": 105.0, "low": 102.0, "close": 103.0},
        {"ts": "2024-01-02 00:01Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0901", "open": 103.0, "high": 104.0, "low": 101.0, "close": 102.0},
        {"ts": "2024-01-02 00:02Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0902", "open": 102.0, "high": 103.0, "low": 100.0, "close": 101.0},
        {"ts": "2024-01-02 00:03Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0903", "open": 100.5, "high": 105.5, "low": 100.0, "close": 104.5},
        {"ts": "2024-01-02 00:04Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0904", "open": 104.5, "high": 105.0, "low": 104.0, "close": 104.8},
        {"ts": "2024-01-02 00:05Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0905", "open": 104.8, "high": 118.5, "low": 104.7, "close": 115.0},
    ]
    df = _frame(rows)
    config = BacktestConfig(
        slope_threshold=0.0,
        rr=3.0,
        atr_buffer_fraction=0.0,
        round_trip_cost_bps=0.0,
        use_filters=False,
        trailing=False,
    )

    result = run_backtest(df, config)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["side"] == "long"
    assert trade["entry_price"] == pytest.approx(104.5)
    assert trade["exit_reason"] == "target"
    assert trade["r_multiple"] == pytest.approx(3.0)


def test_run_backtest_from_signals_reuses_indicator_frame() -> None:
    rows = [
        {"ts": "2024-01-02 00:00Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0900", "open": 104.0, "high": 105.0, "low": 102.0, "close": 103.0},
        {"ts": "2024-01-02 00:01Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0901", "open": 103.0, "high": 104.0, "low": 101.0, "close": 102.0},
        {"ts": "2024-01-02 00:02Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0902", "open": 102.0, "high": 103.0, "low": 100.0, "close": 101.0},
        {"ts": "2024-01-02 00:03Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0903", "open": 100.5, "high": 105.5, "low": 100.0, "close": 104.5},
        {"ts": "2024-01-02 00:04Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0904", "open": 104.5, "high": 105.0, "low": 104.0, "close": 104.8},
        {"ts": "2024-01-02 00:05Z", "trade_date_kst": "2024-01-02", "hhmm_kst": "0905", "open": 104.8, "high": 118.5, "low": 104.7, "close": 115.0},
    ]
    df = _frame(rows)
    config = BacktestConfig(slope_threshold=0.0, use_filters=False, trailing=False, round_trip_cost_bps=0.0)
    signals = compute_indicators(df, slope_threshold=0.0)

    direct = run_backtest(df, config)
    reused = run_backtest_from_signals(signals, config)

    pd.testing.assert_frame_equal(reused.trades, direct.trades)
    assert reused.summary == direct.summary
