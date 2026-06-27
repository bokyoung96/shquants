from __future__ import annotations

import pandas as pd

from scripts.run_tech_gamma_long_only import TechGammaConfig, build_features, simulate_intraday


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


def test_52w_high_breakout_positivity_option_filters_entries() -> None:
    positive_warmup = [_dated_day("A005930", close, offset) for offset, close in zip([7, 6, 5, 4], [100.0, 101.0, 102.0, 103.0], strict=True)]
    weak_warmup = [_dated_day("A000660", close, offset) for offset, close in zip([7, 6, 5, 4], [103.0, 102.0, 101.0, 100.0], strict=True)]
    breakout = [103, 103, 103, 103, 104, 104.5, 105, 105]
    config = TechGammaConfig(
        scheme="52w_high_breakout",
        range_end_hhmm="0920",
        use_positivity=True,
        positivity_lookback_days=3,
        min_daily_positivity=0.6,
    )

    frame = build_features(
        pd.concat(
            [*positive_warmup, *weak_warmup, _ticker_frame("A005930", breakout), _ticker_frame("A000660", breakout)],
            ignore_index=True,
        ),
        config,
    )
    trades = simulate_intraday(frame, config)

    assert "daily_positivity" in frame.columns
    assert trades["ticker"].tolist() == ["A005930"]
