from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

import pandas as pd

from backtesting.strategies.positivity import positivity_score

if TYPE_CHECKING:
    from scripts.run_tech_gamma_long_only import TechGammaConfig


class LongOnlyScheme(Protocol):
    name: str
    score_column: str
    signal_columns: tuple[str, ...]

    def build_features(self, raw: pd.DataFrame, config: TechGammaConfig) -> pd.DataFrame: ...

    def intraday_entry_ok(self, row: pd.Series, config: TechGammaConfig) -> bool: ...

    def overnight_entry_ok(self, row: pd.Series, config: TechGammaConfig) -> bool: ...


@dataclass(frozen=True, slots=True)
class TechGammaScheme:
    name: str = "tech_gamma"
    score_column: str = "gamma_score"
    signal_columns: tuple[str, ...] = (
        "vwap",
        "opening_high",
        "breakout_bps",
        "vwap_bps",
        "volume_spike",
        "gamma_score",
        "signal_score",
    )

    def build_features(self, raw: pd.DataFrame, config: TechGammaConfig) -> pd.DataFrame:
        frame = _base_features(raw, config)
        frame["gamma_score"] = _gamma_score(frame)
        frame["signal_score"] = frame["gamma_score"]
        return _with_next_bar(frame)

    def intraday_entry_ok(self, row: pd.Series, config: TechGammaConfig) -> bool:
        return bool(
            pd.notna(row["next_open"])
            and str(row["hhmm"]) > config.range_end_hhmm
            and str(row["hhmm"]) < config.exit_hhmm
            and float(row["breakout_bps"]) >= config.range_buffer_bps
            and float(row["close"]) > float(row["vwap"])
            and float(row["signal_score"]) >= config.min_score
        )

    def overnight_entry_ok(self, row: pd.Series, config: TechGammaConfig) -> bool:
        return bool(
            _has_base_signal(row)
            and float(row["signal_score"]) >= config.min_overnight_score
            and float(row["close"]) > float(row["vwap"])
        )


@dataclass(frozen=True, slots=True)
class OpeningRangeVwapScheme:
    name: str = "opening_range_vwap"
    score_column: str = "opening_range_vwap_score"
    signal_columns: tuple[str, ...] = (
        "vwap",
        "opening_high",
        "breakout_bps",
        "vwap_bps",
        "volume_spike",
        "opening_range_vwap_score",
        "signal_score",
    )

    def build_features(self, raw: pd.DataFrame, config: TechGammaConfig) -> pd.DataFrame:
        frame = _base_features(raw, config)
        frame["opening_range_vwap_score"] = _opening_range_vwap_score(frame)
        frame["signal_score"] = frame["opening_range_vwap_score"]
        return _with_next_bar(frame)

    def intraday_entry_ok(self, row: pd.Series, config: TechGammaConfig) -> bool:
        return bool(
            pd.notna(row["next_open"])
            and str(row["hhmm"]) > config.range_end_hhmm
            and str(row["hhmm"]) < config.exit_hhmm
            and float(row["breakout_bps"]) >= config.range_buffer_bps
            and float(row["vwap_bps"]) > 0.0
            and float(row["volume_spike"]) >= 1.2
            and float(row["signal_score"]) >= config.min_score
        )

    def overnight_entry_ok(self, row: pd.Series, config: TechGammaConfig) -> bool:
        return bool(
            _has_base_signal(row)
            and float(row["signal_score"]) >= config.min_overnight_score
            and float(row["high_close_pos"]) >= 0.65
        )


@dataclass(frozen=True, slots=True)
class HighBreakout52wScheme:
    name: str = "52w_high_breakout"
    score_column: str = "high_52w_breakout_score"
    signal_columns: tuple[str, ...] = (
        "vwap",
        "prior_52w_close_high",
        "breakout_52w_bps",
        "atr",
        "volume_spike",
        "high_52w_breakout_score",
        "signal_score",
    )

    def build_features(self, raw: pd.DataFrame, config: TechGammaConfig) -> pd.DataFrame:
        frame = _base_features(raw, config)
        frame = _with_daily_high_features(frame, config)
        if config.use_positivity and config.positivity_benchmark == "absolute":
            frame = _with_daily_positivity(frame, config)
        frame["high_52w_breakout_score"] = _high_breakout_score(frame)
        frame["signal_score"] = frame["high_52w_breakout_score"]
        return _with_next_bar(frame)

    def intraday_entry_ok(self, row: pd.Series, config: TechGammaConfig) -> bool:
        return bool(
            _has_high_breakout_signal(row)
            and pd.notna(row["next_open"])
            and str(row["hhmm"]) > config.range_end_hhmm
            and str(row["hhmm"]) < config.exit_hhmm
            and float(row["breakout_52w_bps"]) >= config.range_buffer_bps
            and _positivity_ok(row, config)
        )

    def overnight_entry_ok(self, row: pd.Series, config: TechGammaConfig) -> bool:
        return bool(
            _has_high_breakout_signal(row)
            and float(row["breakout_52w_bps"]) >= config.range_buffer_bps
            and float(row["signal_score"]) >= config.min_overnight_score
            and _positivity_ok(row, config)
        )


SCHEMES: Final[dict[str, LongOnlyScheme]] = {
    "52w_high_breakout": HighBreakout52wScheme(),
    "opening_range_vwap": OpeningRangeVwapScheme(),
    "tech_gamma": TechGammaScheme(),
}


def get_scheme(name: str) -> LongOnlyScheme:
    try:
        return SCHEMES[name]
    except KeyError as exc:
        available = ", ".join(scheme_names())
        raise KeyError(f"unknown scheme {name!r}; available: {available}") from exc


def scheme_names() -> tuple[str, ...]:
    return tuple(sorted(SCHEMES))


def _base_features(raw: pd.DataFrame, config: TechGammaConfig) -> pd.DataFrame:
    frame = raw.copy()
    frame["ts"] = pd.to_datetime(frame["ts"])
    frame = frame.sort_values(["ticker", "ts"]).reset_index(drop=True)
    frame["date"] = frame["ts"].dt.normalize()
    frame["hhmm"] = frame["ts"].dt.strftime("%H%M")
    grouped = frame.groupby(["ticker", "date"], sort=False, group_keys=False)
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    traded_value = typical * frame["volume"]
    volume_sum = frame["volume"].groupby([frame["ticker"], frame["date"]]).cumsum()
    frame["vwap"] = traded_value.groupby([frame["ticker"], frame["date"]]).cumsum().divide(
        volume_sum.replace(0.0, float("nan"))
    )
    frame["ret_15m"] = grouped["close"].pct_change(3)
    frame["ret_30m"] = grouped["close"].pct_change(6)
    frame["volume_base"] = grouped["volume"].transform(lambda item: item.shift(1).rolling(6, min_periods=2).mean())
    frame["volume_spike"] = frame["volume"].divide(frame["volume_base"].replace(0.0, float("nan"))).fillna(0.0)
    frame = frame.join(_opening_ranges(frame, config), how="left")
    frame["breakout_bps"] = (frame["close"] / frame["opening_high"] - 1.0) * 10_000.0
    frame["vwap_bps"] = (frame["close"] / frame["vwap"] - 1.0) * 10_000.0
    price_range = (frame["high"] - frame["low"]).replace(0.0, float("nan"))
    frame["high_close_pos"] = (frame["close"] - frame["low"]).divide(price_range).fillna(0.0)
    frame["atr"] = _intraday_atr(frame, config)
    frame["previous_intraday_close"] = grouped["close"].shift(1)
    return frame


def _with_next_bar(frame: pd.DataFrame) -> pd.DataFrame:
    frame["next_ts"] = frame.groupby("ticker")["ts"].shift(-1)
    frame["next_open"] = frame.groupby("ticker")["open"].shift(-1)
    return frame


def _opening_ranges(frame: pd.DataFrame, config: TechGammaConfig) -> pd.DataFrame:
    opening = frame[frame["hhmm"].le(config.range_end_hhmm)]
    highs = opening.groupby(["ticker", "date"])["high"].max().rename("opening_high")
    lows = opening.groupby(["ticker", "date"])["low"].min().rename("opening_low")
    keys = frame.set_index(["ticker", "date"]).index
    return pd.DataFrame({"opening_high": highs.reindex(keys).to_numpy(), "opening_low": lows.reindex(keys).to_numpy()}, index=frame.index)


def _gamma_score(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["breakout_bps"].clip(lower=0.0).divide(10.0)
        + frame["vwap_bps"].clip(lower=0.0).divide(12.0)
        + frame["ret_15m"].fillna(0.0).clip(lower=0.0).mul(100.0)
        + frame["ret_30m"].fillna(0.0).clip(lower=0.0).mul(80.0)
        + frame["volume_spike"].clip(upper=5.0).sub(1.0).clip(lower=0.0)
        + frame["high_close_pos"].clip(lower=0.0, upper=1.0)
    )


def _opening_range_vwap_score(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["breakout_bps"].clip(lower=0.0).divide(12.0)
        + frame["vwap_bps"].clip(lower=0.0).divide(10.0)
        + frame["volume_spike"].clip(upper=4.0).sub(1.0).clip(lower=0.0).mul(1.5)
        + frame["high_close_pos"].clip(lower=0.0, upper=1.0)
    )


def _with_daily_high_features(frame: pd.DataFrame, config: TechGammaConfig) -> pd.DataFrame:
    daily = frame.groupby(["ticker", "date"], sort=True)["close"].last().rename("daily_close").reset_index()
    grouped = daily.groupby("ticker", sort=False)["daily_close"]
    daily["prior_52w_close_high"] = grouped.transform(lambda item: item.shift(1).rolling(252, min_periods=1).max())
    daily = daily[["ticker", "date", "prior_52w_close_high"]]
    enriched = frame.merge(daily, on=["ticker", "date"], how="left", sort=False)
    enriched["breakout_52w_bps"] = (enriched["close"] / enriched["prior_52w_close_high"] - 1.0) * 10_000.0
    return enriched


def _with_daily_positivity(frame: pd.DataFrame, config: TechGammaConfig) -> pd.DataFrame:
    daily = frame.groupby(["ticker", "date"], sort=True)["close"].last().rename("daily_close").reset_index()
    returns = daily.pivot(index="date", columns="ticker", values="daily_close").pct_change(fill_method=None)
    score = positivity_score(returns, lookback=config.positivity_lookback_days, min_periods=config.positivity_lookback_days)
    daily_pos = score.shift(1).stack().rename("daily_positivity").reset_index()
    return frame.merge(daily_pos, on=["ticker", "date"], how="left", sort=False)


def _intraday_atr(frame: pd.DataFrame, config: TechGammaConfig) -> pd.Series:
    previous_close = frame.groupby("ticker", sort=False)["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.groupby(frame["ticker"], sort=False).transform(
        lambda item: item.rolling(config.atr_lookback_bars, min_periods=1).mean()
    )


def _high_breakout_score(frame: pd.DataFrame) -> pd.Series:
    score = frame["breakout_52w_bps"].clip(lower=0.0).divide(10.0)
    if "daily_positivity" in frame.columns:
        score = score + frame["daily_positivity"].fillna(0.0).clip(lower=0.0).mul(5.0)
    return score + frame["volume_spike"].clip(upper=5.0).sub(1.0).clip(lower=0.0)


def _positivity_ok(row: pd.Series, config: TechGammaConfig) -> bool:
    if "factor_filter_ok" in row.index and not bool(row["factor_filter_ok"]):
        return False
    if not config.use_positivity:
        return True
    if "positivity_filter_ok" in row.index:
        return bool(row["positivity_filter_ok"])
    return bool(pd.notna(row["daily_positivity"]) and float(row["daily_positivity"]) >= config.min_daily_positivity)


def _has_base_signal(row: pd.Series) -> bool:
    return bool(
        pd.notna(row["signal_score"])
        and pd.notna(row["vwap"])
        and pd.notna(row["opening_high"])
    )


def _has_high_breakout_signal(row: pd.Series) -> bool:
    return bool(
        pd.notna(row["signal_score"])
        and pd.notna(row["prior_52w_close_high"])
        and pd.notna(row["atr"])
        and float(row["close"]) > float(row["prior_52w_close_high"])
        and (
            pd.isna(row["previous_intraday_close"])
            or float(row["previous_intraday_close"]) <= float(row["prior_52w_close_high"])
        )
    )
