from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ASSETALLOCATION_TICKERS: dict[str, str] = {
    "USYC2Y10 Index": "curve_2y10",
    "USYC3M2Y Index": "curve_3m2y",
    "USDJPY Curncy": "usdjpy",
    "USGG10YR Index": "us10y",
    "GC1 Comdty": "gold",
    "CL1 Comdty": "oil",
    "HG1 Comdty": "copper",
    "SPX Index": "spx",
    "INDU Index": "dow",
    "RTY Index": "russell2000",
    "SPY US Equity": "spy",
    "IEF US Equity": "ief",
}

PRICE_TICKERS: tuple[str, ...] = (
    "USDJPY Curncy",
    "GC1 Comdty",
    "HG1 Comdty",
    "SPX Index",
    "INDU Index",
    "RTY Index",
    "SPY US Equity",
    "IEF US Equity",
)

LEVEL_TICKERS: tuple[str, ...] = (
    "USYC2Y10 Index",
    "USYC3M2Y Index",
    "USGG10YR Index",
    "CL1 Comdty",
)

OHLC_STEMS: tuple[str, ...] = ("open", "high", "low", "close")


@dataclass(frozen=True, slots=True)
class FeatureBuildResult:
    feature_path: Path
    target_path: Path
    rows: int
    feature_columns: int
    target_columns: int


@dataclass(slots=True)
class FeatureBuilder:
    data_dir: Path
    output_dir: Path
    horizons: tuple[int, ...] = (5, 20, 60)
    volatility_windows: tuple[int, ...] = (20, 60)
    zscore_window: int = 252
    bond_duration_years: float = 7.0
    target_entry_lag: int = 1

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.horizons = _normalize_windows(self.horizons, "horizons")
        self.volatility_windows = _normalize_windows(self.volatility_windows, "volatility_windows")
        if self.zscore_window <= 1:
            raise ValueError("zscore_window must be greater than 1")
        if self.target_entry_lag < 0:
            raise ValueError("target_entry_lag must not be negative")

    def build(self) -> FeatureBuildResult:
        frames = self.load_ohlc()
        features = self.build_features(frames)
        targets = self.build_targets(frames)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        feature_path = self.output_dir / "features.parquet"
        target_path = self.output_dir / "targets.parquet"
        features.to_parquet(feature_path, engine="pyarrow")
        targets.to_parquet(target_path, engine="pyarrow")

        return FeatureBuildResult(
            feature_path=feature_path,
            target_path=target_path,
            rows=len(features),
            feature_columns=len(features.columns),
            target_columns=len(targets.columns),
        )

    def load_ohlc(self) -> dict[str, pd.DataFrame]:
        frames = {
            stem: pd.read_parquet(self.data_dir / f"{stem}.parquet", engine="pyarrow").sort_index()
            for stem in OHLC_STEMS
        }
        self._validate_frames(frames)
        return frames

    def build_features(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = frames["open"]
        high = frames["high"]
        low = frames["low"]
        close = frames["close"]
        columns: dict[str, pd.Series] = {}

        for ticker in PRICE_TICKERS:
            slug = ASSETALLOCATION_TICKERS[ticker]
            price = close[ticker].astype(float)
            daily_return = price.pct_change()
            columns[f"{slug}_ret_1d"] = daily_return
            for horizon in self.horizons:
                columns[f"{slug}_mom_{horizon}d"] = price.pct_change(horizon)
            for window in self.volatility_windows:
                columns[f"{slug}_vol_{window}d"] = daily_return.rolling(window).std() * np.sqrt(252.0)
            columns[f"{slug}_drawdown_{max(self.horizons)}d"] = price / price.rolling(max(self.horizons)).max() - 1.0

        for ticker in LEVEL_TICKERS:
            slug = ASSETALLOCATION_TICKERS[ticker]
            level = close[ticker].astype(float)
            columns[f"{slug}_level"] = level
            for horizon in self.horizons:
                suffix = "bp" if ticker == "USGG10YR Index" else "level"
                scale = 100.0 if ticker == "USGG10YR Index" else 1.0
                columns[f"{slug}_chg_{horizon}d_{suffix}"] = level.diff(horizon) * scale
            for window in self.volatility_windows:
                columns[f"{slug}_chg_vol_{window}d"] = level.diff().rolling(window).std()
            columns[f"{slug}_z_{self.zscore_window}d"] = _rolling_zscore(level, self.zscore_window)

        for ticker, slug in ASSETALLOCATION_TICKERS.items():
            columns[f"{slug}_hl_range"] = (high[ticker].astype(float) - low[ticker].astype(float)) / close[ticker].abs().replace(0.0, np.nan)
            columns[f"{slug}_oc_change"] = (close[ticker].astype(float) - open_[ticker].astype(float)) / open_[ticker].abs().replace(0.0, np.nan)

        bond_proxy = self._us10y_proxy_return(close)
        spx_return = close["SPX Index"].astype(float).pct_change()
        spy_return = close["SPY US Equity"].astype(float).pct_change()
        ief_return = close["IEF US Equity"].astype(float).pct_change()
        columns["us10y_proxy_ret_1d"] = bond_proxy
        columns["spx_excess_us10y_proxy_ret_1d"] = spx_return - bond_proxy
        columns["spy_excess_ief_ret_1d"] = spy_return - ief_return

        for horizon in self.horizons:
            spx_momentum = close["SPX Index"].astype(float).pct_change(horizon)
            bond_momentum = bond_proxy.rolling(horizon).sum()
            columns[f"spx_vs_us10y_proxy_mom_{horizon}d"] = spx_momentum - bond_momentum
            columns[f"spy_vs_ief_mom_{horizon}d"] = (
                close["SPY US Equity"].astype(float).pct_change(horizon)
                - close["IEF US Equity"].astype(float).pct_change(horizon)
            )

        columns["curve_2y10_inverted"] = close["USYC2Y10 Index"].lt(0.0).astype(float)
        columns["curve_3m2y_inverted"] = close["USYC3M2Y Index"].lt(0.0).astype(float)
        columns["gold_vs_spx_mom_20d"] = close["GC1 Comdty"].pct_change(20) - close["SPX Index"].pct_change(20)
        columns["copper_vs_gold_mom_20d"] = close["HG1 Comdty"].pct_change(20) - close["GC1 Comdty"].pct_change(20)
        columns["equity_breadth_mom_20d"] = pd.concat(
            [
                close["SPX Index"].pct_change(20),
                close["INDU Index"].pct_change(20),
                close["RTY Index"].pct_change(20),
            ],
            axis=1,
        ).mean(axis=1)

        feature_frame = pd.DataFrame(columns, index=close.index)
        feature_frame.index.name = "date"
        return feature_frame.replace([np.inf, -np.inf], np.nan)

    def build_targets(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = frames["close"]
        spy = close["SPY US Equity"].astype(float)
        ief = close["IEF US Equity"].astype(float)
        targets: dict[str, pd.Series] = {}

        for horizon in self.horizons:
            entry_lag = int(self.target_entry_lag)
            exit_lag = entry_lag + int(horizon)
            spy_forward = spy.shift(-exit_lag) / spy.shift(-entry_lag) - 1.0
            ief_forward = ief.shift(-exit_lag) / ief.shift(-entry_lag) - 1.0
            excess = spy_forward - ief_forward
            targets[f"target_spy_fwd_{horizon}d"] = spy_forward
            targets[f"target_ief_fwd_{horizon}d"] = ief_forward
            targets[f"target_spy_excess_ief_fwd_{horizon}d"] = excess
            targets[f"target_spy_over_ief_direction_{horizon}d"] = excess.gt(0.0).where(excess.notna())

        target_frame = pd.DataFrame(targets, index=close.index)
        target_frame.index.name = "date"
        return target_frame.replace([np.inf, -np.inf], np.nan)

    def _validate_frames(self, frames: dict[str, pd.DataFrame]) -> None:
        missing_files = [stem for stem in OHLC_STEMS if stem not in frames]
        if missing_files:
            raise ValueError(f"missing OHLC frames: {', '.join(missing_files)}")

        expected_index = frames["close"].index
        missing_tickers = sorted(set(ASSETALLOCATION_TICKERS) - set(frames["close"].columns))
        if missing_tickers:
            raise ValueError(f"missing required assetallocation tickers: {', '.join(missing_tickers)}")

        for stem, frame in frames.items():
            if not frame.index.equals(expected_index):
                raise ValueError(f"{stem}.parquet index does not match close.parquet")
            missing = sorted(set(ASSETALLOCATION_TICKERS) - set(frame.columns))
            if missing:
                raise ValueError(f"{stem}.parquet is missing required assetallocation tickers: {', '.join(missing)}")

    def _us10y_proxy_return(self, close: pd.DataFrame) -> pd.Series:
        yield_decimal_change = close["USGG10YR Index"].astype(float).diff() / 100.0
        proxy = -float(self.bond_duration_years) * yield_decimal_change
        proxy.name = "us10y_proxy_ret_1d"
        return proxy


def default_builder() -> FeatureBuilder:
    root = Path(__file__).resolve().parents[1]
    return FeatureBuilder(
        data_dir=root / "data",
        output_dir=root / "feature",
    )


def main() -> None:
    result = default_builder().build()
    print(f"features={result.feature_path}")
    print(f"targets={result.target_path}")
    print(f"rows={result.rows}")
    print(f"feature_columns={result.feature_columns}")
    print(f"target_columns={result.target_columns}")


def _normalize_windows(values: Iterable[int], name: str) -> tuple[int, ...]:
    windows = tuple(int(value) for value in values)
    if not windows:
        raise ValueError(f"{name} must not be empty")
    if any(value <= 0 for value in windows):
        raise ValueError(f"{name} must contain positive integers")
    return tuple(sorted(set(windows)))


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window)
    mean = rolling.mean()
    std = rolling.std().replace(0.0, np.nan)
    return (series - mean) / std


if __name__ == "__main__":
    main()
