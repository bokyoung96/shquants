from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from scripts.run_tech_gamma_long_only import TechGammaConfig


def kospi200_tickers(parquet_dir: Path, config: TechGammaConfig) -> tuple[str, ...]:
    membership = pd.read_parquet(parquet_dir / "qw_k200_yn.parquet", engine="pyarrow")
    window = membership.loc[pd.Timestamp(config.start) : pd.Timestamp(config.end)]
    if window.empty:
        window = membership.loc[: pd.Timestamp(config.end)]
    if config.universe in {"kospi200_ever", "kospi200_historical"}:
        active = window.gt(0).any(axis=0)
    else:
        active = window.ffill().iloc[-1].gt(0)
    return tuple(sorted(str(ticker) for ticker in active.index[active]))


def filter_kospi200_historical_members(frame: pd.DataFrame, parquet_dir: Path) -> pd.DataFrame:
    if frame.empty:
        return frame
    membership = pd.read_parquet(parquet_dir / "qw_k200_yn.parquet", engine="pyarrow")
    tickers = pd.Index(frame["ticker"].drop_duplicates())
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(frame["date"]).unique()))
    active = membership.reindex(index=membership.index.union(dates), columns=tickers).ffill().reindex(dates).fillna(0).gt(0)
    active_values = active.to_numpy(dtype=bool)
    date_codes = pd.Categorical(pd.to_datetime(frame["date"]), categories=dates).codes
    ticker_codes = pd.Categorical(frame["ticker"], categories=tickers).codes
    valid = (date_codes >= 0) & (ticker_codes >= 0)
    keep = np.zeros(len(frame), dtype=bool)
    keep[valid] = active_values[date_codes[valid], ticker_codes[valid]]
    return frame.loc[keep].reset_index(drop=True)
