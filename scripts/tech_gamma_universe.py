from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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
    active = membership.reindex(index=dates, columns=tickers).ffill().fillna(0).gt(0)
    active.index.name = "date"
    active.columns.name = "ticker"
    active_rows = active.stack(future_stack=True).rename("k200_member").reset_index()
    keyed = frame.merge(active_rows, on=["date", "ticker"], how="left", sort=False)
    return keyed.loc[keyed["k200_member"].fillna(False)].drop(columns="k200_member").reset_index(drop=True)
