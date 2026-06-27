from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


FIELD_NAMES = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
}


@dataclass(frozen=True, slots=True)
class KrStock5mDataset:
    root: Path

    def field_path(self, month: str, field: str) -> Path:
        year, month_number = month.split("-", maxsplit=1)
        return self.root / f"year={year}" / f"month={month_number}" / f"{field}.parquet"


def normalize_ticker(ticker: str) -> str:
    stripped = ticker.strip().upper()
    if stripped.startswith("A"):
        return stripped
    return f"A{stripped.zfill(6)}"


def available_months(dataset: KrStock5mDataset) -> list[str]:
    months: list[str] = []
    for year_dir in sorted(dataset.root.glob("year=*")):
        year = year_dir.name.removeprefix("year=")
        for month_dir in sorted(year_dir.glob("month=*")):
            month = month_dir.name.removeprefix("month=")
            months.append(f"{year}-{month}")
    return months


def months_between(start: str | pd.Timestamp, end: str | pd.Timestamp) -> list[str]:
    start_month = pd.Timestamp(start).to_period("M")
    end_month = pd.Timestamp(end).to_period("M")
    return [str(month) for month in pd.period_range(start_month, end_month, freq="M")]


def read_ticker_bars(
    dataset: KrStock5mDataset,
    ticker: str,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    normalized = normalize_ticker(ticker)
    frames = [
        _read_ticker_month(dataset, normalized, month)
        for month in months_between(start, end)
        if dataset.field_path(month, "c").exists()
    ]
    if not frames:
        return _empty_bars()

    bars = pd.concat(frames).sort_index()
    bars = bars.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    bars.insert(0, "ticker", normalized)
    return bars.reset_index().rename(columns={bars.index.name or "index": "ts"})


def read_tickers_bars(
    dataset: KrStock5mDataset,
    tickers: tuple[str, ...],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    normalized = tuple(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers))
    frames = [
        _read_tickers_month(dataset, normalized, month)
        for month in months_between(start, end)
        if dataset.field_path(month, "c").exists()
    ]
    if not frames:
        return _empty_bars()

    bars = pd.concat(frames, ignore_index=True).sort_values(["ts", "ticker"])
    window = bars["ts"].between(pd.Timestamp(start), pd.Timestamp(end))
    return bars.loc[window].reset_index(drop=True)


def summarize_dataset(dataset: KrStock5mDataset) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    for month in available_months(dataset):
        close_path = dataset.field_path(month, "c")
        if close_path.exists():
            close = pd.read_parquet(close_path, engine="pyarrow")
            rows.append(
                {
                    "month": month,
                    "rows": len(close),
                    "tickers": len(close.columns),
                    "fields": sum(dataset.field_path(month, field).exists() for field in FIELD_NAMES),
                }
            )
    return pd.DataFrame(rows, columns=["month", "rows", "tickers", "fields"])


def _read_ticker_month(dataset: KrStock5mDataset, ticker: str, month: str) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for field, name in FIELD_NAMES.items():
        frame = pd.read_parquet(dataset.field_path(month, field), columns=[ticker], engine="pyarrow")
        columns[name] = frame[ticker]
    bars = pd.DataFrame(columns)
    bars.index.name = "ts"
    return bars


def _read_tickers_month(dataset: KrStock5mDataset, tickers: tuple[str, ...], month: str) -> pd.DataFrame:
    available = set(_parquet_columns(dataset.field_path(month, "c")))
    selected = [ticker for ticker in tickers if ticker in available]
    if not selected:
        return _empty_bars()
    fields = {
        name: pd.read_parquet(dataset.field_path(month, field), columns=selected, engine="pyarrow")
        for field, name in FIELD_NAMES.items()
    }
    rows: list[pd.DataFrame] = []
    for ticker in selected:
        bars = pd.DataFrame({name: frame[ticker] for name, frame in fields.items()})
        bars.insert(0, "ticker", ticker)
        rows.append(bars.reset_index().rename(columns={bars.index.name or "index": "ts"}))
    return pd.concat(rows, ignore_index=True)


def _parquet_columns(path: Path) -> tuple[str, ...]:
    schema = pq.read_schema(path)
    return tuple(schema.names)


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts", "ticker", "open", "high", "low", "close", "volume"])
