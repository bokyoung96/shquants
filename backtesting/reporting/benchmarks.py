from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtesting.catalog import DataCatalog, DatasetId
from backtesting.data import ParquetStore
from backtesting.ingest import IngestJob
from backtesting.ingest.io import find_raw_path
from root import ROOT
from .models import BenchmarkConfig


@dataclass(frozen=True, slots=True)
class BenchmarkSeries:
    name: str
    prices: pd.Series
    returns: pd.Series


class BenchmarkRepository:
    def __init__(self, prices: pd.DataFrame) -> None:
        self.prices = prices.sort_index()

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "BenchmarkRepository":
        return cls(prices=frame)

    @classmethod
    def default(cls) -> "BenchmarkRepository":
        return cls(prices=_load_default_frame(DatasetId.QW_BM))

    def load_series(self, config: BenchmarkConfig, start: str, end: str) -> BenchmarkSeries:
        prices = self.prices.loc[start:end, config.code].astype(float).rename(config.name).rename_axis("date")
        returns = prices.pct_change().fillna(0.0).rename(config.name).rename_axis("date")
        return BenchmarkSeries(name=config.name, prices=prices, returns=returns)

    def load_returns(self, config: BenchmarkConfig, start: str, end: str) -> pd.Series:
        return self.load_series(config, start, end).returns


class SectorRepository:
    def __init__(
        self,
        sector: pd.DataFrame,
        prices: pd.DataFrame | None = None,
        *,
        sector_name_map: dict[str, str] | None = None,
        stock_name_map: dict[str, str] | None = None,
    ) -> None:
        self.sector_name_map = {str(key): str(value) for key, value in (sector_name_map or {}).items()}
        self.stock_name_map = {self._normalize_symbol_key(key): str(value) for key, value in (stock_name_map or {}).items()}
        self.sector = sector.sort_index().apply(lambda column: column.map(self._display_sector_label))
        self.prices = prices.sort_index() if prices is not None else None

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        prices: pd.DataFrame | None = None,
        *,
        sector_name_map: dict[str, str] | None = None,
        stock_name_map: dict[str, str] | None = None,
    ) -> "SectorRepository":
        return cls(
            sector=frame,
            prices=prices,
            sector_name_map=sector_name_map,
            stock_name_map=stock_name_map,
        )

    @classmethod
    def from_historical_excel(
        cls,
        path: Path,
        prices: pd.DataFrame | None = None,
        *,
        date_column: str = "DATE",
        symbol_column: str = "TICKER",
        sector_column: str = "GICS_SECTOR_LV1_NAME",
        sector_name_map: dict[str, str] | None = None,
        stock_name_map: dict[str, str] | None = None,
    ) -> "SectorRepository":
        return cls(
            sector=_read_historical_sector_frame(
                path,
                date_column=date_column,
                symbol_column=symbol_column,
                sector_column=sector_column,
            ),
            prices=prices,
            sector_name_map=sector_name_map,
            stock_name_map=stock_name_map,
        )

    @classmethod
    def default(cls) -> "SectorRepository":
        sector_name_map, stock_name_map = _load_display_name_maps(ROOT.raw_path / "map.xlsx")
        return cls(
            sector=_load_default_frame(DatasetId.QW_WICS_SEC_BIG),
            prices=_load_default_frame(DatasetId.QW_ADJ_C),
            sector_name_map=sector_name_map,
            stock_name_map=stock_name_map,
        )

    def display_symbol(self, symbol: str) -> str:
        raw = str(symbol)
        stock_name = self.stock_name_map.get(self._normalize_symbol_key(raw))
        if not stock_name:
            return raw
        ticker = raw[1:] if raw.startswith("A") and raw[1:].isdigit() else raw
        return f"{stock_name} ({ticker})"

    def latest_sector_row(self, as_of: pd.Timestamp) -> pd.Series:
        sector_history = self.sector.loc[:as_of]
        if sector_history.empty:
            raise KeyError(f"no sector mapping available on or before {as_of.date()}")
        return sector_history.iloc[-1]

    def latest_sector_counts(self, weights: pd.DataFrame) -> pd.Series:
        aligned = self._latest_aligned_weights(weights)
        counts = (
            aligned.loc[aligned["weight"].ne(0.0)]
            .groupby("sector", sort=False)
            .size()
            .astype(float)
            .rename("count")
        )
        counts.index.name = "sector"
        return counts

    def latest_sector_weights(self, weights: pd.DataFrame) -> pd.Series:
        aligned = self._latest_aligned_weights(weights)
        exposure = aligned.groupby("sector", sort=False)["weight"].sum()
        return exposure.sort_values(ascending=False).rename_axis(None).rename(None)

    def sector_weight_timeseries(self, weights: pd.DataFrame) -> pd.DataFrame:
        records: list[pd.Series] = []
        for date, row in weights.fillna(0.0).sort_index().iterrows():
            grouped = self._group_row_by_sector(pd.Timestamp(date), row.astype(float))
            grouped.name = pd.Timestamp(date)
            records.append(grouped)

        if not records:
            return pd.DataFrame()

        frame = pd.DataFrame(records).fillna(0.0).sort_index()
        frame.index.name = "date"
        return frame

    def sector_contribution_timeseries(self, qty: pd.DataFrame, equity: pd.Series) -> pd.DataFrame:
        if self.prices is None:
            return pd.DataFrame()

        records: list[pd.Series] = []
        quantities = qty.fillna(0.0).astype(float).sort_index()
        aligned_prices = self.prices.reindex(columns=quantities.columns).reindex(quantities.index).astype(float)

        for index, date in enumerate(quantities.index):
            timestamp = pd.Timestamp(date)
            if index == 0:
                grouped = self._group_row_by_sector(timestamp, pd.Series(0.0, index=quantities.columns, dtype=float))
                grouped.name = timestamp
                records.append(grouped)
                continue

            prev_date = pd.Timestamp(quantities.index[index - 1])
            prev_equity = float(equity.reindex(quantities.index).iloc[index - 1])
            if abs(prev_equity) < 1e-12:
                grouped = self._group_row_by_sector(timestamp, pd.Series(0.0, index=quantities.columns, dtype=float))
                grouped.name = timestamp
                records.append(grouped)
                continue

            current_qty = quantities.iloc[index]
            previous_qty = quantities.iloc[index - 1]
            current_close = aligned_prices.iloc[index].fillna(0.0)
            previous_close = aligned_prices.iloc[index - 1].fillna(0.0)

            holding_pnl = current_qty.mul(current_close.sub(previous_close), fill_value=0.0).fillna(0.0)
            holding_contribution = holding_pnl.div(prev_equity)
            sector_holding = self._group_row_by_sector(timestamp, holding_contribution.astype(float)).fillna(0.0)

            actual_net_return = float(equity.reindex(quantities.index).pct_change().fillna(0.0).iloc[index])
            gross_total_return = float(sector_holding.sum())
            residual_return = actual_net_return - gross_total_return

            trade_value = current_qty.sub(previous_qty).abs().mul(current_close, fill_value=0.0).fillna(0.0)
            sector_trade = self._group_row_by_sector(timestamp, trade_value.astype(float)).fillna(0.0)
            sector_exposure = self._group_row_by_sector(timestamp, current_qty.abs().mul(current_close).astype(float)).fillna(0.0)
            allocation_basis = sector_trade.where(sector_trade.gt(0.0), 0.0)
            if float(allocation_basis.sum()) <= 1e-12:
                allocation_basis = sector_exposure.where(sector_exposure.gt(0.0), 0.0)

            if float(allocation_basis.sum()) > 1e-12:
                sector_residual = allocation_basis.div(float(allocation_basis.sum())).mul(residual_return)
            else:
                sector_residual = pd.Series(0.0, index=sector_holding.index, dtype=float)

            grouped = sector_holding.add(sector_residual, fill_value=0.0).fillna(0.0)
            grouped.name = timestamp
            records.append(grouped)

        if not records:
            return pd.DataFrame()

        contributions = pd.DataFrame(records).fillna(0.0).sort_index().cumsum()
        contributions.index.name = "date"
        return contributions

    def _latest_aligned_weights(self, weights: pd.DataFrame) -> pd.DataFrame:
        latest_date = weights.index.max()
        latest_weight_row = weights.loc[:latest_date].iloc[-1].astype(float)
        latest_sector_row = self.latest_sector_row(latest_date)
        return pd.DataFrame(
            {
                "sector": latest_sector_row.reindex(latest_weight_row.index),
                "weight": latest_weight_row,
            }
        ).dropna(subset=["sector"])

    def _group_row_by_sector(self, as_of: pd.Timestamp, weights: pd.Series) -> pd.Series:
        try:
            sector_row = self.latest_sector_row(as_of)
        except KeyError:
            return pd.Series(dtype=float)
        aligned = pd.DataFrame(
            {
                "sector": sector_row.reindex(weights.index),
                "weight": weights,
            }
        ).dropna(subset=["sector"])
        grouped = aligned.groupby("sector", sort=False)["weight"].sum()
        grouped.index.name = None
        return grouped

    def _display_sector_label(self, value: object) -> object:
        if value is None or pd.isna(value):
            return value
        return self.sector_name_map.get(str(value), value)

    @staticmethod
    def _normalize_symbol_key(symbol: str) -> str:
        raw = str(symbol).strip().upper()
        if not raw:
            return raw
        if raw.startswith("A") and raw[1:].isdigit():
            return raw
        if raw.isdigit():
            return f"A{raw.zfill(6)}"
        return raw


def default_repositories_for_universe(
    universe_id: str | None,
    *,
    sector_source: str = "wics",
) -> tuple[BenchmarkRepository, SectorRepository]:
    sector_source_value = str(sector_source).strip().lower()
    if sector_source_value not in {"wics", "gics"}:
        raise ValueError(f"unsupported sector_source: {sector_source}")

    if universe_id == "kosdaq150":
        sector_name_map, stock_name_map = _load_display_name_maps(ROOT.raw_path / "map.xlsx")
        if sector_source_value == "gics":
            gics_path = _resolve_kosdaq_gics_level_path(ROOT.raw_path, level="lv1")
            if gics_path.exists():
                sector_repo = SectorRepository.from_historical_excel(
                    gics_path,
                    prices=_load_default_frame(DatasetId.QW_KSDQ_ADJ_C),
                    stock_name_map=stock_name_map,
                )
                return (
                    BenchmarkRepository.default(),
                    sector_repo,
                )
        sector_repo = SectorRepository.from_frame(
            _load_default_frame(DatasetId.QW_KSDQ_WICS_SEC_BIG),
            prices=_load_default_frame(DatasetId.QW_KSDQ_ADJ_C),
            sector_name_map=sector_name_map,
            stock_name_map=stock_name_map,
        )
        return (
            BenchmarkRepository.default(),
            sector_repo,
        )
    if universe_id == "etf":
        _sector_name_map, stock_name_map = _load_display_name_maps(ROOT.raw_path / "map.xlsx")
        sector_repo = SectorRepository.from_frame(
            _read_static_sector_frame(ROOT.raw_path / "map.xlsx", sector_value="ETF"),
            prices=_load_default_frame(DatasetId.QW_ETF_ADJ_C),
            stock_name_map=stock_name_map,
        )
        return (
            BenchmarkRepository.default(),
            sector_repo,
        )
    return BenchmarkRepository.default(), SectorRepository.default()


def _load_default_frame(dataset_id: DatasetId) -> pd.DataFrame:
    catalog = DataCatalog.default()
    store = ParquetStore(ROOT.parquet_path)
    parquet_path = ROOT.parquet_path / f"{dataset_id.value}.parquet"
    if not parquet_path.exists():
        raw_path = find_raw_path(ROOT.raw_path, dataset_id.value)
        if dataset_id is DatasetId.QW_BM and raw_path.suffix == ".xlsx":
            frame = _read_quantwise_benchmark_frame(raw_path)
            store.write(dataset_id.value, frame)
            return frame
        IngestJob(catalog=catalog, raw_dir=ROOT.raw_path, parquet_dir=ROOT.parquet_path).run(dataset_id)
    return store.read(dataset_id.value)


def _read_quantwise_benchmark_frame(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    leading = raw.iloc[:, 0].astype(str).str.strip().str.upper()

    code_rows = leading[leading.eq("CODE")]
    date_rows = leading[leading.eq("D A T E")]
    if code_rows.empty or date_rows.empty:
        raise KeyError(f"unable to locate QuantWise benchmark headers in {path.name}")

    code_row = int(code_rows.index[0])
    date_row = int(date_rows.index[0])
    codes = raw.iloc[code_row, 1:]
    valid_columns = [int(column) for column, value in codes.items() if pd.notna(value)]

    frame = raw.loc[date_row + 1 :, [0, *valid_columns]].copy()
    frame.columns = ["date", *[str(codes[column]).strip() for column in valid_columns]]
    frame = frame.dropna(subset=["date"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").set_index("date")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame.index.name = "date"
    return frame


def _read_historical_sector_frame(
    path: Path,
    *,
    date_column: str = "DATE",
    symbol_column: str = "TICKER",
    sector_column: str = "GICS_SECTOR_LV1_NAME",
) -> pd.DataFrame:
    frame = pd.read_excel(path)
    columns = {str(column).strip().upper(): column for column in frame.columns}
    resolved_sector_column = _resolve_sector_column(columns, sector_column)
    required = {
        date_column.strip().upper(): date_column,
        symbol_column.strip().upper(): symbol_column,
        resolved_sector_column.strip().upper(): resolved_sector_column,
    }
    missing = [original for normalized, original in required.items() if normalized not in columns]
    if missing:
        raise KeyError(f"missing sector history columns in {path.name}: {', '.join(missing)}")

    selected = frame.loc[
        :,
        [
            columns[date_column.strip().upper()],
            columns[symbol_column.strip().upper()],
            columns[resolved_sector_column.strip().upper()],
        ],
    ].copy()
    selected.columns = ["date", "symbol", "sector"]
    selected = selected.dropna(subset=["date", "symbol", "sector"])
    selected["date"] = pd.to_datetime(selected["date"]).dt.normalize()
    selected["symbol"] = selected["symbol"].map(lambda value: SectorRepository._normalize_symbol_key(str(value)))
    selected["sector"] = selected["sector"].astype(str).str.strip()
    selected = selected.drop_duplicates(subset=["date", "symbol"], keep="last")

    pivoted = (
        selected.pivot(index="date", columns="symbol", values="sector")
        .sort_index()
        .sort_index(axis=1)
    )
    pivoted.index.name = "date"
    pivoted.columns.name = None
    return pivoted


def _read_static_sector_frame(
    path: Path,
    *,
    ticker_column: str = "TICKER",
    sector_column: str = "GICS_SECTOR_NAME",
    sector_value: str | None = None,
) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    requested_ticker = ticker_column.strip().upper()
    requested_sector = sector_column.strip().upper()
    normalized_filter = None if sector_value is None else sector_value.strip().upper()

    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name)
        columns = {str(column).strip().upper(): column for column in frame.columns}
        if requested_ticker not in columns or requested_sector not in columns:
            continue

        selected = frame.loc[:, [columns[requested_ticker], columns[requested_sector]]].copy()
        selected.columns = ["symbol", "sector"]
        selected = selected.dropna(subset=["symbol", "sector"])
        selected["symbol"] = selected["symbol"].map(lambda value: SectorRepository._normalize_symbol_key(str(value)))
        selected["sector"] = selected["sector"].astype(str).str.strip()
        if normalized_filter is not None:
            selected = selected.loc[selected["sector"].str.upper().eq(normalized_filter)]
        selected = selected.drop_duplicates(subset=["symbol"], keep="last")
        if selected.empty:
            continue

        frame = pd.DataFrame(
            [dict(zip(selected["symbol"], selected["sector"], strict=True))],
            index=pd.to_datetime(["1900-01-01"]),
        ).sort_index(axis=1)
        frame.index.name = "date"
        return frame

    raise KeyError(f"missing static sector mapping in {path.name}: {ticker_column}, {sector_column}")


def _resolve_kosdaq_gics_level_path(raw_dir: Path, *, level: str = "lv1") -> Path:
    normalized = str(level).strip().lower()
    candidate_names = [
        f"snp_ksdq_gics_sector_big_{normalized}.xlsx",
        f"snp_ksdq_gics_sector_big_{normalized}.xls",
    ]
    candidate_dirs = [
        raw_dir / "ksdq",
        raw_dir,
    ]
    for directory in candidate_dirs:
        for name in candidate_names:
            path = directory / name
            if path.exists():
                return path

    legacy = raw_dir / "snp_ksdq_gics_sector_big.xlsx"
    if legacy.exists():
        return legacy

    return raw_dir / "ksdq" / candidate_names[0]


def _resolve_sector_column(columns: dict[str, object], requested: str) -> str:
    normalized = str(requested).strip().upper()
    if normalized in columns:
        return str(requested).strip()
    fallbacks = {
        "GICS_SECTOR_LV1_NAME": ["GICS_SECTOR_NAME"],
        "GICS_SECTOR_LV2_NAME": [],
        "GICS_SECTOR_NAME": ["GICS_SECTOR_LV1_NAME"],
    }
    for candidate in fallbacks.get(normalized, []):
        if candidate in columns:
            return candidate
    return str(requested).strip()


def _load_display_name_maps(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    if not path.exists():
        return {}, {}

    workbook = pd.ExcelFile(path)
    sector_name_map: dict[str, str] = {}
    stock_name_map: dict[str, str] = {}

    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name)
        columns = {str(column).strip().lower(): column for column in frame.columns}
        if {"code", "name"} <= set(columns):
            pairs = frame.loc[:, [columns["code"], columns["name"]]].dropna()
            for _, row in pairs.iterrows():
                sector_name_map[str(row.iloc[0]).strip()] = str(row.iloc[1]).strip()
            continue
        if {"ticker", "name"} <= set(columns):
            pairs = frame.loc[:, [columns["ticker"], columns["name"]]].dropna()
            for _, row in pairs.iterrows():
                ticker = SectorRepository._normalize_symbol_key(str(row.iloc[0]).strip())
                stock_name_map[ticker] = str(row.iloc[1]).strip()

    return sector_name_map, stock_name_map
