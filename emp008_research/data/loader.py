from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from .catalog import DataCatalog, DatasetId


@dataclass(slots=True)
class MarketData:
    frames: dict[str, pd.DataFrame]
    universe: pd.DataFrame | None = None
    benchmark: pd.Series | None = None


class ParquetStore:
    def __init__(self, root: Path): self.root = Path(root)
    def read(self, stem: str) -> pd.DataFrame:
        path = self.root / f"{stem}.parquet"
        if not path.exists(): raise FileNotFoundError(f"missing parquet dataset: {path}")
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index)
        return frame.sort_index()
    def write(self, stem: str, frame: pd.DataFrame) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{stem}.parquet"; frame.to_parquet(path); return path


FRAME_KEYS = {
    DatasetId.QW_ADJ_C: "close", DatasetId.QW_ADJ_O: "open", DatasetId.QW_ADJ_H: "high", DatasetId.QW_ADJ_L: "low",
    DatasetId.QW_BM_WEIGHTS: "bm_weights", DatasetId.QW_C: "close_raw", DatasetId.QW_DIVIDEND_YLD_FY0: "dividend_yld_fy0",
    DatasetId.QW_DPS_TTM: "dps_ttm", DatasetId.QW_FCF: "free_cash_flow", DatasetId.QW_INT_BEARING_LIAB_NFQ0: "interest_bearing_liability",
    DatasetId.QW_K200_YN: "k200_yn", DatasetId.QW_MKTCAP: "market_cap", DatasetId.QW_MKTCAP_FLT: "float_market_cap",
    DatasetId.QW_OP_FWD_12M: "op_fwd_12m", DatasetId.QW_QUICK_ASSETS_NFQ0: "quick_asset", DatasetId.QW_RETAIL: "retail_flow",
    DatasetId.QW_WI_SEC_26_BIG: "sector_big", DatasetId.QW_WICS_SEC_BIG: "sector_big",
}


def _load_frame(store: ParquetStore, spec, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = store.read(spec.stem)
    if spec.validity == "daily": return frame.loc[start:end]
    month_start = start.to_period("M").start_time
    frame = frame.loc[month_start:end]
    calendar = pd.date_range(start, end, freq="D")
    monthly = frame.copy()
    monthly.index = pd.DatetimeIndex(monthly.index).to_period("M")
    monthly = monthly.loc[~monthly.index.duplicated(keep="last")]
    expanded = monthly.reindex(calendar.to_period("M"))
    expanded.index = calendar
    return expanded


def load_market_data(parquet_dir: Path, *, start: str, end: str, config) -> MarketData:
    from emp008.data import padded_history_start, padded_snapshot_end, required_datasets
    requested_start, requested_end = pd.Timestamp(start), pd.Timestamp(end)
    load_start, load_end = pd.Timestamp(padded_history_start(start, config)), pd.Timestamp(padded_snapshot_end(end, config))
    store, catalog = ParquetStore(parquet_dir), DataCatalog.default()
    frames: dict[str, pd.DataFrame] = {}
    for dataset_id in required_datasets(config):
        spec = catalog.get(dataset_id)
        key = FRAME_KEYS[dataset_id]
        if key in frames and dataset_id != config.sector_neutral_dataset: continue
        frames[key] = _load_frame(store, spec, load_start, load_end)
    neutral = config.sector_neutral_dataset or config.sector_dataset
    if neutral != config.sector_dataset:
        frames["sector_neutral_big"] = _load_frame(store, catalog.get(neutral), load_start, load_end)
    else:
        frames["sector_neutral_big"] = frames["sector_big"]
    definition = config.monthly_snapshot_forward_days
    if definition > 0:
        frames = {key: frame if key == "dividend_yld_fy0" else frame.loc[:requested_end] for key, frame in frames.items()}
    return MarketData(frames=frames, universe=None, benchmark=None)
