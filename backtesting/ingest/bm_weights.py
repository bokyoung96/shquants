from pathlib import Path

import pandas as pd

from .io import find_raw_path, read_raw_frame
from .normalize import normalize_frame


KOSPI200_WEIGHT_FILE = "krx_ks200_weight.xlsx"
KOSPI200_WEIGHT_SHEET = "Sheet2"
REQUIRED_KRX_COLUMNS = {
    "Work_Dt",
    "Constituent_Code",
    "Index_Share",
    "Free_Float_Factor",
}


def read_kospi200_bm_weights(raw_dir: Path) -> pd.DataFrame:
    krx = read_krx_ks200_weight_sheet(raw_dir / KOSPI200_WEIGHT_FILE)
    close = normalize_frame(read_raw_frame(find_raw_path(raw_dir, "qw_c")))
    return build_kospi200_bm_weights(krx, close)


def read_krx_ks200_weight_sheet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing KRX KOSPI200 weight source: {path}")
    frame = pd.read_excel(path, sheet_name=KOSPI200_WEIGHT_SHEET)
    missing = REQUIRED_KRX_COLUMNS.difference(frame.columns)
    if missing:
        raise KeyError(f"missing KRX KOSPI200 weight columns: {sorted(missing)}")
    return frame


def build_kospi200_bm_weights(krx_weights: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    krx = krx_weights.loc[:, sorted(REQUIRED_KRX_COLUMNS)].copy()
    krx["Work_Dt"] = pd.to_datetime(krx["Work_Dt"]).dt.normalize()
    krx["float_index_shares"] = krx["Index_Share"].astype(float) * krx["Free_Float_Factor"].astype(float)

    if krx.duplicated(["Work_Dt", "Constituent_Code"]).any():
        raise ValueError("duplicate KOSPI200 weight rows for date/ticker")

    float_shares = (
        krx.pivot(index="Work_Dt", columns="Constituent_Code", values="float_index_shares")
        .sort_index()
        .astype(float)
    )
    float_shares.index.name = "date"

    close = close.loc[:, ~close.columns.duplicated()]
    aligned_close = close.reindex(index=float_shares.index, columns=float_shares.columns)
    float_market_value = float_shares * aligned_close
    row_sum = float_market_value.sum(axis=1, min_count=1)
    weights = float_market_value.div(row_sum.where(row_sum.gt(0)), axis=0)
    weights = weights.dropna(how="all").fillna(0.0).astype(float)
    weights.index.name = "date"
    weights.columns.name = None
    return weights
