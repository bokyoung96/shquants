from pathlib import Path
import json
import pandas as pd
from .catalog import DataCatalog, DatasetId
from .loader import ParquetStore


def _read_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx": return pd.read_excel(path)
    for encoding in ("utf-8", "utf-8-sig", "cp949"):
        try: return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError: continue
    return pd.read_csv(path)


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    date_col = next((c for c in frame.columns if str(c).lower() in {"date", "work_dt", "trd_dt"}), frame.columns[0])
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).set_index(date_col).sort_index()
    frame.index.name = "date"
    return frame[~frame.index.duplicated(keep="last")]


def convert_dataset(*, raw_dir: Path, parquet_dir: Path, dataset_id: DatasetId) -> Path:
    catalog = DataCatalog.default(); stem = catalog.get(dataset_id).stem
    if dataset_id is DatasetId.QW_BM_WEIGHTS:
        krx_path = Path(raw_dir) / "krx_ks200_weight.xlsx"
        krx = pd.read_excel(krx_path, sheet_name="Sheet2")
        close = _normalize(_read_raw(next(Path(raw_dir).rglob("qw_c.csv"))))
        krx["Work_Dt"] = pd.to_datetime(krx["Work_Dt"]).dt.normalize()
        krx["float_index_shares"] = krx["Index_Share"].astype(float) * krx["Free_Float_Factor"].astype(float)
        shares = krx.pivot(index="Work_Dt", columns="Constituent_Code", values="float_index_shares").sort_index()
        weights = (shares * close.reindex(index=shares.index, columns=shares.columns)).pipe(lambda x: x.div(x.sum(axis=1), axis=0)).fillna(0.0)
        weights.index.name = "date"; weights.columns.name = None
        return ParquetStore(parquet_dir).write(stem, weights)
    candidates = list(Path(raw_dir).rglob(f"{stem}.csv")) + list(Path(raw_dir).rglob(f"{stem}.xlsx"))
    if not candidates: raise FileNotFoundError(f"raw source not found for {stem} under {raw_dir}")
    return ParquetStore(parquet_dir).write(stem, _normalize(_read_raw(candidates[0])))


def convert_required(*, raw_dir: Path, parquet_dir: Path, config=None) -> dict[str, str]:
    if config is None:
        from emp008.data import Emp008Config, required_datasets
        config = Emp008Config()
    outputs = {}
    for dataset_id in required_datasets(config): outputs[dataset_id.value] = str(convert_dataset(raw_dir=raw_dir, parquet_dir=parquet_dir, dataset_id=dataset_id))
    manifest = Path(parquet_dir) / "manifest.json"; manifest.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    return outputs
