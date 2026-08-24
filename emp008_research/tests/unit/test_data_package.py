from pathlib import Path
import pandas as pd

from data.catalog import DataCatalog, DatasetId
from data.convert import convert_dataset


def test_catalog_exposes_required_runtime_specs():
    catalog = DataCatalog.default()
    assert catalog.get(DatasetId.QW_ADJ_C).kind == "price"
    assert catalog.get(DatasetId.QW_BM_WEIGHTS).kind == "benchmark_weights"
    assert catalog.get(DatasetId.QW_WICS_SEC_BIG).validity == "month_only"


def test_csv_to_parquet_conversion_is_sorted(tmp_path: Path):
    raw, out = tmp_path / "raw", tmp_path / "parquet"
    raw.mkdir()
    pd.DataFrame({"date": ["2024-01-02", "2024-01-01"], "A": [2.0, 1.0]}).to_csv(raw / "qw_adj_c.csv", index=False)
    result = convert_dataset(raw_dir=raw, parquet_dir=out, dataset_id=DatasetId.QW_ADJ_C)
    stored = pd.read_parquet(result)
    assert stored.index.tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
