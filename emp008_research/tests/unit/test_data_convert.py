import pandas as pd
from data.convert import convert_dataset
from data.catalog import DatasetId


def test_csv_to_parquet_round_trip(tmp_path):
    raw = tmp_path / "raw"; out = tmp_path / "parquet"; raw.mkdir()
    pd.DataFrame({"date": ["2020-01-02", "2020-01-01"], "AAA": [2, 1]}).to_csv(raw / "qw_adj_c.csv", index=False)
    path = convert_dataset(raw_dir=raw, parquet_dir=out, dataset_id=DatasetId.QW_ADJ_C)
    frame = pd.read_parquet(path)
    assert frame.index.tolist() == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    assert frame["AAA"].tolist() == [1, 2]
