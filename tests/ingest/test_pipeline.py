from pathlib import Path
import json

import pandas as pd
import pytest

from backtesting.catalog import DataCatalog, DatasetId
from backtesting.ingest.pipeline import IngestJob


def test_ingest_writes_parquet_and_report(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    parquet_dir = tmp_path / "parquet"
    raw_dir.mkdir()
    parquet_dir.mkdir()

    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "005930": [100.0, 101.0],
            "000660": [50.0, 49.5],
        }
    )
    frame.to_csv(raw_dir / "qw_adj_c.csv", index=False)

    job = IngestJob(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
    )

    result = job.run(DatasetId.QW_ADJ_C)

    assert (parquet_dir / "qw_adj_c.parquet").exists()
    assert (parquet_dir / "qw_adj_c.json").exists()
    assert result.rows == 2
    assert result.columns == 2
    assert result.missing == 0
    assert result.date_start.isoformat() == "2024-01-02"
    assert result.date_end.isoformat() == "2024-01-03"
    assert result.shape == [2, 2]
    assert result.dtypes == {"005930": "float64", "000660": "float64"}

    stored = pd.read_parquet(parquet_dir / "qw_adj_c.parquet", engine="pyarrow")
    assert stored.index.tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert stored.columns.tolist() == ["005930", "000660"]

    report = result.to_dict()
    assert report == {
        "stem": "qw_adj_c",
        "rows": 2,
        "columns": 2,
        "missing": 0,
        "date_start": "2024-01-02",
        "date_end": "2024-01-03",
        "shape": [2, 2],
        "dtypes": {"005930": "float64", "000660": "float64"},
    }

    report_path = parquet_dir / "qw_adj_c.json"
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_ingest_reads_xlsx_sources(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    parquet_dir = tmp_path / "parquet"
    raw_dir.mkdir()
    parquet_dir.mkdir()

    frame = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02"],
            "005930": [101.0, 100.0],
        }
    )
    frame.to_excel(raw_dir / "qw_adj_c.xlsx", index=False)

    job = IngestJob(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
    )

    result = job.run(DatasetId.QW_ADJ_C)

    assert result.rows == 2
    stored = pd.read_parquet(parquet_dir / "qw_adj_c.parquet", engine="pyarrow")
    assert stored.index.tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))


def test_ingest_reads_cp949_csv_sources(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    parquet_dir = tmp_path / "parquet"
    raw_dir.mkdir()
    parquet_dir.mkdir()

    frame = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02"],
            "A005930": ["에너지", "정보기술"],
        }
    )
    frame.to_csv(raw_dir / "qw_wics_sec_big.csv", index=False, encoding="cp949")

    job = IngestJob(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
    )

    result = job.run(DatasetId.QW_WICS_SEC_BIG)

    assert result.rows == 2
    stored = pd.read_parquet(parquet_dir / "qw_wics_sec_big.parquet", engine="pyarrow")
    assert stored.index.tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert stored.iloc[0, 0] == "정보기술"
    assert stored.iloc[1, 0] == "에너지"


def test_ingest_reads_wi26_sector_code_sources(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    parquet_dir = tmp_path / "parquet"
    raw_dir.mkdir()
    parquet_dir.mkdir()

    frame = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02"],
            "A005930": ["WI62010", "WI61040"],
        }
    )
    frame.to_csv(raw_dir / "qw_wi_sec_26.csv", index=False, encoding="cp949")

    job = IngestJob(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
    )

    result = job.run(DatasetId.QW_WI_SEC_26)

    assert result.rows == 2
    stored = pd.read_parquet(parquet_dir / "qw_wi_sec_26.parquet", engine="pyarrow")
    assert stored.index.tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert stored.iloc[0, 0] == "WI61040"
    assert stored.iloc[1, 0] == "WI62010"


def test_ingest_reads_wi26_big_sector_code_sources(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    parquet_dir = tmp_path / "parquet"
    raw_dir.mkdir()
    parquet_dir.mkdir()

    frame = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02"],
            "A005930": ["WI26B20", "WI26B10"],
        }
    )
    frame.to_excel(raw_dir / "qw_wi_sec_26_big.xlsx", index=False)

    job = IngestJob(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
    )

    result = job.run(DatasetId.QW_WI_SEC_26_BIG)

    assert result.rows == 2
    stored = pd.read_parquet(parquet_dir / "qw_wi_sec_26_big.parquet", engine="pyarrow")
    assert stored.index.tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert stored.iloc[0, 0] == "WI26B10"
    assert stored.iloc[1, 0] == "WI26B20"


def test_ingest_reads_dividend_cash_ttm_sources(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    parquet_dir = tmp_path / "parquet"
    raw_dir.mkdir()
    parquet_dir.mkdir()

    frame = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02"],
            "A005930": [1444.0, 1200.0],
        }
    )
    frame.to_csv(raw_dir / "qw_dividend_cash_ttm.csv", index=False)

    job = IngestJob(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
    )

    result = job.run(DatasetId.QW_DIVIDEND_CASH_TTM)

    assert result.rows == 2
    stored = pd.read_parquet(parquet_dir / "qw_dividend_cash_ttm.parquet", engine="pyarrow")
    assert stored.index.tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert stored.iloc[0, 0] == 1200.0
    assert stored.iloc[1, 0] == 1444.0


def test_ingest_rejects_duplicate_dates(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    parquet_dir = tmp_path / "parquet"
    raw_dir.mkdir()
    parquet_dir.mkdir()

    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "005930": [100.0, 101.0],
        }
    )
    frame.to_csv(raw_dir / "qw_adj_c.csv", index=False)

    job = IngestJob(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
    )

    with pytest.raises(ValueError, match="duplicate date"):
        job.run(DatasetId.QW_ADJ_C)


def test_ingest_rejects_duplicate_days_with_different_times(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    parquet_dir = tmp_path / "parquet"
    raw_dir.mkdir()
    parquet_dir.mkdir()

    frame = pd.DataFrame(
        {
            "date": ["2024-01-02 09:00:00", "2024-01-02 15:30:00"],
            "005930": [100.0, 101.0],
        }
    )
    frame.to_csv(raw_dir / "qw_adj_c.csv", index=False)

    job = IngestJob(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
    )

    with pytest.raises(ValueError, match="duplicate date"):
        job.run(DatasetId.QW_ADJ_C)


def test_ingest_creates_missing_parquet_directory(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    parquet_dir = tmp_path / "parquet"
    raw_dir.mkdir()

    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "005930": [100.0],
        }
    )
    frame.to_csv(raw_dir / "qw_adj_c.csv", index=False)

    job = IngestJob(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
    )

    result = job.run(DatasetId.QW_ADJ_C)

    assert result.rows == 1
    assert parquet_dir.is_dir()
    assert (parquet_dir / "qw_adj_c.parquet").exists()


def test_ingest_prefers_csv_when_csv_and_xlsx_exist(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    parquet_dir = tmp_path / "parquet"
    raw_dir.mkdir()
    parquet_dir.mkdir()

    pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "005930": [100.0],
        }
    ).to_csv(raw_dir / "qw_adj_c.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "005930": [999.0],
        }
    ).to_excel(raw_dir / "qw_adj_c.xlsx", index=False)

    job = IngestJob(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
    )

    job.run(DatasetId.QW_ADJ_C)

    stored = pd.read_parquet(parquet_dir / "qw_adj_c.parquet", engine="pyarrow")
    assert stored.iloc[0, 0] == 100.0


def test_ingest_accepts_unnamed_date_column(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    parquet_dir = tmp_path / "parquet"
    raw_dir.mkdir()
    parquet_dir.mkdir()

    frame = pd.DataFrame(
        {
            "Unnamed: 0": ["2024-01-02", "2024-01-03"],
            "005930": [100.0, 101.0],
        }
    )
    frame.to_csv(raw_dir / "qw_adj_c.csv", index=False)

    job = IngestJob(
        catalog=DataCatalog.default(),
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
    )

    result = job.run(DatasetId.QW_ADJ_C)

    assert result.rows == 2
    stored = pd.read_parquet(parquet_dir / "qw_adj_c.parquet", engine="pyarrow")
    assert stored.index.tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))


def test_ingest_finds_nested_raw_dataset(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    parquet_dir = tmp_path / "parquet"
    nested = raw_dir / "ksdq"
    nested.mkdir(parents=True)
    parquet_dir.mkdir()

    pd.DataFrame(
        {
            "Unnamed: 0": ["2024-01-02", "2024-01-03"],
            "A035720": [100.0, 101.0],
        }
    ).to_csv(nested / "qw_ksdq_adj_c.csv", index=False)

    job = IngestJob(DataCatalog.default(), raw_dir=raw_dir, parquet_dir=parquet_dir)

    result = job.run(DatasetId.QW_KSDQ_ADJ_C)

    assert result.rows == 2
    assert (parquet_dir / "qw_ksdq_adj_c.parquet").exists()


def test_ingest_rejects_ambiguous_nested_raw_dataset(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    parquet_dir = tmp_path / "parquet"
    nested_a = raw_dir / "a"
    nested_b = raw_dir / "b"
    nested_a.mkdir(parents=True)
    nested_b.mkdir(parents=True)
    parquet_dir.mkdir()

    frame = pd.DataFrame(
        {
            "Unnamed: 0": ["2024-01-02"],
            "A035720": [100.0],
        }
    )
    frame.to_csv(nested_a / "qw_ksdq_adj_c.csv", index=False)
    frame.to_csv(nested_b / "qw_ksdq_adj_c.csv", index=False)

    job = IngestJob(DataCatalog.default(), raw_dir=raw_dir, parquet_dir=parquet_dir)

    with pytest.raises(ValueError, match=r"ambiguous raw dataset: qw_ksdq_adj_c"):
        job.run(DatasetId.QW_KSDQ_ADJ_C)
