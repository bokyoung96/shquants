from pathlib import Path

import pandas as pd
import pytest

from backtesting.catalog import DataCatalog, DatasetGroup, DatasetId, DatasetSpec
from backtesting.data.loader import DataLoader, LoadRequest
from backtesting.data.store import ParquetStore


def test_loader_returns_market_data(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_adj_c",
        pd.DataFrame(
            {"005930": [100.0, 101.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
    )

    loader = DataLoader(DataCatalog.default(), store)
    data = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_ADJ_C],
            start="2024-01-02",
            end="2024-01-03",
        )
    )

    assert "close" in data.frames
    assert list(data.frames["close"].columns) == ["005930"]


def test_store_caches_reads_and_keeps_frames_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    path = parquet_dir / "qw_adj_c.parquet"
    pd.DataFrame(
        {"005930": [100.0, 101.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    ).to_parquet(path, engine="pyarrow")

    calls: list[Path] = []
    read_parquet = pd.read_parquet

    def spy(path: Path, *args, **kwargs):
        calls.append(Path(path))
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)

    store = ParquetStore(parquet_dir)
    first = store.read("qw_adj_c")
    first.iloc[0, 0] = 999.0
    second = store.read("qw_adj_c")

    assert [item.name for item in calls] == ["qw_adj_c.parquet"]
    assert second.loc["2024-01-02", "005930"] == 100.0

    store.write(
        "qw_adj_c",
        pd.DataFrame(
            {"005930": [200.0]},
            index=pd.to_datetime(["2024-01-04"]),
        ),
    )
    third = store.read("qw_adj_c")

    assert [item.name for item in calls] == ["qw_adj_c.parquet"]
    assert third.loc["2024-01-04", "005930"] == 200.0


def test_loader_uses_semantic_key_for_volume_data(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_v",
        pd.DataFrame(
            {"005930": [10.0, 20.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
    )

    loader = DataLoader(DataCatalog.default(), store)
    data = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_V],
            start="2024-01-02",
            end="2024-01-03",
        )
    )

    assert "volume" in data.frames
    assert "qw_v" not in data.frames


def test_loader_expands_month_only_data_without_crossing_missing_months(
    tmp_path: Path,
) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_adj_c",
        pd.DataFrame(
            {"005930": [1.0, 3.0]},
            index=pd.to_datetime(["2024-03-31", "2024-05-31"]),
        ),
    )

    catalog = DataCatalog(
        specs={
            DatasetId.QW_ADJ_C: DatasetSpec(
                id=DatasetId.QW_ADJ_C,
                stem="qw_adj_c",
                group=DatasetGroup.PRICE,
                freq="M",
                kind="price",
                fill="none",
                validity="month_only",
                lag=0,
                dtype="float64",
            )
        }
    )
    loader = DataLoader(catalog, store)

    data = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_ADJ_C],
            start="2024-03-01",
            end="2024-05-31",
        )
    )

    close = data.frames["close"]
    assert close.loc["2024-03-15", "005930"] == 1.0
    assert pd.isna(close.loc["2024-04-15", "005930"])
    assert close.loc["2024-05-15", "005930"] == 3.0


def test_loader_rejects_unsupported_price_mode(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_adj_c",
        pd.DataFrame(
            {"005930": [100.0]},
            index=pd.to_datetime(["2024-01-02"]),
        ),
    )

    loader = DataLoader(DataCatalog.default(), store)

    with pytest.raises(ValueError, match="unsupported price_mode: raw"):
        loader.load(
            LoadRequest(
                datasets=[DatasetId.QW_ADJ_C],
                start="2024-01-02",
                end="2024-01-02",
                price_mode="raw",
            )
        )


def test_loader_uses_semantic_key_for_op_fwd_data(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_op_nfy1",
        pd.DataFrame(
            {"005930": [10.0]},
            index=pd.to_datetime(["2024-01-31"]),
        ),
    )

    loader = DataLoader(DataCatalog.default(), store)
    data = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_OP_NFY1],
            start="2024-01-01",
            end="2024-01-31",
        )
    )

    assert "op_fwd" in data.frames
    assert "qw_op_nfy1" not in data.frames


def test_loader_uses_semantic_key_for_kosdaq_close_data(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_ksdq_adj_c",
        pd.DataFrame(
            {"A035720": [10.0, 11.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
    )

    loader = DataLoader(DataCatalog.default(), store)
    data = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_KSDQ_ADJ_C],
            start="2024-01-02",
            end="2024-01-03",
        )
    )

    assert "close" in data.frames
    assert "qw_ksdq_adj_c" not in data.frames
    assert list(data.frames["close"].columns) == ["A035720"]


def test_loader_rejects_duplicate_semantic_frame_keys(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_adj_c",
        pd.DataFrame(
            {"005930": [100.0]},
            index=pd.to_datetime(["2024-01-02"]),
        ),
    )
    store.write(
        "qw_ksdq_adj_c",
        pd.DataFrame(
            {"A035720": [10.0]},
            index=pd.to_datetime(["2024-01-02"]),
        ),
    )

    loader = DataLoader(DataCatalog.default(), store)

    with pytest.raises(ValueError, match="duplicate semantic frame key: close"):
        loader.load(
            LoadRequest(
                datasets=[DatasetId.QW_ADJ_C, DatasetId.QW_KSDQ_ADJ_C],
                start="2024-01-02",
                end="2024-01-02",
            )
        )
