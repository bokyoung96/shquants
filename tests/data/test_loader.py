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


def test_loader_uses_semantic_key_for_trading_value_data(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_v_value",
        pd.DataFrame(
            {"005930": [1_000_000.0, 2_000_000.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
    )

    loader = DataLoader(DataCatalog.default(), store)
    data = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_V_VALUE],
            start="2024-01-02",
            end="2024-01-03",
        )
    )

    assert "trading_value" in data.frames
    assert "qw_v_value" not in data.frames


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


def test_loader_applies_lag_after_expanding_month_only_data(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_adj_c",
        pd.DataFrame(
            {"005930": [1.0, 2.0]},
            index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
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
                lag=31,
                dtype="float64",
            )
        }
    )
    loader = DataLoader(catalog, store)

    data = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_ADJ_C],
            start="2024-02-01",
            end="2024-03-31",
        )
    )

    close = data.frames["close"]
    assert close.loc["2024-02-01", "005930"] == 1.0
    assert close.loc["2024-03-01", "005930"] == 1.0
    assert close.loc["2024-03-31", "005930"] == 2.0


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


def test_loader_uses_semantic_key_for_op_fwd_12m_data(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_op_fwd_12m",
        pd.DataFrame(
            {"005930": [10.0]},
            index=pd.to_datetime(["2024-01-31"]),
        ),
    )

    loader = DataLoader(DataCatalog.default(), store)
    data = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_OP_FWD_12M],
            start="2024-01-01",
            end="2024-01-31",
        )
    )

    assert "op_fwd_12m" in data.frames
    assert "qw_op_fwd_12m" not in data.frames


def test_loader_uses_semantic_key_for_dps_ttm_data(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_dps_ttm",
        pd.DataFrame(
            {"005930": [1444.0, 1444.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
    )

    loader = DataLoader(DataCatalog.default(), store)
    data = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_DPS_TTM],
            start="2024-01-02",
            end="2024-01-03",
        )
    )

    assert "dps_ttm" in data.frames
    assert "qw_dps_ttm" not in data.frames


def test_loader_uses_semantic_key_for_dividend_cash_data(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_dividend_cash",
        pd.DataFrame(
            {"005930": [361.0, 0.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
    )

    loader = DataLoader(DataCatalog.default(), store)
    data = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_DIVIDEND_CASH],
            start="2024-01-02",
            end="2024-01-03",
        )
    )

    assert "dividend_cash" in data.frames
    assert "qw_dividend_cash" not in data.frames


def test_loader_uses_semantic_key_for_dividend_cash_ttm_data(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_dividend_cash_ttm",
        pd.DataFrame(
            {"005930": [1444.0, 1444.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
    )

    loader = DataLoader(DataCatalog.default(), store)
    data = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_DIVIDEND_CASH_TTM],
            start="2024-01-02",
            end="2024-01-03",
        )
    )

    assert "dividend_cash_ttm" in data.frames
    assert "qw_dividend_cash_ttm" not in data.frames


def test_loader_uses_semantic_key_for_dividend_yld_fy0_data(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_dividend_yld_fy0",
        pd.DataFrame(
            {"005930": [0.025, 0.026]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
    )

    loader = DataLoader(DataCatalog.default(), store)
    data = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_DIVIDEND_YLD_FY0],
            start="2024-01-02",
            end="2024-01-03",
        )
    )

    assert "dividend_yld_fy0" in data.frames
    assert "qw_dividend_yld_fy0" not in data.frames


def test_loader_uses_semantic_key_for_wi26_sector_data(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_wi_sec_26",
        pd.DataFrame(
            {"A005930": ["WI62010"]},
            index=pd.to_datetime(["2024-01-31"]),
        ),
    )

    loader = DataLoader(DataCatalog.default(), store)
    data = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_WI_SEC_26],
            start="2024-01-01",
            end="2024-01-31",
        )
    )

    assert "sector_big" in data.frames
    assert "qw_wi_sec_26" not in data.frames


def test_loader_uses_semantic_key_for_wi26_big_sector_data(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_wi_sec_26_big",
        pd.DataFrame(
            {"A005930": ["WI26B10"]},
            index=pd.to_datetime(["2024-01-31"]),
        ),
    )

    loader = DataLoader(DataCatalog.default(), store)
    data = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_WI_SEC_26_BIG],
            start="2024-01-01",
            end="2024-01-31",
        )
    )

    assert "sector_big" in data.frames
    assert "qw_wi_sec_26_big" not in data.frames


@pytest.mark.parametrize(
    ("dataset_id", "stem", "frame_key"),
    [
        (DatasetId.QW_FCF, "qw_fcf", "free_cash_flow"),
        (DatasetId.QW_INT_BEARING_LIAB_NFQ0, "qw_int_bearing_liab_nfq0", "interest_bearing_liability"),
        (DatasetId.QW_QUICK_ASSETS_NFQ0, "qw_quick_assets_nfq0", "quick_asset"),
        (DatasetId.QW_TANGIBLE_ASSETS_NFQ0, "qw_tangible_assets_nfq0", "tangible_asset"),
    ],
)
def test_loader_uses_semantic_key_for_value_fundamental_data(
    tmp_path: Path,
    dataset_id: DatasetId,
    stem: str,
    frame_key: str,
) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        stem,
        pd.DataFrame(
            {"005930": [10.0]},
            index=pd.to_datetime(["2024-01-31"]),
        ),
    )

    loader = DataLoader(DataCatalog.default(), store)
    data = loader.load(
        LoadRequest(
            datasets=[dataset_id],
            start="2024-01-01",
            end="2024-01-31",
        )
    )

    assert frame_key in data.frames
    assert stem not in data.frames


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


def test_loader_uses_semantic_key_for_etf_close_data(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_etf_adj_c",
        pd.DataFrame(
            {"A069500": [35000.0, 35100.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
    )

    loader = DataLoader(DataCatalog.default(), store)
    data = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_ETF_ADJ_C],
            start="2024-01-02",
            end="2024-01-03",
        )
    )

    assert "close" in data.frames
    assert "qw_etf_adj_c" not in data.frames
    assert list(data.frames["close"].columns) == ["A069500"]


def test_loader_uses_semantic_key_for_etf_adjusted_volume_data(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    store = ParquetStore(parquet_dir)
    store.write(
        "qw_etf_adj_v",
        pd.DataFrame(
            {"A069500": [1000.0, 1100.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
    )

    loader = DataLoader(DataCatalog.default(), store)
    data = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_ETF_ADJ_V],
            start="2024-01-02",
            end="2024-01-03",
        )
    )

    assert "volume" in data.frames
    assert "qw_etf_adj_v" not in data.frames
    assert list(data.frames["volume"].columns) == ["A069500"]


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
