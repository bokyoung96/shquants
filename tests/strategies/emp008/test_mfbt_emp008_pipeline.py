from pathlib import Path

import pandas as pd
import pytest

from backtesting.catalog import DatasetId
from backtesting.data import MarketData
from backtesting.strategies.emp008.mfbt_emp008 import MfbtEmp008Result, run_mfbt_emp008_smoke
from backtesting.strategies.emp008 import mfbt_emp008_data
from backtesting.strategies.emp008.mfbt_emp008_data import (
    MfbtEmp008Config,
    load_mfbt_emp008_market,
    padded_history_start,
    required_datasets,
)


def test_smoke_uses_benchmark_only_loader(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_load_bm_weights(*, parquet_dir, start, end, config):
        captured["parquet_dir"] = parquet_dir
        captured["start"] = start
        captured["end"] = end
        captured["config"] = config
        return pd.DataFrame({"A": [0.6], "B": [0.4]}, index=pd.to_datetime([start]))

    monkeypatch.setattr("backtesting.strategies.emp008.mfbt_emp008.load_mfbt_emp008_bm_weights", fake_load_bm_weights)

    result = run_mfbt_emp008_smoke(parquet_dir=tmp_path, start="2025-01-31", end="2025-03-31")

    assert captured["parquet_dir"] == tmp_path
    assert captured["start"] == "2025-01-31"
    assert captured["end"] == "2025-03-31"
    assert result.target_weights.index.tolist() == list(pd.to_datetime(["2025-01-31"]))


def test_smoke_rejects_empty_or_zero_benchmark_weights(monkeypatch, tmp_path) -> None:
    def fake_load_bm_weights(*, parquet_dir, start, end, config):
        return pd.DataFrame({"A": [0.0], "B": [0.0]}, index=pd.to_datetime([start]))

    monkeypatch.setattr("backtesting.strategies.emp008.mfbt_emp008.load_mfbt_emp008_bm_weights", fake_load_bm_weights)

    with pytest.raises(ValueError, match="usable benchmark weight rows"):
        run_mfbt_emp008_smoke(parquet_dir=tmp_path, start="2025-01-31", end="2025-03-31")


def test_mfbt_emp008_config_defaults_to_wi26_big_sector_and_bm_weights() -> None:
    config = MfbtEmp008Config()

    assert config.sector_dataset is DatasetId.QW_WI_SEC_26_BIG
    assert config.bm_weights_dataset is DatasetId.QW_BM_WEIGHTS
    assert config.universe_dataset is DatasetId.QW_K200_YN
    assert config.float_market_cap_dataset is DatasetId.QW_MKTCAP_FLT
    assert pd.Timedelta(days=config.retail_flow_lookback_days) == pd.Timedelta(days=252)


def test_required_datasets_include_all_mfbt_emp008_inputs() -> None:
    datasets = set(required_datasets(MfbtEmp008Config()))

    assert DatasetId.QW_ADJ_C in datasets
    assert DatasetId.QW_C in datasets
    assert DatasetId.QW_BM_WEIGHTS in datasets
    assert DatasetId.QW_OP_FWD_12M in datasets
    assert DatasetId.QW_DPS_TTM in datasets
    assert DatasetId.QW_RETAIL in datasets
    assert DatasetId.QW_WI_SEC_26_BIG in datasets
    assert DatasetId.QW_MKTCAP in datasets
    assert DatasetId.QW_MKTCAP_FLT in datasets
    assert DatasetId.QW_FCF in datasets
    assert DatasetId.QW_INT_BEARING_LIAB_NFQ0 in datasets
    assert DatasetId.QW_QUICK_ASSETS_NFQ0 in datasets
    assert DatasetId.QW_K200_YN in datasets


def test_result_exports_date_by_ticker_and_ticker_by_date() -> None:
    target = pd.DataFrame(
        {"A": [0.6, 0.5], "B": [0.4, 0.5]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    result = MfbtEmp008Result(target_weights=target, active_weights=target * 0.0, diagnostics=pd.DataFrame())

    assert result.target_weights.index.tolist() == list(pd.to_datetime(["2024-01-31", "2024-02-29"]))
    assert result.weights_for_export().index.tolist() == ["A", "B"]
    assert result.weights_for_export().columns.tolist() == list(pd.to_datetime(["2024-01-31", "2024-02-29"]))


def test_result_write_outputs_preserves_weight_orientations(tmp_path) -> None:
    target = pd.DataFrame(
        {"A": [0.6, 0.5], "B": [0.4, 0.5]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    result = MfbtEmp008Result(target_weights=target, active_weights=target * 0.0, diagnostics=pd.DataFrame())

    result.write_outputs(tmp_path)

    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "target_weights.parquet"), target)
    export = pd.read_excel(tmp_path / "weights_export.xlsx", sheet_name="weights_ticker_by_date", index_col=0)
    export.columns = pd.to_datetime(export.columns)
    pd.testing.assert_frame_equal(export, target.T)


def test_loader_requests_padded_history_for_rolling_factors(monkeypatch, tmp_path) -> None:
    captured = {}

    class CapturingLoader:
        def __init__(self, catalog, store) -> None:
            self.catalog = catalog
            self.store = store

        def load(self, request):
            captured["request"] = request
            return MarketData(frames={}, universe=None, benchmark=None)

    monkeypatch.setattr(mfbt_emp008_data, "DataLoader", CapturingLoader)
    config = MfbtEmp008Config()

    load_mfbt_emp008_market(
        parquet_dir=tmp_path,
        start="2025-01-31",
        end="2025-03-31",
        config=config,
    )

    assert captured["request"].start == padded_history_start("2025-01-31", config)
    assert captured["request"].end == "2025-03-31"
    assert pd.Timestamp(captured["request"].start) < pd.Timestamp("2025-01-31")


def test_padded_history_start_covers_warm_risk_window() -> None:
    config = MfbtEmp008Config()
    start = "2025-01-31"
    padded = padded_history_start(start, config)
    business_days = pd.bdate_range(padded, start)
    warmed_month_ends = business_days[config.retail_flow_lookback_days :].to_period("M").nunique()

    assert warmed_month_ends >= config.risk_window


def test_real_data_smoke_run_produces_weight_rows() -> None:
    if not Path("parquet/qw_bm_weights.parquet").exists():
        pytest.skip("local parquet data not available")

    result = run_mfbt_emp008_smoke(
        parquet_dir=Path("parquet"),
        start="2025-01-31",
        end="2025-03-31",
    )

    assert not result.target_weights.empty
    assert result.target_weights.index.min() >= pd.Timestamp("2025-01-31")
    assert result.target_weights.index.max() <= pd.Timestamp("2025-03-31")
    assert result.target_weights.sum(axis=1).round(8).eq(1.0).all()
    assert not result.diagnostics.empty
