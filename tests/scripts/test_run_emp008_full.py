import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import pandas as pd

import backtesting.strategies.emp008.optimize as emp008_optimize

from backtesting.catalog import DataCatalog, DatasetId
from backtesting.data import MarketData
from backtesting.strategies.emp008.run_backtest import (
    active_share_payload,
    active_share_summary,
    build_target_weight_spec,
    default_weights_csv,
    load_weight_dates,
    resolve_run_output_dirs,
    write_active_share,
)
from backtesting.strategies.emp008.run_weights import (
    _parser as weights_parser,
    build_emp008_config,
    latest_common_end,
    write_target_weights_csv,
)
from backtesting.strategies.emp008 import run_full
from backtesting.strategies.emp008.run_full import _parser as full_parser
from backtesting.strategies.emp008.comparison import (
    active_weight_abs_sum_frame,
    build_emp008_comparison,
    excess_summary_bps,
    monthly_compounded_returns,
    monthly_excess_heatmap_frame,
    performance_metrics,
)
from backtesting.strategies.emp008.attribution import FactorAttributionResult, factor_attribution_row, write_factor_attribution
from backtesting.strategies.emp008.strategy import Emp008Result
from backtesting.strategies.emp008.factor_builders import _sector_relative_retail_flow
from backtesting.strategies.emp008.factors import build_raw_factors
from backtesting.strategies.emp008.strategy import (
    apply_expected_alpha_policy,
    _has_sufficient_risk_history,
    _stock_excess_covariance_for_target_universe,
)
from backtesting.strategies.emp008.data import (
    Emp008Config,
    _trim_non_forward_snapshot_frames,
    load_emp008_market,
    padded_snapshot_end,
    required_datasets,
)
from backtesting.strategies.emp008.factor_registry import FactorSetId
from backtesting.strategies.emp008.optimize import optimize_active_weights_with_covariance
from backtesting.strategies.emp008.preprocess import preprocess_factor_frame
from backtesting.strategies.emp008.factor_pipeline import PreparedEmp008Factors, complete_benchmark_history
from backtesting.strategies.emp008.factor_registry import get_factor_set_definition


def _frame(
    dates: pd.DatetimeIndex,
    columns: list[str],
    rows: list[list[object]],
) -> pd.DataFrame:
    return pd.DataFrame(rows, index=dates, columns=columns)


def make_prepared_bundle() -> PreparedEmp008Factors:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    columns = ["A", "B", "C", "D"]
    close = _frame(
        dates,
        columns,
        [
            [10.0, 20.0, 30.0, 40.0],
            [11.0, 19.0, 33.0, 38.0],
            [12.0, 18.0, 36.0, 37.0],
        ],
    )
    market_cap = _frame(
        dates,
        columns,
        [
            [100.0, 300.0, 200.0, 400.0],
            [110.0, 290.0, 210.0, 390.0],
            [120.0, 280.0, 220.0, 380.0],
        ],
    )
    universe = _frame(
        dates,
        columns,
        [
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
        ],
    ).astype(bool)
    factor_names = [factor_id.value for factor_id in get_factor_set_definition("mfbt").factors]
    alpha_factors = {
        factor_name: _frame(
            dates,
            columns,
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5],
                [0.3, 0.4, 0.5, 0.6],
            ],
        )
        for factor_name in factor_names
    }
    sectors = _frame(
        dates,
        columns,
        [
            ["Tech", "Finance", "Health", "Utilities"],
            ["Tech", "Finance", "Health", "Utilities"],
            ["Tech", "Finance", "Health", "Utilities"],
        ],
    )
    benchmark_weights = _frame(
        dates,
        columns,
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.10, 0.20, 0.30, 0.40],
            [0.10, 0.20, 0.30, 0.40],
        ],
    )
    market = MarketData(
        frames={
            "close": close,
            "market_cap": market_cap,
            "float_market_cap": market_cap,
            "k200_yn": universe,
            "sector_neutral_big": sectors,
            "bm_weights": benchmark_weights,
        },
        universe=None,
        benchmark=None,
    )
    return PreparedEmp008Factors(
        config=Emp008Config(),
        market=market,
        factor_set_definition=get_factor_set_definition("mfbt"),
        raw_factors=dict(alpha_factors),
        alpha_factors=alpha_factors,
        sector_factors={},
        close=close,
        market_cap=market_cap,
        float_market_cap=market_cap,
        universe=universe,
        sector=sectors,
        benchmark_weights=benchmark_weights,
        monthly_dates=tuple(dates),
    )


def make_emp008_result() -> Emp008Result:
    index = pd.to_datetime(["2024-01-31", "2024-02-29"])
    target_weights = pd.DataFrame({"A": [0.6, 0.55], "B": [0.4, 0.45]}, index=index)
    active_weights = pd.DataFrame({"A": [0.1, 0.05], "B": [-0.1, -0.05]}, index=index)
    diagnostics = pd.DataFrame(
        {
            "target_date": index,
            "success": [True, True],
        }
    )
    return Emp008Result(
        target_weights=target_weights,
        active_weights=active_weights,
        diagnostics=diagnostics,
    )


class _FakeQuantileResult:
    def __init__(self, payload: dict[str, object]) -> None:
        self.write_outputs = Mock(return_value=payload)


@dataclass(frozen=True)
class _FakeBacktestConfig:
    name: str = "mfbt_emp008"


def patch_backtest_report_and_attribution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    report = SimpleNamespace(output_dir=tmp_path / "backtests" / "fake_run")

    class _FakeRunner:
        def __init__(self, **_: object) -> None:
            pass

        def resolve_spec(self, spec: object) -> object:
            return spec

        def run_spec(self, spec: object) -> object:
            return report

    class _FakeReportCli:
        def __init__(self, **_: object) -> None:
            pass

        def run(self, _: list[str]) -> dict[str, object]:
            return {"report_html": str(tmp_path / "reports" / "mfbt_emp008" / "report.html")}

    monkeypatch.setattr(run_full, "BacktestRunner", _FakeRunner)
    monkeypatch.setattr(run_full, "ReportCli", _FakeReportCli)
    monkeypatch.setattr(run_full, "backtest_summary", Mock(return_value={"output_dir": str(report.output_dir)}))
    monkeypatch.setattr(
        run_full,
        "active_share_payload",
        Mock(return_value={"active_share_csv": str(tmp_path / "weights" / "active_share.csv")}),
    )


def test_latest_common_end_uses_required_dataset_minimum_end_date(tmp_path: Path) -> None:
    catalog = DataCatalog.default()
    config = Emp008Config()
    common_index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    shorter_index = pd.to_datetime(["2024-01-02"])

    for dataset_id in required_datasets(config):
        spec = catalog.get(dataset_id)
        index = shorter_index if dataset_id is config.sector_dataset else common_index
        pd.DataFrame({"A": range(len(index))}, index=index).to_parquet(tmp_path / f"{spec.stem}.parquet")

    assert latest_common_end(tmp_path, config) == "2024-01-02"


def test_latest_common_end_treats_month_only_data_as_valid_through_month_end(tmp_path: Path) -> None:
    catalog = DataCatalog.default()
    config = Emp008Config(
        factor_set="origin",
        sector_neutral_dataset=DatasetId.QW_WICS_SEC_BIG,
    )

    for dataset_id in required_datasets(config):
        spec = catalog.get(dataset_id)
        date = "2024-01-15" if spec.validity == "month_only" else "2024-01-31"
        pd.DataFrame({"A": [1.0]}, index=pd.to_datetime([date])).to_parquet(
            tmp_path / f"{spec.stem}.parquet"
        )

    assert latest_common_end(tmp_path, config) == "2024-01-31"


def test_required_datasets_includes_distinct_sector_neutral_dataset() -> None:
    config = Emp008Config(sector_neutral_dataset=DatasetId.QW_WICS_SEC_BIG)

    datasets = required_datasets(config)

    assert DatasetId.QW_WI_SEC_26_BIG in datasets
    assert DatasetId.QW_WICS_SEC_BIG in datasets
    assert len(datasets) == len(set(datasets))


def test_factor_set_parser_choices_match_registry_values() -> None:
    expected = tuple(member.value for member in FactorSetId)

    assert tuple(weights_parser()._option_string_actions["--factor-set"].choices) == expected
    assert tuple(full_parser()._option_string_actions["--factor-set"].choices) == expected


def test_full_parser_exposes_factor_quantile_flags() -> None:
    args = full_parser().parse_args(["--factor-quantiles", "7", "--no-factor-quantiles"])

    assert args.factor_quantiles == 7
    assert args.no_factor_quantiles is True


def test_full_run_prepares_once_and_runs_factor_quantiles_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = make_prepared_bundle()
    prepare = Mock(return_value=prepared)
    optimizer = Mock(return_value=make_emp008_result())
    quantiles = Mock(return_value=_FakeQuantileResult({"summary_csv": str(tmp_path / "factor_quantiles" / "summary.csv")}))
    attribution = Mock(return_value={"excel": str(tmp_path / "factor_attribution" / "factor_attribution.xlsx")})

    monkeypatch.setattr(run_full, "load_and_prepare_emp008_factors", prepare)
    monkeypatch.setattr(run_full, "run_emp008", optimizer)
    monkeypatch.setattr(run_full, "run_emp008_factor_quantiles", quantiles)
    monkeypatch.setattr(run_full, "build_emp008_factor_attribution", attribution)
    patch_backtest_report_and_attribution(monkeypatch, tmp_path)

    run_full.main(["--end", "2024-06-30", "--output-root", str(tmp_path), "--no-comparison"])

    prepare.assert_called_once()
    assert optimizer.call_args.kwargs["prepared"] is prepared
    assert quantiles.call_args.kwargs == {
        "prepared": prepared,
        "start": run_full.DEFAULT_START,
        "end": "2024-06-30",
        "q": 5,
    }
    assert attribution.call_args.kwargs["prepared"] is prepared
    summary = json.loads((tmp_path / "mfbt_emp008" / "run_summary.json").read_text(encoding="utf-8"))
    assert "factor_quantiles" in summary


def test_full_run_can_skip_factor_quantiles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = make_prepared_bundle()
    prepare = Mock(return_value=prepared)
    optimizer = Mock(return_value=make_emp008_result())
    quantiles = Mock()

    monkeypatch.setattr(run_full, "load_and_prepare_emp008_factors", prepare)
    monkeypatch.setattr(run_full, "run_emp008", optimizer)
    monkeypatch.setattr(run_full, "run_emp008_factor_quantiles", quantiles)
    monkeypatch.setattr(
        run_full,
        "build_emp008_factor_attribution",
        Mock(return_value={"excel": str(tmp_path / "factor_attribution" / "factor_attribution.xlsx")}),
    )
    patch_backtest_report_and_attribution(monkeypatch, tmp_path)

    run_full.main(
        [
            "--end",
            "2024-06-30",
            "--output-root",
            str(tmp_path),
            "--no-comparison",
            "--no-factor-quantiles",
        ]
    )

    prepare.assert_called_once()
    assert optimizer.call_args.kwargs["prepared"] is prepared
    quantiles.assert_not_called()
    summary = json.loads((tmp_path / "mfbt_emp008" / "run_summary.json").read_text(encoding="utf-8"))
    assert "factor_quantiles" not in summary


def test_full_run_orders_quantiles_after_weights_and_before_backtest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = make_prepared_bundle()
    events: list[str] = []

    def fake_prepare(**_: object) -> PreparedEmp008Factors:
        events.append("prepare")
        return prepared

    def fake_optimize(**kwargs: object) -> Emp008Result:
        assert kwargs["prepared"] is prepared
        events.append("optimizer")
        return make_emp008_result()

    class _RecordingQuantileResult:
        def write_outputs(self, *_: object, **__: object) -> dict[str, object]:
            events.append("quantiles_write")
            return {"summary_csv": str(tmp_path / "factor_quantiles" / "summary.csv")}

    def fake_quantiles(**kwargs: object) -> _RecordingQuantileResult:
        assert kwargs["prepared"] is prepared
        events.append("quantiles")
        return _RecordingQuantileResult()

    report = SimpleNamespace(output_dir=tmp_path / "backtests" / "fake_run")

    class _RecordingRunner:
        def __init__(self, **_: object) -> None:
            pass

        def resolve_spec(self, spec: object) -> object:
            return spec

        def run_spec(self, spec: object) -> object:
            del spec
            events.append("backtest")
            return report

    class _RecordingReportCli:
        def __init__(self, **_: object) -> None:
            pass

        def run(self, _: list[str]) -> dict[str, object]:
            events.append("report")
            return {"report_html": str(tmp_path / "reports" / "mfbt_emp008" / "report.html")}

    monkeypatch.setattr(run_full, "load_and_prepare_emp008_factors", fake_prepare)
    monkeypatch.setattr(run_full, "run_emp008", fake_optimize)
    monkeypatch.setattr(run_full, "run_emp008_factor_quantiles", fake_quantiles)
    monkeypatch.setattr(run_full, "BacktestRunner", _RecordingRunner)
    monkeypatch.setattr(run_full, "ReportCli", _RecordingReportCli)
    monkeypatch.setattr(run_full, "backtest_summary", Mock(return_value={"output_dir": str(report.output_dir)}))
    monkeypatch.setattr(
        run_full,
        "active_share_payload",
        Mock(return_value={"active_share_csv": str(tmp_path / "weights" / "active_share.csv")}),
    )
    monkeypatch.setattr(
        run_full,
        "build_emp008_factor_attribution",
        Mock(return_value={"excel": str(tmp_path / "factor_attribution" / "factor_attribution.xlsx")}),
    )

    run_full.main(["--end", "2024-06-30", "--output-root", str(tmp_path), "--no-comparison"])

    assert events[:5] == ["prepare", "optimizer", "quantiles", "quantiles_write", "backtest"]
    assert events.index("quantiles_write") < events.index("backtest")
    assert events.index("backtest") < events.index("report")


def test_full_run_propagates_empty_quantile_evaluation_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = make_prepared_bundle()
    prepare = Mock(return_value=prepared)
    optimizer = Mock(return_value=make_emp008_result())
    quantiles = Mock(side_effect=ValueError("no factor quantile observations for mfbt in requested range 2024-01-31 to 2024-06-30"))
    attribution = Mock(return_value={"excel": str(tmp_path / "factor_attribution" / "factor_attribution.xlsx")})

    monkeypatch.setattr(run_full, "load_and_prepare_emp008_factors", prepare)
    monkeypatch.setattr(run_full, "run_emp008", optimizer)
    monkeypatch.setattr(run_full, "run_emp008_factor_quantiles", quantiles)
    monkeypatch.setattr(run_full, "build_emp008_factor_attribution", attribution)
    patch_backtest_report_and_attribution(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="no factor quantile observations for mfbt in requested range 2024-01-31 to 2024-06-30"):
        run_full.main(["--end", "2024-06-30", "--output-root", str(tmp_path), "--no-comparison"])

    prepare.assert_called_once()
    quantiles.assert_called_once()
    assert not (tmp_path / "mfbt_emp008" / "run_summary.json").exists()
    assert attribution.call_count == 0


def test_full_run_propagates_quantile_value_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = make_prepared_bundle()
    monkeypatch.setattr(run_full, "load_and_prepare_emp008_factors", Mock(return_value=prepared))
    monkeypatch.setattr(run_full, "run_emp008", Mock(return_value=make_emp008_result()))
    monkeypatch.setattr(run_full, "run_emp008_factor_quantiles", Mock(side_effect=ValueError("q must be at least 2")))
    monkeypatch.setattr(
        run_full,
        "build_emp008_factor_attribution",
        Mock(return_value={"excel": str(tmp_path / "factor_attribution" / "factor_attribution.xlsx")}),
    )
    patch_backtest_report_and_attribution(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="q must be at least 2"):
        run_full.main(["--end", "2024-06-30", "--output-root", str(tmp_path), "--no-comparison"])


def test_full_run_propagates_quantile_writer_value_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = make_prepared_bundle()
    writer_result = _FakeQuantileResult({"summary_csv": str(tmp_path / "factor_quantiles" / "summary.csv")})
    writer_result.write_outputs.side_effect = ValueError("result must be nonempty before writing artifacts")

    monkeypatch.setattr(run_full, "load_and_prepare_emp008_factors", Mock(return_value=prepared))
    monkeypatch.setattr(run_full, "run_emp008", Mock(return_value=make_emp008_result()))
    monkeypatch.setattr(run_full, "run_emp008_factor_quantiles", Mock(return_value=writer_result))
    monkeypatch.setattr(
        run_full,
        "build_emp008_factor_attribution",
        Mock(return_value={"excel": str(tmp_path / "factor_attribution" / "factor_attribution.xlsx")}),
    )
    patch_backtest_report_and_attribution(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="result must be nonempty before writing artifacts"):
        run_full.main(["--end", "2024-06-30", "--output-root", str(tmp_path), "--no-comparison"])


@pytest.mark.parametrize(
    ("factor_set", "dividend_dataset"),
    [
        ("origin", DatasetId.QW_DIVIDEND_YLD_FY0),
        ("origin_new_dividend", DatasetId.QW_DPS_TTM),
    ],
)
def test_origin_required_datasets_include_only_inputs_used_by_the_strategy(
    factor_set: str,
    dividend_dataset: DatasetId,
) -> None:
    config = Emp008Config(
        factor_set=factor_set,
        sector_neutral_dataset=DatasetId.QW_WICS_SEC_BIG,
    )

    assert set(required_datasets(config)) == {
        DatasetId.QW_ADJ_C,
        DatasetId.QW_BM_WEIGHTS,
        DatasetId.QW_WICS_SEC_BIG,
        DatasetId.QW_MKTCAP,
        DatasetId.QW_MKTCAP_FLT,
        DatasetId.QW_K200_YN,
        dividend_dataset,
    }


def test_required_datasets_follow_registry_dataset_selection_by_factor_set() -> None:
    origin = required_datasets(Emp008Config(factor_set="origin"))
    origin_new = required_datasets(Emp008Config(factor_set="origin_new_dividend"))
    mfbt = required_datasets(Emp008Config())

    assert DatasetId.QW_DIVIDEND_YLD_FY0 in origin
    assert DatasetId.QW_DPS_TTM not in origin
    assert DatasetId.QW_DPS_TTM in origin_new
    assert DatasetId.QW_DIVIDEND_YLD_FY0 not in origin_new
    assert DatasetId.QW_FCF in mfbt
    assert DatasetId.QW_DIVIDEND_YLD_FY0 not in mfbt


def test_origin_required_datasets_with_wics_include_common_inputs_neutral_sector_and_selected_dividend_only() -> None:
    config = Emp008Config(
        factor_set="origin",
        sector_neutral_dataset=DatasetId.QW_WICS_SEC_BIG,
    )

    assert required_datasets(config) == (
        DatasetId.QW_ADJ_C,
        DatasetId.QW_BM_WEIGHTS,
        DatasetId.QW_DIVIDEND_YLD_FY0,
        DatasetId.QW_WICS_SEC_BIG,
        DatasetId.QW_MKTCAP,
        DatasetId.QW_MKTCAP_FLT,
        DatasetId.QW_K200_YN,
    )


def test_mfbt_required_datasets_omit_unused_raw_close() -> None:
    datasets = required_datasets(Emp008Config())

    assert DatasetId.QW_C not in datasets


def test_mfbt_required_datasets_with_wics_include_both_construction_and_neutral_sector_datasets() -> None:
    config = Emp008Config(sector_neutral_dataset=DatasetId.QW_WICS_SEC_BIG)

    assert required_datasets(config) == (
        DatasetId.QW_ADJ_C,
        DatasetId.QW_BM_WEIGHTS,
        DatasetId.QW_OP_FWD_12M,
        DatasetId.QW_DPS_TTM,
        DatasetId.QW_RETAIL,
        DatasetId.QW_FCF,
        DatasetId.QW_INT_BEARING_LIAB_NFQ0,
        DatasetId.QW_QUICK_ASSETS_NFQ0,
        DatasetId.QW_WI_SEC_26_BIG,
        DatasetId.QW_WICS_SEC_BIG,
        DatasetId.QW_MKTCAP,
        DatasetId.QW_MKTCAP_FLT,
        DatasetId.QW_K200_YN,
    )


def test_complete_benchmark_history_uses_float_cap_only_before_official_weights() -> None:
    dates = pd.to_datetime(["2019-11-29", "2019-12-30", "2020-01-02"])
    bm_weights = pd.DataFrame(
        {"A": [np.nan, np.nan, 0.7], "B": [np.nan, np.nan, 0.3]},
        index=dates,
    )
    float_mktcap = pd.DataFrame(
        {"A": [60.0, 80.0, 90.0], "B": [40.0, 20.0, 10.0]},
        index=dates,
    )
    universe = pd.DataFrame(True, index=dates, columns=["A", "B"])
    result = complete_benchmark_history(bm_weights, float_mktcap, universe)

    assert result.loc["2019-11-29"].tolist() == pytest.approx([0.6, 0.4])
    assert result.loc["2019-12-30"].tolist() == pytest.approx([0.8, 0.2])
    assert result.loc["2020-01-02"].tolist() == pytest.approx([0.7, 0.3])


def test_load_market_keeps_retail_and_sector_neutral_frames_separate(tmp_path: Path) -> None:
    catalog = DataCatalog.default()
    config = Emp008Config(sector_neutral_dataset=DatasetId.QW_WICS_SEC_BIG)
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])

    for dataset_id in required_datasets(config):
        spec = catalog.get(dataset_id)
        if dataset_id is config.sector_dataset:
            frame = pd.DataFrame({"A": ["WI100", "WI100"]}, index=index)
        elif dataset_id is config.sector_neutral_dataset:
            frame = pd.DataFrame({"A": ["G45", "G45"]}, index=index)
        else:
            frame = pd.DataFrame({"A": [1.0, 1.0]}, index=index)
        frame.to_parquet(tmp_path / f"{spec.stem}.parquet")

    market = load_emp008_market(
        parquet_dir=tmp_path,
        start="2024-01-02",
        end="2024-01-03",
        config=config,
    )

    assert market.frames["sector_big"].loc["2024-01-03", "A"] == "WI100"
    assert market.frames["sector_neutral_big"].loc["2024-01-03", "A"] == "G45"


def test_write_target_weights_csv_uses_iso_date_index(tmp_path: Path) -> None:
    weights = pd.DataFrame(
        {"A": [0.6], "B": [0.4]},
        index=pd.to_datetime(["2024-01-31"]),
    )

    path = write_target_weights_csv(weights, tmp_path / "target_weights.csv")

    assert path == tmp_path / "target_weights.csv"
    text = path.read_text(encoding="utf-8")
    assert "2024-01-31" in text


def test_build_target_weight_spec_uses_weight_dates_as_custom_schedule(tmp_path: Path) -> None:
    weights_csv = tmp_path / "target_weights.csv"
    dates = ("2024-01-31", "2024-02-29")

    spec = build_target_weight_spec(
        name="emp008_test",
        weights_csv=weights_csv,
        dates=dates,
        end="2024-02-29",
        fill_mode="close",
    )

    assert spec.start == "2024-01-31"
    assert spec.end == "2024-02-29"
    assert spec.name == "emp008_test"
    assert spec.target_weights is not None
    assert spec.target_weights.path == str(weights_csv)
    assert spec.schedule.kind == "custom_dates"
    assert spec.schedule.dates == dates


def test_build_target_weight_spec_preserves_backtest_conditions(tmp_path: Path) -> None:
    weights_csv = tmp_path / "target_weights.csv"

    spec = build_target_weight_spec(
        name="emp008_costed",
        weights_csv=weights_csv,
        dates=("2024-01-31",),
        end="2024-01-31",
        fill_mode="next_open",
        capital=250_000_000.0,
        fee=0.0002,
        sell_tax=0.0015,
        slippage=0.0005,
        allow_fractional=False,
    )

    assert spec.capital == 250_000_000.0
    assert spec.fill_mode == "next_open"
    assert spec.fee == 0.0002
    assert spec.sell_tax == 0.0015
    assert spec.slippage == 0.0005
    assert spec.allow_fractional is False


def test_default_weights_csv_points_to_named_weights_run() -> None:
    assert default_weights_csv(Path("results") / "emp008_runs", "emp008") == Path(
        "results/emp008_runs/emp008/weights/target_weights.csv"
    )


def test_resolve_run_output_dirs_defaults_backtests_and_reports_inside_run_root() -> None:
    run_root, backtests_root, reports_root = resolve_run_output_dirs(
        output_root=Path("results") / "emp008_runs",
        name="mfbt_emp008",
        backtests_root=None,
        reports_root=None,
    )

    assert run_root == Path("results/emp008_runs/mfbt_emp008")
    assert backtests_root == Path("results/emp008_runs/mfbt_emp008/backtests")
    assert reports_root == Path("results/emp008_runs/mfbt_emp008/reports")


def test_resolve_run_output_dirs_preserves_explicit_roots() -> None:
    run_root, backtests_root, reports_root = resolve_run_output_dirs(
        output_root=Path("results") / "emp008_runs",
        name="mfbt_emp008",
        backtests_root=Path("custom_backtests"),
        reports_root=Path("custom_reports"),
    )

    assert run_root == Path("results/emp008_runs/mfbt_emp008")
    assert backtests_root == Path("custom_backtests")
    assert reports_root == Path("custom_reports")


def test_load_weight_dates_reads_iso_dates_from_csv(tmp_path: Path) -> None:
    path = tmp_path / "target_weights.csv"
    path.write_text(",A\n2024-01-31,1.0\n2024-02-29,1.0\n", encoding="utf-8")

    assert load_weight_dates(path) == ("2024-01-31", "2024-02-29")


def test_write_active_share_uses_half_active_weight_l1_norm(tmp_path: Path) -> None:
    active = pd.DataFrame(
        {"A": [0.10, -0.02], "B": [-0.04, 0.02], "C": [-0.06, 0.00]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    active_path = tmp_path / "active_weights.parquet"
    active.to_parquet(active_path)

    output = write_active_share(active_path)

    assert output["active_share_csv"] == str(tmp_path / "active_share.csv")
    assert output["active_share_parquet"] == str(tmp_path / "active_share.parquet")
    result = pd.read_csv(tmp_path / "active_share.csv")
    assert result["date"].tolist() == ["2024-01-31", "2024-02-29"]
    assert result["active_share"].tolist() == pytest.approx([0.10, 0.02])
    assert result["active_share_pct"].tolist() == pytest.approx([10.0, 2.0])


def test_active_share_summary_reports_monthly_distribution(tmp_path: Path) -> None:
    active_share = pd.DataFrame(
        {"active_share": [0.10, 0.02], "active_share_pct": [10.0, 2.0]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    path = tmp_path / "active_share.parquet"
    active_share.to_parquet(path)

    summary = active_share_summary(path)

    assert summary["rows"] == 2
    assert summary["date_start"] == "2024-01-31"
    assert summary["date_end"] == "2024-02-29"
    assert summary["mean_pct"] == pytest.approx(6.0)
    assert summary["max_pct"] == pytest.approx(10.0)


def test_active_share_payload_also_writes_saved_backtest_series(tmp_path: Path) -> None:
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    pd.DataFrame(
        {"A": [0.10], "B": [-0.10]},
        index=pd.to_datetime(["2024-01-31"]),
    ).to_parquet(weights_dir / "active_weights.parquet")
    weights_csv = weights_dir / "target_weights.csv"
    weights_csv.write_text(",A,B\n2024-01-31,0.6,0.4\n", encoding="utf-8")
    backtest_dir = tmp_path / "backtest"

    payload = active_share_payload(weights_csv, backtest_output_dir=backtest_dir)

    assert Path(payload["active_share_csv"]).exists()
    assert Path(payload["backtest_active_share_csv"]).exists()
    assert Path(payload["backtest_active_share_csv"]).parent == backtest_dir / "series"


def test_build_emp008_config_converts_annual_tracking_error_to_monthly() -> None:
    config = build_emp008_config(tracking_error_annual=0.03)

    assert config.tracking_error == pytest.approx(0.03 / (12**0.5))


def test_build_emp008_config_sets_direct_covariance_risk_model() -> None:
    config = build_emp008_config(tracking_error_annual=0.007, risk_model="direct_covariance")

    assert config.risk_model == "direct_covariance"
    assert config.tracking_error == pytest.approx(0.007 / (12**0.5))


def test_build_emp008_config_rejects_negative_tracking_error() -> None:
    with pytest.raises(ValueError, match="tracking error"):
        build_emp008_config(tracking_error_annual=-0.01)


def test_build_emp008_config_rejects_unknown_risk_model() -> None:
    with pytest.raises(ValueError, match="risk_model"):
        build_emp008_config(risk_model="raw_cov")


def test_build_emp008_config_sets_origin_three_factor_variant() -> None:
    config = build_emp008_config(tracking_error_annual=0.007, factor_set="origin")

    assert config.factor_set is FactorSetId.ORIGIN
    assert config.expected_alpha_policy == "origin_sign"
    assert config.rank_transform_factors == ("LnMktcap",)
    assert config.large_bm_neutral_factor_names == ()
    assert config.monthly_snapshot_forward_days == 7
    assert config.tracking_error == pytest.approx(0.007 / (12**0.5))
    assert DatasetId.QW_DIVIDEND_YLD_FY0 in required_datasets(config)


def test_build_emp008_config_sets_origin_with_new_dividend_variant() -> None:
    config = build_emp008_config(tracking_error_annual=0.007, factor_set="origin_new_dividend")

    assert config.factor_set is FactorSetId.ORIGIN_NEW_DIVIDEND
    assert config.expected_alpha_policy == "origin_sign"
    assert config.rank_transform_factors == ("LnMktcap",)
    assert config.large_bm_neutral_factor_names == ()
    assert config.monthly_snapshot_forward_days == 0
    assert config.tracking_error == pytest.approx(0.007 / (12**0.5))
    assert DatasetId.QW_DIVIDEND_YLD_FY0 not in required_datasets(config)


def test_build_emp008_config_sets_mfbt_positivity_variant() -> None:
    config = build_emp008_config(tracking_error_annual=0.007, factor_set="mfbt_pos")

    assert config.factor_set is FactorSetId.MFBT_POS
    assert config.expected_alpha_policy == "mean"
    assert config.rank_transform_factors == ("ln_market_cap",)
    assert config.tracking_error == pytest.approx(0.007 / (12**0.5))
    assert DatasetId.QW_DIVIDEND_YLD_FY0 not in required_datasets(config)


def test_build_emp008_config_adds_only_origin_small_cap_policy_to_mfbt() -> None:
    baseline = build_emp008_config(tracking_error_annual=0.007, factor_set="mfbt")

    config = build_emp008_config(tracking_error_annual=0.007, factor_set="mfbt_origin_smallcap")

    assert config.factor_set is FactorSetId.MFBT_ORIGIN_SMALLCAP
    assert config.expected_alpha_policy == "origin_small_cap"
    assert config.rank_transform_factors == baseline.rank_transform_factors
    assert config.large_bm_neutral_factor_names == baseline.large_bm_neutral_factor_names
    assert config.monthly_snapshot_forward_days == baseline.monthly_snapshot_forward_days
    assert config.tracking_error == baseline.tracking_error
    assert required_datasets(config) == required_datasets(baseline)


def test_build_emp008_config_sets_wics_sector_neutral_dataset() -> None:
    config = build_emp008_config(tracking_error_annual=0.007, sector_neutral_dataset="wics")

    assert config.sector_dataset == DatasetId.QW_WI_SEC_26_BIG
    assert config.sector_neutral_dataset == DatasetId.QW_WICS_SEC_BIG
    assert DatasetId.QW_WI_SEC_26_BIG in required_datasets(config)
    assert DatasetId.QW_WICS_SEC_BIG in required_datasets(config)


def test_build_emp008_config_rejects_unknown_factor_set() -> None:
    with pytest.raises(
        ValueError,
        match="unknown factor set 'legacy'. Supported values: mfbt, mfbt_pos, mfbt_origin_smallcap, origin, origin_new_dividend",
    ):
        build_emp008_config(factor_set="legacy")


def test_preprocess_factor_frame_caps_final_zscore_exposure() -> None:
    raw = pd.DataFrame(
        {"A": [0.0], "B": [1.0], "C": [100.0]},
        index=pd.to_datetime(["2024-01-31"]),
    )
    float_mktcap = pd.DataFrame(
        {"A": [1.0], "B": [1.0], "C": [0.01]},
        index=raw.index,
    )
    universe = pd.DataFrame(True, index=raw.index, columns=raw.columns)

    result = preprocess_factor_frame(raw, float_mktcap, universe, zscore_cap=2.0)

    assert result.abs().max(axis=1).iloc[0] <= 2.0


def test_preprocess_factor_frame_winsorizes_raw_cross_section_before_zscore() -> None:
    raw = pd.DataFrame(
        {"A": [0.0], "B": [1.0], "C": [10.0], "D": [100.0]},
        index=pd.to_datetime(["2024-01-31"]),
    )
    float_mktcap = pd.DataFrame(
        {"A": [1.0], "B": [1.0], "C": [1.0], "D": [0.01]},
        index=raw.index,
    )
    universe = pd.DataFrame(True, index=raw.index, columns=raw.columns)

    baseline = preprocess_factor_frame(raw, float_mktcap, universe)
    winsorized = preprocess_factor_frame(raw, float_mktcap, universe, winsor_quantile=0.20)

    assert winsorized.loc["2024-01-31", "D"] < baseline.loc["2024-01-31", "D"]


def test_origin_raw_factors_use_ln_mktcap_twelve_month_momentum_and_fy0_dividend_yield() -> None:
    dates = pd.date_range("2023-01-31", "2024-02-29", freq="ME")
    close = pd.DataFrame({"A": [100.0, *([101.0] * 11), 120.0, 99.0]}, index=dates)
    market_cap = pd.DataFrame({"A": [1000.0, *([1100.0] * 11), 1300.0, 1500.0]}, index=dates)
    dividend_yld = pd.DataFrame({"A": [0.50, *([0.55] * 11), 0.60, 0.65]}, index=dates)
    market = MarketData(
        frames={
            "close": close,
            "market_cap": market_cap,
            "dividend_yld_fy0": dividend_yld,
        },
        universe=None,
        benchmark=None,
    )

    factors = build_raw_factors(market, Emp008Config(factor_set="origin"))

    assert list(factors) == ["LnMktcap", "Momentum_12M", "DY"]
    assert factors["LnMktcap"].loc["2024-02-29", "A"] == pytest.approx(np.log(1500.0))
    assert factors["Momentum_12M"].loc["2024-01-31", "A"] == pytest.approx(0.20)
    assert factors["Momentum_12M"].loc["2024-02-29", "A"] == pytest.approx(99.0 / 101.0 - 1.0)
    assert factors["DY"].loc["2024-02-29", "A"] == pytest.approx(0.65)


def test_origin_new_dividend_keeps_origin_size_and_momentum_but_uses_ttm_dividend_yield() -> None:
    dates = pd.date_range("2023-01-31", "2024-02-29", freq="ME")
    close = pd.DataFrame({"A": [100.0, *([101.0] * 11), 120.0, 100.0]}, index=dates)
    market_cap = pd.DataFrame({"A": [1000.0, *([1100.0] * 11), 1300.0, 1500.0]}, index=dates)
    dps_ttm = pd.DataFrame({"A": [2.0, *([2.0] * 11), 3.0, 4.0]}, index=dates)
    market = MarketData(
        frames={"close": close, "market_cap": market_cap, "dps_ttm": dps_ttm},
        universe=None,
        benchmark=None,
    )

    factors = build_raw_factors(market, Emp008Config(factor_set="origin_new_dividend"))

    assert list(factors) == ["LnMktcap", "Momentum_12M", "dividend_yield"]
    assert factors["LnMktcap"].loc["2024-02-29", "A"] == pytest.approx(np.log(1500.0))
    assert factors["Momentum_12M"].loc["2024-01-31", "A"] == pytest.approx(0.20)
    assert factors["dividend_yield"].loc["2024-02-29", "A"] == pytest.approx(0.04)


def test_mfbt_positivity_raw_factors_replace_price_high_ratio_with_rolling_positivity() -> None:
    dates = pd.bdate_range("2024-01-02", periods=24)
    close = pd.DataFrame(
        {
            "A": [100.0, 101.0, 100.0, 102.0, 104.0, 103.0, *([104.0] * 18)],
            "B": [100.0, 99.0, 98.0, 99.0, 98.0, 97.0, *([96.0] * 18)],
        },
        index=dates,
    )
    market = MarketData(
        frames={
            "close": close,
            "op_fwd_12m": close * 1_000_000_000.0,
            "dps_ttm": close * 0.02,
            "retail_flow": pd.DataFrame(1.0, index=dates, columns=close.columns),
            "sector_big": pd.DataFrame("Tech", index=dates, columns=close.columns),
            "market_cap": close * 10_000_000.0,
            "free_cash_flow": close * 10_000.0,
            "interest_bearing_liability": close * 1_000.0,
            "quick_asset": close * 500.0,
        },
        universe=None,
        benchmark=None,
    )

    config = Emp008Config(
        factor_set="mfbt_pos",
        positivity_momentum_lookback_days=3,
        retail_flow_lookback_days=3,
    )
    factors = build_raw_factors(market, config)

    assert list(factors) == [
        "positivity_momentum",
        "earnings_momentum",
        "dividend_yield",
        "retail_flow",
        "value",
        "ln_market_cap",
    ]
    assert "price_momentum" not in factors
    expected = close.pct_change(fill_method=None).ge(0.0).rolling(3, min_periods=3).mean()
    month_end = dates[-1]
    assert factors["positivity_momentum"].loc[month_end, "A"] == pytest.approx(expected.loc[month_end, "A"])
    assert factors["positivity_momentum"].loc[month_end, "B"] == pytest.approx(expected.loc[month_end, "B"])


def test_origin_dividend_yield_maps_later_month_snapshot_to_close_month_end() -> None:
    close_dates = pd.to_datetime(["2026-04-30", "2026-05-28"])
    close = pd.DataFrame({"A": [100.0, 110.0]}, index=close_dates)
    market_cap = pd.DataFrame({"A": [1000.0, 1200.0]}, index=close_dates)
    dividend_yld = pd.DataFrame(
        {"A": [0.60, 0.70]},
        index=pd.to_datetime(["2026-04-30", "2026-05-29"]),
    )
    market = MarketData(
        frames={
            "close": close,
            "market_cap": market_cap,
            "dividend_yld_fy0": dividend_yld,
        },
        universe=None,
        benchmark=None,
    )

    factors = build_raw_factors(market, Emp008Config(factor_set="origin"))

    assert factors["DY"].loc["2026-05-28", "A"] == pytest.approx(0.70)


def test_origin_market_load_keeps_forward_snapshot_but_trims_price_frames_to_requested_end() -> None:
    dates = pd.to_datetime(["2026-05-28", "2026-05-29"])
    market = MarketData(
        frames={
            "close": pd.DataFrame({"A": [100.0, 101.0]}, index=dates),
            "dividend_yld_fy0": pd.DataFrame({"A": [0.60, 0.70]}, index=dates),
        },
        universe=None,
        benchmark=None,
    )

    result = _trim_non_forward_snapshot_frames(
        market,
        end="2026-05-28",
        config=Emp008Config(factor_set="origin"),
    )

    assert result.frames["close"].index.max() == pd.Timestamp("2026-05-28")
    assert result.frames["dividend_yld_fy0"].index.max() == pd.Timestamp("2026-05-29")


def test_origin_snapshot_padding_comes_from_registry_without_constructor_override() -> None:
    config = Emp008Config(factor_set="origin")

    assert config.monthly_snapshot_forward_days == 7
    assert padded_snapshot_end("2026-05-28", config) == "2026-06-04"


def test_origin_expected_alpha_policy_matches_w_emp008_sign_rules() -> None:
    expected_alpha = pd.Series(
        {
            "LnMktcap": 0.01,
            "Momentum_12M": -0.02,
            "DY": 0.03,
            "sector_tech": 0.0,
        }
    )

    result = apply_expected_alpha_policy(
        expected_alpha,
        Emp008Config(factor_set="origin"),
    )

    assert result["LnMktcap"] == 0.0
    assert result["Momentum_12M"] == 0.0
    assert result["DY"] == pytest.approx(0.03)
    assert result["sector_tech"] == 0.0


def test_origin_expected_alpha_policy_applies_dividend_direction_to_new_dividend_name() -> None:
    expected_alpha = pd.Series({"LnMktcap": -0.01, "Momentum_12M": 0.02, "dividend_yield": -0.03})

    result = apply_expected_alpha_policy(
        expected_alpha,
        Emp008Config(factor_set="origin_new_dividend"),
    )

    assert result["LnMktcap"] == pytest.approx(-0.01)
    assert result["Momentum_12M"] == pytest.approx(0.02)
    assert result["dividend_yield"] == 0.0


def test_mfbt_origin_small_cap_policy_preserves_origin_factor_directions() -> None:
    expected_alpha = pd.Series(
        {
            "ln_market_cap": 0.01,
            "price_momentum": -0.02,
            "earnings_momentum": -0.025,
            "dividend_yield": -0.03,
            "retail_flow": -0.035,
            "value": -0.04,
            "sector_tech": 0.0,
        }
    )

    result = apply_expected_alpha_policy(
        expected_alpha,
        Emp008Config(factor_set="mfbt_origin_smallcap"),
    )

    assert result["ln_market_cap"] == 0.0
    assert result.eq(0.0).all()


def test_mfbt_origin_small_cap_policy_keeps_negative_size_alpha() -> None:
    expected_alpha = pd.Series({"ln_market_cap": -0.01, "value": 0.04})

    result = apply_expected_alpha_policy(
        expected_alpha,
        Emp008Config(factor_set="mfbt_origin_smallcap"),
    )

    assert result.equals(expected_alpha)


def test_expected_alpha_policy_ignores_unknown_and_sector_names() -> None:
    expected_alpha = pd.Series({"mystery_factor": -0.5, "sector_tech": 0.1, "value": -0.02})

    result = apply_expected_alpha_policy(
        expected_alpha,
        Emp008Config(factor_set="mfbt_origin_smallcap"),
    )

    assert result["mystery_factor"] == pytest.approx(-0.5)
    assert result["sector_tech"] == pytest.approx(0.1)
    assert result["value"] == 0.0


def test_direct_covariance_optimizer_uses_stock_covariance_risk_budget() -> None:
    exposures = pd.DataFrame(
        {
            "value": {"A": 1.0, "B": -1.0, "C": 0.0},
            "sector_tech": {"A": 0.5, "B": 0.5, "C": -1.0},
        }
    )
    stock_cov = pd.DataFrame(
        {
            "A": {"A": 0.04, "B": 0.01, "C": 0.0},
            "B": {"A": 0.01, "B": 0.04, "C": 0.0},
            "C": {"A": 0.0, "B": 0.0, "C": 0.02},
        }
    )
    expected_alpha = pd.Series({"value": 0.10, "sector_tech": 0.0})
    bm_weights = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})

    result = optimize_active_weights_with_covariance(
        exposures=exposures,
        stock_cov=stock_cov,
        expected_alpha=expected_alpha,
        bm_weights=bm_weights,
        sector_factor_names=["sector_tech"],
        tracking_error=0.03,
    )

    realized_te = (result.active_weights.to_numpy().T @ stock_cov.to_numpy() @ result.active_weights.to_numpy()) ** 0.5
    assert result.success is True
    assert result.active_weights.sum() == pytest.approx(0.0, abs=1e-8)
    assert realized_te <= 0.0300001
    assert result.tracking_error == pytest.approx(realized_te)


def test_optimizer_keeps_every_benchmark_constituent_above_holding_floor() -> None:
    exposures = pd.DataFrame(
        {"value": {"A": 1.0, "B": 0.0, "C": -1.0}}
    )
    benchmark_weights = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})

    result = optimize_active_weights_with_covariance(
        exposures=exposures,
        stock_cov=pd.DataFrame(np.eye(3), index=exposures.index, columns=exposures.index),
        expected_alpha=pd.Series({"value": 1.0}),
        bm_weights=benchmark_weights,
        sector_factor_names=[],
        tracking_error=10.0,
    )

    assert result.success is True
    assert result.final_weights.gt(1e-12).all()
    assert result.final_weights.sum() == pytest.approx(1.0)
    assert result.active_weights.sum() == pytest.approx(0.0)


def test_optimizer_selects_only_independent_sector_constraints() -> None:
    exposures = pd.DataFrame(
        {
            "value": {"A": 1.0, "B": -1.0, "C": 0.0},
            "sector_tech": {"A": 0.5, "B": 0.5, "C": -1.0},
            "sector_other": {"A": -0.5, "B": -0.5, "C": 1.0},
        }
    )
    selector = getattr(emp008_optimize, "_independent_sector_factor_names", None)

    assert callable(selector)
    assert selector(exposures, ["sector_tech", "sector_other"]) == ["sector_tech"]


def test_optimizer_retries_numeric_failure_with_relaxed_ftol(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []
    constraint_counts: list[int] = []

    def fake_minimize(**kwargs: object) -> object:
        options = kwargs["options"]
        constraints = kwargs["constraints"]
        assert isinstance(options, dict)
        assert isinstance(constraints, list)
        calls.append(float(options["ftol"]))
        constraint_counts.append(len(constraints))
        return type(
            "Result",
            (),
            {
                "success": len(calls) == 2,
                "x": np.zeros(3),
                "fun": 0.0,
            },
        )()

    monkeypatch.setattr(emp008_optimize, "minimize", fake_minimize)
    exposures = pd.DataFrame(
        {
            "value": {"A": 1.0, "B": -1.0, "C": 0.0},
            "sector_tech": {"A": 0.5, "B": 0.5, "C": -1.0},
            "sector_other": {"A": -0.5, "B": -0.5, "C": 1.0},
        }
    )

    result = optimize_active_weights_with_covariance(
        exposures=exposures,
        stock_cov=pd.DataFrame(np.eye(3), index=exposures.index, columns=exposures.index),
        expected_alpha=pd.Series({"value": 0.1, "sector_tech": 0.0, "sector_other": 0.0}),
        bm_weights=pd.Series({"A": 0.4, "B": 0.4, "C": 0.2}),
        sector_factor_names=["sector_tech", "sector_other"],
        tracking_error=0.03,
    )

    assert result.success is True
    assert calls == [1e-12, 1e-9]
    assert constraint_counts == [3, 3]


def test_stock_excess_covariance_fills_returns_before_covariance_to_keep_psd() -> None:
    stock_excess = pd.DataFrame(
        {
            "A": [0.125730, -0.535669, None, -2.325031, -0.544259, -0.128535],
            "B": [-0.132105, 0.361595, -1.265421, -0.218792, None, 1.366463],
            "C": [0.640423, 1.304000, -0.623274, -1.245911, 0.411631, None],
            "D": [0.104900, 0.947081, None, -0.732267, None, 0.351510],
        }
    )

    cov = _stock_excess_covariance_for_target_universe(
        stock_excess,
        target_tickers=pd.Index(["A", "B", "C", "D"]),
        window=6,
    )

    min_eigenvalue = np.linalg.eigvalsh(cov.to_numpy()).min()
    assert min_eigenvalue >= -1e-12


def test_active_weight_abs_sum_frame_reports_sum_abs_and_active_share() -> None:
    active = pd.DataFrame(
        {"A": [0.10, -0.02], "B": [-0.04, 0.02], "C": [-0.06, 0.00]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )

    result = active_weight_abs_sum_frame(active)

    assert result["sum_abs_active_weight"].tolist() == pytest.approx([0.20, 0.04])
    assert result["active_share"].tolist() == pytest.approx([0.10, 0.02])
    assert result["sum_abs_active_weight_pct"].tolist() == pytest.approx([20.0, 4.0])
    assert result["active_share_pct"].tolist() == pytest.approx([10.0, 2.0])


def test_sufficient_risk_history_requires_full_configured_window() -> None:
    config = Emp008Config(risk_window=36)
    factors = pd.DataFrame({"factor": range(36)})

    assert _has_sufficient_risk_history(factors.iloc[:35], config) is False
    assert _has_sufficient_risk_history(factors, config) is True


def test_monthly_compounded_returns_uses_within_month_compounding() -> None:
    returns = pd.DataFrame(
        {"Gross excess": [0.01, 0.02, -0.01]},
        index=pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-01"]),
    )

    result = monthly_compounded_returns(returns)

    assert result.index.strftime("%Y-%m-%d").tolist() == ["2024-01-31", "2024-02-29"]
    assert result["Gross excess"].tolist() == pytest.approx([(1.01 * 1.02) - 1.0, -0.01])


def test_excess_summary_bps_reports_total_and_monthly_bps() -> None:
    active_returns = pd.DataFrame(
        {"Gross excess": [0.01, 0.02, -0.01]},
        index=pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-01"]),
    )

    result = excess_summary_bps(active_returns, periods_per_year=252)

    expected_total = (1.01 * 1.02 * 0.99) - 1.0
    expected_monthly_mean = (((1.01 * 1.02) - 1.0) + -0.01) / 2.0
    assert result.loc["Gross excess", "total_excess_bp"] == pytest.approx(expected_total * 10_000.0)
    assert result.loc["Gross excess", "monthly_mean_excess_bp"] == pytest.approx(expected_monthly_mean * 10_000.0)


def test_monthly_excess_heatmap_frame_pivots_year_by_month() -> None:
    monthly_active = pd.DataFrame(
        {"Gross excess": [0.01, -0.02]},
        index=pd.to_datetime(["2024-01-31", "2025-03-31"]),
    )

    result = monthly_excess_heatmap_frame(monthly_active, "Gross excess")

    assert result.loc[2024, 1] == pytest.approx(1.0)
    assert result.loc[2025, 3] == pytest.approx(-2.0)
    assert pd.isna(result.loc[2024, 2])


def test_build_emp008_comparison_writes_core_artifacts(tmp_path: Path) -> None:
    dates = pd.bdate_range("2024-01-31", periods=24)
    gross_dir = tmp_path / "gross"
    costed_dir = tmp_path / "costed"
    for run_dir in (gross_dir, costed_dir):
        (run_dir / "series").mkdir(parents=True)

    pd.DataFrame(
        {"date": dates, "returns": [0.001] * len(dates)},
    ).to_csv(gross_dir / "series" / "returns.csv", index=False)
    pd.DataFrame(
        {"date": dates, "returns": [0.0008] * len(dates)},
    ).to_csv(costed_dir / "series" / "returns.csv", index=False)

    benchmark_dates = dates.insert(0, dates[0] - pd.offsets.BDay(1))
    benchmark = pd.DataFrame(
        {("IKS200", "close"): [100.0 + idx * 0.05 for idx in range(len(benchmark_dates))]},
        index=benchmark_dates,
    )
    benchmark.columns = pd.MultiIndex.from_tuples(benchmark.columns, names=["code", "field"])
    benchmark_path = tmp_path / "benchmark.parquet"
    benchmark.to_parquet(benchmark_path)

    active_weights = pd.DataFrame(
        {"A": [0.01, 0.02], "B": [-0.01, -0.02]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    active_path = tmp_path / "active_weights.parquet"
    active_weights.to_parquet(active_path)

    payload = build_emp008_comparison(
        gross_run_dir=gross_dir,
        costed_run_dir=costed_dir,
        active_weights_parquet=active_path,
        benchmark_parquet=benchmark_path,
        output_dir=tmp_path / "comparison",
    )

    for key in (
        "performance_xlsx",
        "cumulative_png",
        "monthly_excess_heatmap_png",
        "active_weight_sum_png",
        "active_weight_sum_csv",
        "active_weight_sum_xlsx",
    ):
        assert Path(payload[key]).exists()
    assert (tmp_path / "comparison_summary.json").exists()
    assert "excess_summary_bps" in payload
    assert "active_weight_sum" in payload

    daily_returns = pd.read_excel(payload["performance_xlsx"], sheet_name="daily_returns", index_col=0)
    assert daily_returns.loc[dates[0], "KOSPI200 BM"] == pytest.approx(0.0)


def test_factor_attribution_row_reconciles_factor_and_residual_contribution() -> None:
    active = pd.Series({"A": 0.10, "B": -0.10})
    exposures = pd.DataFrame(
        {
            "value": {"A": 1.0, "B": -1.0},
            "momentum": {"A": 0.5, "B": 0.5},
            "sector": {"A": 0.2, "B": -0.2},
        }
    )
    factor_returns = pd.Series({"value": 0.03, "momentum": 0.02, "sector": -0.01})
    residuals = pd.Series({"A": 0.004, "B": -0.001})

    row = factor_attribution_row(
        active_weights=active,
        exposures=exposures,
        factor_returns=factor_returns,
        residuals=residuals,
        alpha_factor_names=["value", "momentum"],
        sector_factor_names=["sector"],
    )

    assert row["value"] == pytest.approx(0.006)
    assert row["momentum"] == pytest.approx(0.0)
    assert row["alpha_total"] == pytest.approx(0.006)
    assert row["sector_total"] == pytest.approx(-0.0004)
    assert row["specific"] == pytest.approx(0.0005)
    assert row["model_active_return"] == pytest.approx(0.0061)


def test_write_factor_attribution_writes_excel_and_core_charts(tmp_path: Path) -> None:
    index = pd.to_datetime(["2024-01-31", "2024-02-29"])
    monthly = pd.DataFrame(
        {
            "value": [0.001, -0.0002],
            "momentum": [0.0004, 0.0005],
            "sector_total": [0.0, 0.0001],
            "specific": [0.0002, -0.0001],
            "alpha_total": [0.0014, 0.0003],
            "model_active_return": [0.0016, 0.0003],
        },
        index=index,
    )
    result = FactorAttributionResult(
        monthly_contribution=monthly,
        cumulative_contribution=monthly[["value", "momentum", "sector_total", "specific"]].cumsum(),
        yearly_contribution=monthly[["value", "momentum", "sector_total", "specific"]].groupby(index.year).sum(),
        factor_summary_bps=pd.DataFrame({"total_bp": [8.0]}, index=["value"]),
        active_factor_exposure=pd.DataFrame({"value": [0.1, 0.2]}, index=index),
        realized_factor_return=pd.DataFrame({"value": [0.01, -0.001]}, index=index),
        reconciliation=pd.DataFrame({"actual_active_return": [0.0016, 0.0003]}, index=index),
    )

    payload = write_factor_attribution(result, tmp_path / "factor_attribution")

    for key in (
        "excel",
        "cumulative_factor_contribution_png",
        "monthly_factor_heatmap_png",
        "yearly_factor_contribution_png",
    ):
        assert Path(payload[key]).exists()


def test_sector_relative_retail_flow_demeans_within_sector_and_inverts() -> None:
    monthly_flow = pd.DataFrame(
        {"A": [10.0], "B": [30.0], "C": [100.0], "D": [140.0]},
        index=pd.to_datetime(["2024-01-31"]),
    )
    monthly_sector = pd.DataFrame(
        {"A": ["Tech"], "B": ["Tech"], "C": ["Bank"], "D": ["Bank"]},
        index=monthly_flow.index,
    )

    result = _sector_relative_retail_flow(monthly_flow, monthly_sector)

    assert result.loc["2024-01-31", "A"] == pytest.approx(10.0)
    assert result.loc["2024-01-31", "B"] == pytest.approx(-10.0)
    assert result.loc["2024-01-31", "C"] == pytest.approx(20.0)
    assert result.loc["2024-01-31", "D"] == pytest.approx(-20.0)


def test_performance_metrics_reports_cumulative_and_drawdown_stats() -> None:
    returns = pd.Series(
        [0.10, -0.05, 0.02],
        index=pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]),
    )

    metrics = performance_metrics(returns, periods_per_year=12)

    expected_total = (1.10 * 0.95 * 1.02) - 1.0
    expected_mdd = ((1.10 * 0.95) / 1.10) - 1.0
    assert metrics["total_return_pct"] == pytest.approx(expected_total * 100.0)
    assert metrics["max_drawdown_pct"] == pytest.approx(expected_mdd * 100.0)
    assert metrics["best_month_pct"] == pytest.approx(10.0)
    assert metrics["worst_month_pct"] == pytest.approx(-5.0)
