from pathlib import Path

import numpy as np
import pytest
import pandas as pd

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
    build_emp008_config,
    latest_common_end,
    write_target_weights_csv,
)
from backtesting.strategies.emp008.comparison import (
    _pair_active_weight_display_frame,
    active_weight_abs_sum_frame,
    build_emp008_comparison,
    excess_summary_bps,
    monthly_compounded_returns,
    monthly_excess_heatmap_frame,
    pair_return_display_frame,
    performance_metrics,
    yearly_compounded_returns,
)
from backtesting.strategies.emp008.attribution import FactorAttributionResult, factor_attribution_row, write_factor_attribution
from backtesting.strategies.emp008.mfbt_emp008_factors import _sector_relative_retail_flow, build_raw_mfbt_factors
from backtesting.strategies.emp008.mfbt_emp008 import (
    _apply_expected_alpha_policy,
    _has_sufficient_risk_history,
    _stock_excess_covariance_for_target_universe,
)
from backtesting.strategies.emp008.mfbt_emp008_data import (
    MfbtEmp008Config,
    _trim_non_forward_snapshot_frames,
    load_mfbt_emp008_market,
    required_datasets,
)
from backtesting.strategies.emp008.mfbt_emp008_optimize import optimize_active_weights_with_covariance
from backtesting.strategies.emp008.mfbt_emp008_preprocess import preprocess_factor_frame


def test_latest_common_end_uses_required_dataset_minimum_end_date(tmp_path: Path) -> None:
    catalog = DataCatalog.default()
    config = MfbtEmp008Config()
    common_index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    shorter_index = pd.to_datetime(["2024-01-02"])

    for dataset_id in required_datasets(config):
        spec = catalog.get(dataset_id)
        index = shorter_index if dataset_id is config.sector_dataset else common_index
        pd.DataFrame({"A": range(len(index))}, index=index).to_parquet(tmp_path / f"{spec.stem}.parquet")

    assert latest_common_end(tmp_path, config) == "2024-01-02"


def test_required_datasets_includes_distinct_sector_neutral_dataset() -> None:
    config = MfbtEmp008Config(sector_neutral_dataset=DatasetId.QW_WICS_SEC_BIG)

    datasets = required_datasets(config)

    assert DatasetId.QW_WI_SEC_26_BIG in datasets
    assert DatasetId.QW_WICS_SEC_BIG in datasets
    assert len(datasets) == len(set(datasets))


def test_load_market_keeps_retail_and_sector_neutral_frames_separate(tmp_path: Path) -> None:
    catalog = DataCatalog.default()
    config = MfbtEmp008Config(sector_neutral_dataset=DatasetId.QW_WICS_SEC_BIG)
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

    market = load_mfbt_emp008_market(
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

    assert config.factor_set == "origin"
    assert config.expected_alpha_policy == "origin_sign"
    assert config.rank_transform_factors == ("LnMktcap",)
    assert config.large_bm_neutral_factor_names == ()
    assert config.monthly_snapshot_forward_days == 7
    assert config.tracking_error == pytest.approx(0.007 / (12**0.5))
    assert DatasetId.QW_DIVIDEND_YLD_FY0 in required_datasets(config)


def test_build_emp008_config_sets_wics_sector_neutral_dataset() -> None:
    config = build_emp008_config(tracking_error_annual=0.007, sector_neutral_dataset="wics")

    assert config.sector_dataset == DatasetId.QW_WI_SEC_26_BIG
    assert config.sector_neutral_dataset == DatasetId.QW_WICS_SEC_BIG
    assert DatasetId.QW_WI_SEC_26_BIG in required_datasets(config)
    assert DatasetId.QW_WICS_SEC_BIG in required_datasets(config)


def test_build_emp008_config_rejects_unknown_factor_set() -> None:
    with pytest.raises(ValueError, match="factor_set"):
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

    factors = build_raw_mfbt_factors(market, MfbtEmp008Config(factor_set="origin"))

    assert list(factors) == ["LnMktcap", "Momentum_12M", "DY"]
    assert factors["LnMktcap"].loc["2024-02-29", "A"] == pytest.approx(np.log(1500.0))
    assert factors["Momentum_12M"].loc["2024-01-31", "A"] == pytest.approx(0.20)
    assert factors["Momentum_12M"].loc["2024-02-29", "A"] == pytest.approx(99.0 / 101.0 - 1.0)
    assert factors["DY"].loc["2024-02-29", "A"] == pytest.approx(0.65)


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

    factors = build_raw_mfbt_factors(market, MfbtEmp008Config(factor_set="origin"))

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
        config=MfbtEmp008Config(factor_set="origin", monthly_snapshot_forward_days=7),
    )

    assert result.frames["close"].index.max() == pd.Timestamp("2026-05-28")
    assert result.frames["dividend_yld_fy0"].index.max() == pd.Timestamp("2026-05-29")


def test_origin_expected_alpha_policy_matches_w_emp008_sign_rules() -> None:
    expected_alpha = pd.Series(
        {
            "LnMktcap": 0.01,
            "Momentum_12M": -0.02,
            "DY": 0.03,
            "sector_tech": 0.0,
        }
    )

    result = _apply_expected_alpha_policy(
        expected_alpha,
        MfbtEmp008Config(factor_set="origin", expected_alpha_policy="origin_sign"),
    )

    assert result["LnMktcap"] == 0.0
    assert result["Momentum_12M"] == 0.0
    assert result["DY"] == pytest.approx(0.03)
    assert result["sector_tech"] == 0.0


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
    config = MfbtEmp008Config(risk_window=36)
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


def test_yearly_compounded_returns_uses_calendar_year_compounding() -> None:
    returns = pd.DataFrame(
        {"MFBT": [0.01, 0.02, -0.01]},
        index=pd.to_datetime(["2024-01-02", "2024-12-31", "2025-01-02"]),
    )

    result = yearly_compounded_returns(returns)

    assert result.index.tolist() == [2024, 2025]
    assert result["MFBT"].tolist() == pytest.approx([(1.01 * 1.02) - 1.0, -0.01])


def test_pair_return_display_frame_keeps_only_gross_strategy_and_benchmark_lines() -> None:
    returns = pd.DataFrame(
        {
            "mfbt_emp008_70bp_36m gross": [0.01],
            "mfbt_emp008_70bp_36m costed": [0.02],
            "origin_emp008 gross": [0.03],
            "origin_emp008 costed": [0.04],
            "KOSPI200 BM": [0.05],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )

    result = pair_return_display_frame(returns)

    assert result.columns.tolist() == ["MFBT", "Origin", "KOSPI200 BM"]
    assert result.iloc[0].tolist() == pytest.approx([0.01, 0.03, 0.05])


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


def test_pair_active_weight_display_frame_accepts_single_strategy_schema() -> None:
    frame = pd.DataFrame(
        {"sum_abs_active_weight_pct": [14.0, 0.5]},
        index=pd.to_datetime(["2023-01-31", "2023-02-28"]),
    )

    result = _pair_active_weight_display_frame(frame)

    assert result.columns.tolist() == ["MFBT"]
    assert result.index.tolist() == [pd.Timestamp("2023-01-31")]
    assert result.loc[pd.Timestamp("2023-01-31"), "MFBT"] == pytest.approx(14.0)


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
