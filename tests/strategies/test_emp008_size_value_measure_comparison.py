from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import backtesting.strategies.emp008.experiments.size_value_measure_comparison as comparison
from backtesting.strategies.emp008.data import Emp008Config
from backtesting.strategies.emp008.experiments.size_value_measure_comparison import (
    DEFAULT_FEE,
    DEFAULT_FLOW_OUTPUT_DIR,
    DEFAULT_MOMENTUM_OUTPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SELL_TAX,
    DEFAULT_SLIPPAGE,
    DEFAULT_VARIANTS,
    FLOW_VARIANTS,
    MOMENTUM_VARIANTS,
    VariantResult,
    _parser,
    _resolve_shared_end,
    build_comparison_tables,
    run_portfolio_variant,
    run_size_value_measure_comparison,
    validate_variants,
    write_comparison_outputs,
)
from backtesting.strategies.emp008.factor_registry import FactorSetId, get_factor_set_definition


EXPECTED_COSTS = {
    "fee": DEFAULT_FEE,
    "sell_tax": DEFAULT_SELL_TAX,
    "slippage": DEFAULT_SLIPPAGE,
}


def test_validate_variants_rejects_duplicates() -> None:
    with pytest.raises(
        ValueError,
        match=r"duplicate variants: \('size_value_fcf_tev',\)",
    ):
        validate_variants(("size_value_fcf_tev", "size_value_fcf_tev"))


def test_validate_variants_rejects_unknown_names() -> None:
    with pytest.raises(
        ValueError,
        match=r"unknown variants: \('baseline',\)",
    ):
        validate_variants(("size_value_fcf_tev", "baseline"))


def test_validate_variants_rejects_empty_selection() -> None:
    with pytest.raises(ValueError, match="variants must not be empty"):
        validate_variants(())


def test_validate_variants_returns_factor_set_ids() -> None:
    variants = validate_variants(DEFAULT_VARIANTS)

    assert DEFAULT_OUTPUT_DIR == Path("backtesting/strategies/emp008/tests/size_value_measure_comparison")
    assert variants == (
        FactorSetId.SIZE_ONLY,
        FactorSetId.SIZE_VALUE_FCF_TEV,
        FactorSetId.SIZE_VALUE_DIVIDEND_FY0,
        FactorSetId.SIZE_VALUE_DIVIDEND_TTM,
    )
    assert DEFAULT_MOMENTUM_OUTPUT_DIR == Path(
        "backtesting/strategies/emp008/tests/size_momentum_measure_comparison"
    )
    assert validate_variants(MOMENTUM_VARIANTS) == (
        FactorSetId.SIZE_ONLY,
        FactorSetId.SIZE_MOMENTUM_12M,
        FactorSetId.SIZE_MOMENTUM_12_1M,
        FactorSetId.SIZE_MOMENTUM_HIGH,
        FactorSetId.SIZE_EARNINGS_MOMENTUM,
    )
    assert comparison._DISPLAY_LABELS["size_earnings_momentum"] == "Size + OP Consensus Momentum"
    assert comparison._YEARLY_DISPLAY_LABELS["size_earnings_momentum"] == "사이즈 + 영업이익 컨센서스"
    assert comparison._VARIANT_COLORS["size_earnings_momentum"] == "#DB2777"
    assert DEFAULT_FLOW_OUTPUT_DIR == Path(
        "backtesting/strategies/emp008/tests/size_flow_measure_comparison"
    )
    assert validate_variants(FLOW_VARIANTS) == (
        FactorSetId.SIZE_ONLY,
        FactorSetId.SIZE_RETAIL_FLOW,
    )
    assert comparison._DISPLAY_LABELS["size_retail_flow"] == "Size + Retail Flow"
    assert comparison._YEARLY_DISPLAY_LABELS["size_retail_flow"] == "사이즈 + 개인수급"
    assert comparison._VARIANT_COLORS["size_retail_flow"] == "#0F766E"


def test_build_comparison_tables_aligns_common_dates_and_ranks_summary() -> None:
    benchmark_returns = pd.Series(
        [0.0000, 0.0010, 0.0010, 0.0000],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    returns_by_variant = {
        "size_value_fcf_tev": pd.Series(
            [0.0020, 0.0030, 0.0010],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        ),
        "size_value_dividend_fy0": pd.Series(
            [0.0000, 0.0020, 0.0025],
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        ),
        "size_value_dividend_ttm": pd.Series(
            [0.0010, 0.0015, 0.0010],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        ),
    }
    active_share_by_variant = {
        "size_value_fcf_tev": {"mean_pct": 1.0},
        "size_value_dividend_fy0": {"mean_pct": 3.0},
        "size_value_dividend_ttm": {"mean_pct": 2.0},
    }

    summary, daily = build_comparison_tables(
        returns_by_variant=returns_by_variant,
        benchmark_returns=benchmark_returns,
        active_share_by_variant=active_share_by_variant,
    )

    assert daily.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert daily.columns.tolist() == [
        "benchmark_return",
        "size_value_fcf_tev_return",
        "size_value_fcf_tev_excess_return",
        "size_value_dividend_fy0_return",
        "size_value_dividend_fy0_excess_return",
        "size_value_dividend_ttm_return",
        "size_value_dividend_ttm_excess_return",
    ]
    assert daily["size_value_fcf_tev_excess_return"].tolist() == pytest.approx([0.0010, 0.0020])
    assert summary["variant"].tolist() == [
        "size_value_fcf_tev",
        "size_value_dividend_fy0",
        "size_value_dividend_ttm",
    ]
    assert summary["mean_active_share_pct"].tolist() == pytest.approx([1.0, 3.0, 2.0])
    assert {"annualized_excess_bp", "annualized_tracking_error_bp", "information_ratio"} <= set(summary.columns)


def test_build_comparison_tables_drops_nan_dates_and_keeps_finite_metrics() -> None:
    benchmark_returns = pd.Series(
        [0.0000, 0.0010, float("nan"), 0.0000, 0.0005],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )
    returns_by_variant = {
        "size_value_fcf_tev": pd.Series(
            [0.0020, float("nan"), 0.0010, 0.0015],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        ),
        "size_value_dividend_fy0": pd.Series(
            [0.0010, 0.0020, 0.0015, 0.0005],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        ),
        "size_value_dividend_ttm": pd.Series(
            [0.0005, 0.0015, 0.0010, 0.0010],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        ),
    }
    active_share_by_variant = {
        "size_value_fcf_tev": {"mean_pct": 1.0},
        "size_value_dividend_fy0": {"mean_pct": 3.0},
        "size_value_dividend_ttm": {"mean_pct": 2.0},
    }

    summary, daily = build_comparison_tables(
        returns_by_variant=returns_by_variant,
        benchmark_returns=benchmark_returns,
        active_share_by_variant=active_share_by_variant,
    )

    assert daily.index.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
    ]
    assert daily.notna().all().all()
    numeric_summary = summary.select_dtypes(include="number")
    assert numeric_summary.notna().all().all()
    assert numeric_summary.apply(lambda column: pd.Series(pd.to_numeric(column, errors="coerce")).map(float).map(pd.notna).all()).all()


def test_build_comparison_tables_rejects_empty_common_dates() -> None:
    with pytest.raises(ValueError, match="no common return dates"):
        build_comparison_tables(
            returns_by_variant={
                "size_value_fcf_tev": pd.Series([0.01], index=pd.to_datetime(["2024-01-02"])),
                "size_value_dividend_fy0": pd.Series([0.01], index=pd.to_datetime(["2024-01-03"])),
                "size_value_dividend_ttm": pd.Series([0.01], index=pd.to_datetime(["2024-01-04"])),
            },
            benchmark_returns=pd.Series([0.0], index=pd.to_datetime(["2024-01-01"])),
            active_share_by_variant={
                "size_value_fcf_tev": {"mean_pct": 1.0},
                "size_value_dividend_fy0": {"mean_pct": 2.0},
                "size_value_dividend_ttm": {"mean_pct": 3.0},
            },
        )


def test_build_comparison_tables_rejects_missing_return_mapping() -> None:
    with pytest.raises(ValueError, match="missing return mappings"):
        build_comparison_tables(
            returns_by_variant={
                "size_value_fcf_tev": pd.Series([0.01], index=pd.to_datetime(["2024-01-02"])),
                "size_value_dividend_fy0": pd.Series([0.01], index=pd.to_datetime(["2024-01-02"])),
            },
            benchmark_returns=pd.Series([0.0], index=pd.to_datetime(["2024-01-02"])),
            active_share_by_variant={
                "size_value_fcf_tev": {"mean_pct": 1.0},
                "size_value_dividend_fy0": {"mean_pct": 2.0},
                "size_value_dividend_ttm": {"mean_pct": 3.0},
            },
        )


def test_build_comparison_tables_rejects_missing_active_share_mapping() -> None:
    with pytest.raises(ValueError, match="missing return mappings"):
        build_comparison_tables(
            returns_by_variant={
                "size_value_fcf_tev": pd.Series([0.01], index=pd.to_datetime(["2024-01-02"])),
                "size_value_dividend_fy0": pd.Series([0.01], index=pd.to_datetime(["2024-01-02"])),
                "size_value_dividend_ttm": pd.Series([0.01], index=pd.to_datetime(["2024-01-02"])),
            },
            benchmark_returns=pd.Series([0.0], index=pd.to_datetime(["2024-01-02"])),
            active_share_by_variant={
                "size_value_fcf_tev": {"mean_pct": 1.0},
                "size_value_dividend_fy0": {"mean_pct": 2.0},
            },
        )


def test_build_comparison_tables_supports_subset_variant_order() -> None:
    selected = ("size_value_dividend_ttm", "size_value_fcf_tev")
    benchmark_returns = pd.Series(
        [0.0000, 0.0010, 0.0005],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    returns_by_variant = {
        "size_value_dividend_ttm": pd.Series(
            [0.0020, 0.0015, 0.0010],
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        ),
        "size_value_fcf_tev": pd.Series(
            [0.0010, 0.0005, 0.0005],
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        ),
    }
    active_share_by_variant = {
        "size_value_dividend_ttm": {"mean_pct": 2.0},
        "size_value_fcf_tev": {"mean_pct": 1.0},
    }

    summary, daily = build_comparison_tables(
        returns_by_variant=returns_by_variant,
        benchmark_returns=benchmark_returns,
        active_share_by_variant=active_share_by_variant,
    )

    assert daily.columns.tolist() == [
        "benchmark_return",
        "size_value_dividend_ttm_return",
        "size_value_dividend_ttm_excess_return",
        "size_value_fcf_tev_return",
        "size_value_fcf_tev_excess_return",
    ]
    assert set(summary["variant"]) == set(selected)


def test_write_comparison_outputs_writes_expected_files_and_interpretation(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "variant": "size_value_fcf_tev",
                "cagr_pct": 12.0,
                "annualized_excess_bp": 150.0,
                "annualized_tracking_error_bp": 50.0,
                "information_ratio": 3.0,
                "mean_active_share_pct": 1.0,
            },
            {
                "variant": "size_value_dividend_fy0",
                "cagr_pct": 10.0,
                "annualized_excess_bp": 120.0,
                "annualized_tracking_error_bp": 60.0,
                "information_ratio": 2.0,
                "mean_active_share_pct": 3.0,
            },
            {
                "variant": "size_value_dividend_ttm",
                "cagr_pct": 9.0,
                "annualized_excess_bp": 90.0,
                "annualized_tracking_error_bp": 45.0,
                "information_ratio": 2.0,
                "mean_active_share_pct": 2.0,
            },
        ]
    )
    daily = pd.DataFrame(
        {
            "benchmark_return": [0.0000, 0.0010, -0.0010],
            "size_value_fcf_tev_return": [0.0010, 0.0030, 0.0000],
            "size_value_fcf_tev_excess_return": [0.0010, 0.0020, 0.0010],
            "size_value_dividend_fy0_return": [0.0010, 0.0020, -0.0002],
            "size_value_dividend_fy0_excess_return": [0.0010, 0.0010, 0.0008],
            "size_value_dividend_ttm_return": [0.0005, 0.0015, -0.0005],
            "size_value_dividend_ttm_excess_return": [0.0005, 0.0005, 0.0005],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    before_fignums = set(plt.get_fignums())
    output_one = tmp_path / "first"
    output_two = tmp_path / "second"
    payload = write_comparison_outputs(
        output_dir=output_one,
        summary=summary,
        daily=daily,
        manifest={"benchmark": "IKS200", "periods_per_year": 252},
        shared_assumptions=(
            "All variants use the same benchmark and daily return alignment.",
            "The comparison is descriptive only.",
        ),
    )
    after_first_fignums = set(plt.get_fignums())
    second_payload = write_comparison_outputs(
        output_dir=output_two,
        summary=summary,
        daily=daily,
        manifest={"benchmark": "IKS200", "periods_per_year": 252},
        shared_assumptions=(
            "All variants use the same benchmark and daily return alignment.",
            "The comparison is descriptive only.",
        ),
    )
    after_second_fignums = set(plt.get_fignums())

    expected_files = {
        "performance_summary_csv": output_one / "performance_summary.csv",
        "performance_summary_xlsx": output_one / "performance_summary.xlsx",
        "performance_table_ko_csv": output_one / "performance_table_ko.csv",
        "daily_returns_csv": output_one / "daily_returns.csv",
        "performance_dashboard_png": output_one / "performance_dashboard.png",
        "cumulative_returns_png": output_one / "cumulative_returns.png",
        "cumulative_excess_returns_png": output_one / "cumulative_excess_returns.png",
        "yearly_excess_returns_png": output_one / "yearly_excess_returns.png",
        "yearly_returns_csv": output_one / "yearly_returns_pct.csv",
        "yearly_excess_returns_csv": output_one / "yearly_excess_returns_bp.csv",
        "yearly_performance_xlsx": output_one / "yearly_performance.xlsx",
        "interpretation_md": output_one / "interpretation.md",
    }
    expected_files_second = {
        key: str(output_two / Path(path).name) for key, path in expected_files.items()
    }
    assert payload == {key: str(path) for key, path in expected_files.items()}
    assert second_payload == expected_files_second
    assert before_fignums == after_first_fignums == after_second_fignums
    for path in expected_files.values():
        assert path.exists()
    assert (output_one / "cumulative_returns.png").stat().st_size > 0
    assert (output_one / "cumulative_excess_returns.png").stat().st_size > 0
    assert (output_one / "yearly_excess_returns.png").stat().st_size > 0
    assert (output_one / "performance_dashboard.png").stat().st_size > 0
    assert (output_one / "performance_dashboard.png").read_bytes() == (
        output_two / "performance_dashboard.png"
    ).read_bytes()
    assert (output_one / "cumulative_returns.png").read_bytes() == (output_two / "cumulative_returns.png").read_bytes()
    assert (output_one / "cumulative_excess_returns.png").read_bytes() == (
        output_two / "cumulative_excess_returns.png"
    ).read_bytes()
    assert pd.read_csv(output_one / "performance_summary.csv").equals(pd.read_csv(output_two / "performance_summary.csv"))
    assert pd.read_csv(output_one / "daily_returns.csv").equals(pd.read_csv(output_two / "daily_returns.csv"))

    markdown = (output_one / "interpretation.md").read_text(encoding="utf-8")
    assert "All variants use the same benchmark and daily return alignment." in markdown
    assert "size_value_fcf_tev" in markdown
    assert "size_value_dividend_fy0" in markdown
    assert "size_value_dividend_ttm" in markdown
    assert "Top annualized excess variant: `size_value_fcf_tev`" in markdown
    assert "Top information ratio variant: `size_value_fcf_tev`" in markdown
    assert "does not claim statistical significance" in markdown


def test_write_comparison_outputs_supports_subset_variants(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "variant": "size_value_dividend_ttm",
                "cagr_pct": 12.0,
                "annualized_excess_bp": 150.0,
                "annualized_tracking_error_bp": 50.0,
                "information_ratio": 3.0,
                "mean_active_share_pct": 2.0,
            },
            {
                "variant": "size_value_fcf_tev",
                "cagr_pct": 10.0,
                "annualized_excess_bp": 100.0,
                "annualized_tracking_error_bp": 40.0,
                "information_ratio": 2.5,
                "mean_active_share_pct": 1.0,
            },
        ]
    )
    daily = pd.DataFrame(
        {
            "benchmark_return": [0.0000, 0.0010],
            "size_value_dividend_ttm_return": [0.0020, 0.0010],
            "size_value_dividend_ttm_excess_return": [0.0020, 0.0000],
            "size_value_fcf_tev_return": [0.0010, 0.0005],
            "size_value_fcf_tev_excess_return": [0.0010, -0.0005],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    payload = write_comparison_outputs(output_dir=tmp_path, summary=summary, daily=daily)

    markdown = Path(payload["interpretation_md"]).read_text(encoding="utf-8")
    assert "size_value_dividend_fy0" not in markdown
    assert "size_value_dividend_ttm, size_value_fcf_tev" in markdown
    assert Path(payload["cumulative_returns_png"]).exists()
    assert Path(payload["cumulative_excess_returns_png"]).exists()
    assert Path(payload["yearly_excess_returns_png"]).exists()
    assert Path(payload["performance_dashboard_png"]).exists()


def test_write_comparison_outputs_includes_historical_variants_in_cumulative_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = pd.DataFrame(
        [
            {
                "variant": "size_only",
                "cagr_pct": 10.0,
                "annualized_excess_bp": 100.0,
                "annualized_tracking_error_bp": 40.0,
                "information_ratio": 2.5,
                "mean_active_share_pct": 1.0,
            }
        ]
    )
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    daily = pd.DataFrame(
        {
            "benchmark_return": [0.005, 0.006],
            "size_only_return": [0.01, 0.02],
            "size_only_excess_return": [0.005, 0.014],
        },
        index=dates,
    )
    annual_input = pd.DataFrame(
        {
            "size_only": [0.01, 0.02],
            "existing_emp008": [0.008, 0.018],
            "first_adjustment_emp008": [0.009, 0.019],
            "second_adjustment_emp008": [0.007, 0.017],
        },
        index=dates,
    )
    captured: dict[str, tuple[str, ...]] = {}

    def capture_returns(*, daily: pd.DataFrame, variants: tuple[str, ...], path: Path) -> None:
        captured["returns"] = variants
        assert all(f"{variant}_return" in daily for variant in variants)
        path.touch()

    def capture_excess(*, daily: pd.DataFrame, variants: tuple[str, ...], path: Path) -> None:
        captured["excess"] = variants
        assert all(f"{variant}_return" in daily for variant in variants)
        path.touch()

    monkeypatch.setattr(comparison, "_plot_cumulative_returns", capture_returns)
    monkeypatch.setattr(comparison, "_plot_cumulative_excess_returns", capture_excess)

    write_comparison_outputs(
        output_dir=tmp_path,
        summary=summary,
        daily=daily,
        yearly_portfolio_returns=annual_input,
    )

    expected = tuple(annual_input.columns)
    assert captured == {"returns": expected, "excess": expected}


def test_reference_style_plot_frames_start_from_common_base() -> None:
    variant = "size_value_fcf_tev"
    daily = pd.DataFrame(
        {
            "benchmark_return": [0.10, 0.00],
            f"{variant}_return": [0.00, 0.10],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    wealth = comparison._normalized_wealth_frame(daily, (variant,))
    excess = comparison._cumulative_excess_bp_frame(wealth, (variant,))

    assert wealth.iloc[0].to_dict() == {"benchmark": 100.0, variant: 100.0}
    assert excess.iloc[0, 0] == pytest.approx(0.0)
    assert excess.iloc[-1, 0] == pytest.approx(1_000.0)


def test_yearly_cumulative_excess_frames_reset_every_year_to_zero() -> None:
    dates = pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"])
    portfolio_returns = pd.DataFrame(
        {
            "size_only": [0.01, 0.02, -0.01, 0.03],
            "existing_emp008": [0.02, 0.01, 0.00, 0.02],
        },
        index=dates,
    )
    benchmark_returns = pd.Series([0.005, 0.01, -0.005, 0.01], index=dates)

    yearly = comparison._yearly_cumulative_excess_bp_frames(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
    )

    assert tuple(yearly) == (2024, 2025)
    for frame in yearly.values():
        assert frame.iloc[0].tolist() == pytest.approx([0.0, 0.0])
    assert yearly[2025].index[0] == pd.Timestamp("2024-12-31")
    expected_2025_size = (((0.99 * 1.03) / (0.995 * 1.01)) - 1.0) * 10_000.0
    assert yearly[2025]["size_only"].iloc[-1] == pytest.approx(expected_2025_size)


def test_yearly_performance_tables_match_subplot_endpoints() -> None:
    dates = pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"])
    portfolio_returns = pd.DataFrame(
        {
            "size_only": [0.01, 0.02, -0.01, 0.03],
            "existing_emp008": [0.02, 0.01, 0.00, 0.02],
        },
        index=dates,
    )
    benchmark_returns = pd.Series([0.005, 0.01, -0.005, 0.01], index=dates)

    yearly_returns, yearly_excess = comparison._yearly_performance_tables(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
    )
    frames = comparison._yearly_cumulative_excess_bp_frames(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
    )

    assert yearly_returns.index.tolist() == [2024, 2025]
    assert yearly_returns.loc[2025, "사이즈 단독(Base)"] == pytest.approx((0.99 * 1.03 - 1.0) * 100.0)
    assert yearly_returns.loc[2025, "KOSPI200 BM"] == pytest.approx((0.995 * 1.01 - 1.0) * 100.0)
    assert yearly_excess["KOSPI200 BM"].eq(0.0).all()
    assert yearly_excess.loc[2025, "사이즈 단독(Base)"] == pytest.approx(frames[2025]["size_only"].iloc[-1])


def test_yearly_outputs_drop_prior_year_zero_baseline_but_keep_first_2020_return() -> None:
    dates = pd.to_datetime(["2019-12-30", "2020-01-02", "2020-01-03"])
    portfolio_returns = pd.DataFrame({"size_only": [0.0, -0.01, 0.03]}, index=dates)
    benchmark_returns = pd.Series([0.0, -0.005, 0.01], index=dates)

    frames = comparison._yearly_cumulative_excess_bp_frames(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
    )
    yearly_returns, yearly_excess = comparison._yearly_performance_tables(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
    )

    assert tuple(frames) == (2020,)
    assert yearly_returns.index.tolist() == [2020]
    assert yearly_excess.index.tolist() == [2020]
    assert yearly_returns.loc[2020, "사이즈 단독(Base)"] == pytest.approx((0.99 * 1.03 - 1.0) * 100.0)
    assert frames[2020].iloc[0, 0] == pytest.approx(0.0)
    assert frames[2020].index[0] == pd.Timestamp("2020-01-01")


def test_performance_table_ko_uses_previous_report_format_and_includes_benchmark() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    portfolio_returns = pd.DataFrame(
        {
            "size_only": [0.01, -0.005, 0.02],
            "existing_emp008": [0.008, -0.004, 0.018],
        },
        index=dates,
    )
    benchmark_returns = pd.Series([0.005, -0.003, 0.01], index=dates)

    table = comparison._performance_table_ko(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
    )

    assert table.columns.tolist() == [
        "모델",
        "CAGR (%)",
        "연 변동성 (%)",
        "Sharpe",
        "MDD (%)",
        "누적수익률 (%)",
    ]
    assert table["모델"].tolist() == ["사이즈 단독(Base)", "기존 EMP008", "KOSPI200 BM"]


def test_load_historical_comparison_returns_validates_and_maps_columns(tmp_path: Path) -> None:
    path = tmp_path / "daily_net_returns.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "기존 EMP008": [0.01, 0.02],
            "1차수정 EMP008": [0.02, 0.01],
            "2차수정 EMP008": [0.00, 0.01],
            "KOSPI200 BM": [0.005, 0.006],
        }
    ).to_csv(path, index=False)

    portfolios, benchmark = comparison._load_historical_comparison_returns(path)

    assert portfolios.columns.tolist() == ["existing_emp008", "first_adjustment_emp008", "second_adjustment_emp008"]
    assert benchmark.name == "benchmark_return"
    assert benchmark.tolist() == pytest.approx([0.005, 0.006])

    pd.DataFrame({"date": ["2024-01-02"], "기존 EMP008": [0.01]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required historical comparison columns"):
        comparison._load_historical_comparison_returns(path)


def test_run_portfolio_variant_runs_weights_and_backtest_and_writes_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    variant_dir = tmp_path / "size_value_fcf_tev"
    runner_calls: list[dict[str, object]] = []
    config_calls: list[dict[str, object]] = []
    active_share_writes: list[Path] = []

    target_weights = pd.DataFrame({"A": [0.6], "B": [0.4]}, index=pd.to_datetime(["2024-01-31"]))
    active_weights = pd.DataFrame({"A": [0.1], "B": [-0.1]}, index=pd.to_datetime(["2024-01-31"]))

    def fake_build_emp008_config(**kwargs: object) -> Emp008Config:
        config_calls.append(kwargs)
        return Emp008Config(
            factor_set=kwargs["factor_set"],
            risk_model=str(kwargs["risk_model"]),
            tracking_error=float(kwargs["tracking_error_annual"]) / (12**0.5),
        )

    def fake_run_emp008(**kwargs: object) -> SimpleNamespace:
        output_dir = Path(str(kwargs["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        active_weights.to_parquet(output_dir / "active_weights.parquet")
        return SimpleNamespace(target_weights=target_weights)

    def fake_write_target_weights_csv(weights: pd.DataFrame, path: Path) -> Path:
        weights.to_csv(path)
        return path

    def fake_write_active_share(path: Path) -> dict[str, str]:
        active_share_writes.append(path)
        payload = pd.DataFrame(
            {"active_share": [0.1], "active_share_pct": [10.0]},
            index=pd.to_datetime(["2024-01-31"]),
        )
        payload.index.name = "date"
        payload.to_parquet(path.parent / "active_share.parquet")
        payload.reset_index().to_csv(path.parent / "active_share.csv", index=False)
        return {
            "active_share_parquet": str(path.parent / "active_share.parquet"),
            "active_share_csv": str(path.parent / "active_share.csv"),
        }

    def fake_active_share_summary(path: Path) -> dict[str, object]:
        assert path == variant_dir / "weights" / "active_share.parquet"
        return {"mean_pct": 10.0, "rows": 1}

    def fake_build_target_weight_spec(**kwargs: object) -> dict[str, object]:
        return kwargs

    class FakeRunner:
        def __init__(self, **kwargs: object) -> None:
            runner_calls.append({"init": kwargs})

        def resolve_spec(self, spec: object) -> object:
            runner_calls.append({"resolve_spec": spec})
            return spec

        def run_spec(self, spec: object) -> SimpleNamespace:
            runner_calls.append({"run_spec": spec})
            returns_dir = variant_dir / "backtests" / "resolved_run" / "series"
            returns_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "date": ["2024-02-29", "2024-01-31"],
                    "returns": [0.02, 0.01],
                }
            ).to_csv(returns_dir / "returns.csv", index=False)
            return SimpleNamespace(output_dir=returns_dir.parent)

    monkeypatch.setattr(comparison, "build_emp008_config", fake_build_emp008_config)
    monkeypatch.setattr(comparison, "run_emp008", fake_run_emp008)
    monkeypatch.setattr(comparison, "write_target_weights_csv", fake_write_target_weights_csv)
    monkeypatch.setattr(comparison, "write_active_share", fake_write_active_share)
    monkeypatch.setattr(comparison, "active_share_summary", fake_active_share_summary)
    monkeypatch.setattr(comparison, "build_target_weight_spec", fake_build_target_weight_spec)
    monkeypatch.setattr(comparison, "BacktestRunner", FakeRunner)

    result = run_portfolio_variant(
        factor_set=FactorSetId.SIZE_VALUE_FCF_TEV,
        parquet_dir=parquet_dir,
        variant_dir=variant_dir,
        start="2024-01-31",
        end="2024-02-29",
        tracking_error_annual=0.007,
        risk_model="factor_idio",
    )

    assert result.factor_set == FactorSetId.SIZE_VALUE_FCF_TEV
    pd.testing.assert_series_equal(
        result.returns,
        pd.Series(
            [0.01, 0.02],
            index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
            name="returns",
        ),
    )
    assert result.returns_csv == variant_dir / "backtests" / "resolved_run" / "series" / "returns.csv"
    assert result.weights_dir == variant_dir / "weights"
    assert result.active_share == {"mean_pct": 10.0, "rows": 1}
    assert config_calls == [
        {
            "tracking_error_annual": 0.007,
            "risk_model": "factor_idio",
            "factor_set": "size_value_fcf_tev",
        }
    ]
    assert active_share_writes == [variant_dir / "weights" / "active_weights.parquet"]
    assert runner_calls[0] == {
        "init": {
            "result_dir": variant_dir / "backtests",
            "write_report_assets": False,
            "profile": True,
        }
    }
    spec = runner_calls[1]["resolve_spec"]
    assert spec["name"] == "emp008_size_value_fcf_tev"
    assert spec["weights_csv"] == variant_dir / "weights" / "target_weights.csv"
    assert spec["dates"] == ("2024-01-31",)
    assert spec["end"] == "2024-02-29"
    assert spec["fill_mode"] == "close"
    assert spec["capital"] == 100_000_000.0
    assert spec["fee"] == DEFAULT_FEE
    assert spec["sell_tax"] == DEFAULT_SELL_TAX
    assert spec["slippage"] == DEFAULT_SLIPPAGE
    assert spec["allow_fractional"] is True
    metadata = json.loads((variant_dir / "backtest_metadata.json").read_text(encoding="utf-8"))
    assert metadata["factor_set"] == "size_value_fcf_tev"
    assert metadata["backtest_output_dir"] == str(variant_dir / "backtests" / "resolved_run")
    assert metadata["returns_csv"] == str(variant_dir / "backtests" / "resolved_run" / "series" / "returns.csv")
    assert metadata["config"]["factor_set"] == "size_value_fcf_tev"
    assert metadata["config"]["risk_model"] == "factor_idio"


def test_run_portfolio_variant_uses_complete_cache_and_reruns_partial_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    variant_dir = tmp_path / "size_value_dividend_fy0"
    weights_dir = variant_dir / "weights"
    weights_dir.mkdir(parents=True)
    pd.DataFrame({"A": [0.7]}, index=["2024-01-31"]).to_csv(weights_dir / "target_weights.csv")
    pd.DataFrame({"A": [0.1]}, index=pd.to_datetime(["2024-01-31"])).to_parquet(weights_dir / "active_weights.parquet")
    pd.DataFrame({"active_share": [0.1], "active_share_pct": [10.0]}, index=pd.to_datetime(["2024-01-31"])).to_parquet(
        weights_dir / "active_share.parquet"
    )

    returns_csv = variant_dir / "backtests" / "cached_run" / "series" / "returns.csv"
    returns_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": ["2024-01-31"], "returns": [0.01]}).to_csv(returns_csv, index=False)
    (variant_dir / "backtest_metadata.json").write_text(
        json.dumps(
            {
                "start": "2024-01-31",
                "end": "2024-02-29",
                "tracking_error_annual": 0.007,
                "risk_model": "factor_idio",
                "factor_set": "size_value_dividend_fy0",
                "fill_mode": "close",
                "costs": EXPECTED_COSTS,
                "capital": 100_000_000.0,
                "allow_fractional": True,
                "config": {"factor_set": "size_value_dividend_fy0"},
                "backtest_output_dir": str(returns_csv.parent.parent),
                "returns_csv": str(returns_csv),
            }
        ),
        encoding="utf-8",
    )

    run_emp008_calls: list[dict[str, object]] = []
    runner_runs: list[object] = []

    monkeypatch.setattr(comparison, "build_emp008_config", lambda **_: Emp008Config(factor_set="size_value_dividend_fy0"))
    monkeypatch.setattr(
        comparison,
        "run_emp008",
        lambda **kwargs: run_emp008_calls.append(kwargs) or SimpleNamespace(target_weights=pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-31"]))),
    )
    monkeypatch.setattr(comparison, "write_target_weights_csv", lambda weights, path: weights.to_csv(path) or path)
    monkeypatch.setattr(
        comparison,
        "write_active_share",
        lambda path: {
            "active_share_parquet": str(path.parent / "active_share.parquet"),
            "active_share_csv": str(path.parent / "active_share.csv"),
        },
    )
    monkeypatch.setattr(comparison, "active_share_summary", lambda _: {"mean_pct": 10.0})
    monkeypatch.setattr(comparison, "build_target_weight_spec", lambda **kwargs: kwargs)

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            pass

        def resolve_spec(self, spec: object) -> object:
            return spec

        def run_spec(self, spec: object) -> SimpleNamespace:
            runner_runs.append(spec)
            rerun_dir = variant_dir / "backtests" / "rerun" / "series"
            rerun_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"date": ["2024-02-29"], "returns": [0.02]}).to_csv(rerun_dir / "returns.csv", index=False)
            return SimpleNamespace(output_dir=rerun_dir.parent)

    monkeypatch.setattr(comparison, "BacktestRunner", FakeRunner)

    cached = run_portfolio_variant(
        factor_set=FactorSetId.SIZE_VALUE_DIVIDEND_FY0,
        parquet_dir=parquet_dir,
        variant_dir=variant_dir,
        start="2024-01-31",
        end="2024-02-29",
        tracking_error_annual=0.007,
        risk_model="factor_idio",
    )
    assert cached.returns_csv == returns_csv
    assert run_emp008_calls == []
    assert runner_runs == []

    returns_csv.unlink()
    rerun = run_portfolio_variant(
        factor_set=FactorSetId.SIZE_VALUE_DIVIDEND_FY0,
        parquet_dir=parquet_dir,
        variant_dir=variant_dir,
        start="2024-01-31",
        end="2024-02-29",
        tracking_error_annual=0.007,
        risk_model="factor_idio",
    )
    assert rerun.returns_csv == variant_dir / "backtests" / "rerun" / "series" / "returns.csv"
    assert len(runner_runs) == 1


@pytest.mark.parametrize(
    ("metadata_updates", "label", "expect_weights_rerun"),
    [
        ({"end": "2024-03-31"}, "cached end covers requested period", False),
        ({"tracking_error_annual": 0.009}, "changed tracking_error_annual", True),
        ({"risk_model": "direct_covariance"}, "changed risk_model", True),
    ],
)
def test_run_portfolio_variant_reruns_when_metadata_contract_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metadata_updates: dict[str, object],
    label: str,
    expect_weights_rerun: bool,
) -> None:
    del label
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    variant_dir = tmp_path / "size_value_dividend_fy0"
    weights_dir = variant_dir / "weights"
    weights_dir.mkdir(parents=True)
    pd.DataFrame({"A": [0.7]}, index=["2024-01-31"]).to_csv(weights_dir / "target_weights.csv")
    pd.DataFrame({"A": [0.1]}, index=pd.to_datetime(["2024-01-31"])).to_parquet(weights_dir / "active_weights.parquet")
    pd.DataFrame({"active_share": [0.1], "active_share_pct": [10.0]}, index=pd.to_datetime(["2024-01-31"])).to_parquet(
        weights_dir / "active_share.parquet"
    )

    returns_csv = variant_dir / "backtests" / "cached_run" / "series" / "returns.csv"
    returns_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": ["2024-01-31"], "returns": [0.01]}).to_csv(returns_csv, index=False)
    metadata = {
        "start": "2024-01-31",
        "end": "2024-02-29",
        "tracking_error_annual": 0.007,
        "risk_model": "factor_idio",
        "factor_set": "size_value_dividend_fy0",
        "fill_mode": "close",
        "costs": EXPECTED_COSTS,
        "capital": 100_000_000.0,
        "allow_fractional": True,
        "backtest_output_dir": str(returns_csv.parent.parent),
        "returns_csv": str(returns_csv),
    }
    metadata.update(metadata_updates)
    (variant_dir / "backtest_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    run_emp008_calls: list[dict[str, object]] = []
    runner_runs: list[object] = []

    monkeypatch.setattr(comparison, "build_emp008_config", lambda **_: Emp008Config(factor_set="size_value_dividend_fy0"))

    def fake_run_emp008(**kwargs: object) -> SimpleNamespace:
        run_emp008_calls.append(kwargs)
        output_dir = Path(str(kwargs["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"A": [0.1]}, index=pd.to_datetime(["2024-01-31"])).to_parquet(output_dir / "active_weights.parquet")
        return SimpleNamespace(target_weights=pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-31"])))

    monkeypatch.setattr(comparison, "run_emp008", fake_run_emp008)
    monkeypatch.setattr(comparison, "write_target_weights_csv", lambda weights, path: weights.to_csv(path) or path)
    monkeypatch.setattr(
        comparison,
        "write_active_share",
        lambda path: pd.DataFrame(
            {"active_share": [0.1], "active_share_pct": [10.0]},
            index=pd.to_datetime(["2024-01-31"]),
        ).rename_axis("date").to_parquet(path.parent / "active_share.parquet")
        or {
            "active_share_parquet": str(path.parent / "active_share.parquet"),
            "active_share_csv": str(path.parent / "active_share.csv"),
        },
    )
    monkeypatch.setattr(comparison, "active_share_summary", lambda _: {"mean_pct": 10.0})
    monkeypatch.setattr(comparison, "build_target_weight_spec", lambda **kwargs: kwargs)

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            pass

        def resolve_spec(self, spec: object) -> object:
            return spec

        def run_spec(self, spec: object) -> SimpleNamespace:
            runner_runs.append(spec)
            rerun_dir = variant_dir / "backtests" / "rerun" / "series"
            rerun_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"date": ["2024-02-29"], "returns": [0.02]}).to_csv(rerun_dir / "returns.csv", index=False)
            return SimpleNamespace(output_dir=rerun_dir.parent)

    monkeypatch.setattr(comparison, "BacktestRunner", FakeRunner)

    result = run_portfolio_variant(
        factor_set=FactorSetId.SIZE_VALUE_DIVIDEND_FY0,
        parquet_dir=parquet_dir,
        variant_dir=variant_dir,
        start="2024-01-31",
        end="2024-02-29",
        tracking_error_annual=0.007,
        risk_model="factor_idio",
    )

    assert len(run_emp008_calls) == int(expect_weights_rerun)
    assert len(runner_runs) == 1
    assert result.returns_csv == variant_dir / "backtests" / "rerun" / "series" / "returns.csv"


def test_run_portfolio_variant_reruns_backtest_when_weights_cache_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    variant_dir = tmp_path / "size_value_dividend_ttm"
    weights_dir = variant_dir / "weights"
    weights_dir.mkdir(parents=True)
    pd.DataFrame({"A": [0.7]}, index=["2024-01-31"]).to_csv(weights_dir / "target_weights.csv")
    pd.DataFrame({"A": [0.1]}, index=pd.to_datetime(["2024-01-31"])).to_parquet(weights_dir / "active_weights.parquet")

    stale_returns_csv = variant_dir / "backtests" / "cached_run" / "series" / "returns.csv"
    stale_returns_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": ["2024-01-31"], "returns": [0.01]}).to_csv(stale_returns_csv, index=False)
    (variant_dir / "backtest_metadata.json").write_text(
        json.dumps(
            {
                "factor_set": "size_value_dividend_ttm",
                "config": {"factor_set": "size_value_dividend_ttm"},
                "backtest_output_dir": str(stale_returns_csv.parent.parent),
                "returns_csv": str(stale_returns_csv),
            }
        ),
        encoding="utf-8",
    )

    run_emp008_calls: list[dict[str, object]] = []
    runner_runs: list[object] = []
    active_share_writes: list[Path] = []

    def fake_run_emp008(**kwargs: object) -> SimpleNamespace:
        run_emp008_calls.append(kwargs)
        output_dir = Path(str(kwargs["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"A": [0.2]}, index=pd.to_datetime(["2024-01-31"])).to_parquet(output_dir / "active_weights.parquet")
        return SimpleNamespace(target_weights=pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-31"])))

    def fake_write_active_share(path: Path) -> dict[str, str]:
        active_share_writes.append(path)
        frame = pd.DataFrame(
            {"active_share": [0.2], "active_share_pct": [20.0]},
            index=pd.to_datetime(["2024-01-31"]),
        )
        frame.index.name = "date"
        frame.to_parquet(path.parent / "active_share.parquet")
        frame.reset_index().to_csv(path.parent / "active_share.csv", index=False)
        return {
            "active_share_parquet": str(path.parent / "active_share.parquet"),
            "active_share_csv": str(path.parent / "active_share.csv"),
        }

    monkeypatch.setattr(comparison, "build_emp008_config", lambda **_: Emp008Config(factor_set="size_value_dividend_ttm"))
    monkeypatch.setattr(comparison, "run_emp008", fake_run_emp008)
    monkeypatch.setattr(comparison, "write_target_weights_csv", lambda weights, path: weights.to_csv(path) or path)
    monkeypatch.setattr(comparison, "write_active_share", fake_write_active_share)
    monkeypatch.setattr(comparison, "active_share_summary", lambda _: {"mean_pct": 20.0})
    monkeypatch.setattr(comparison, "build_target_weight_spec", lambda **kwargs: kwargs)

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            pass

        def resolve_spec(self, spec: object) -> object:
            return spec

        def run_spec(self, spec: object) -> SimpleNamespace:
            runner_runs.append(spec)
            rerun_dir = variant_dir / "backtests" / "rebuilt_run" / "series"
            rerun_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"date": ["2024-02-29"], "returns": [0.02]}).to_csv(rerun_dir / "returns.csv", index=False)
            return SimpleNamespace(output_dir=rerun_dir.parent)

    monkeypatch.setattr(comparison, "BacktestRunner", FakeRunner)

    result = run_portfolio_variant(
        factor_set=FactorSetId.SIZE_VALUE_DIVIDEND_TTM,
        parquet_dir=parquet_dir,
        variant_dir=variant_dir,
        start="2024-01-31",
        end="2024-02-29",
        tracking_error_annual=0.007,
        risk_model="factor_idio",
    )

    assert len(run_emp008_calls) == 1
    assert len(runner_runs) == 1
    assert active_share_writes == [weights_dir / "active_weights.parquet"]
    assert result.returns_csv == variant_dir / "backtests" / "rebuilt_run" / "series" / "returns.csv"
    assert result.returns.tolist() == pytest.approx([0.02])
    metadata = json.loads((variant_dir / "backtest_metadata.json").read_text(encoding="utf-8"))
    assert metadata["returns_csv"] == str(variant_dir / "backtests" / "rebuilt_run" / "series" / "returns.csv")


@pytest.mark.parametrize(
    ("metadata_contents", "expect_weights_rerun"),
    [
        ("{not json", True),
        (
            json.dumps(
                {
                    "start": "2024-01-31",
                    "end": "2024-02-29",
                    "tracking_error_annual": 0.007,
                    "risk_model": "factor_idio",
                    "factor_set": "size_value_dividend_ttm",
                    "fill_mode": "close",
                    "costs": EXPECTED_COSTS,
                    "capital": 100_000_000.0,
                    "allow_fractional": True,
                    "backtest_output_dir": "C:/outside",
                    "returns_csv": "C:/outside/returns.csv",
                }
            ),
            True,
        ),
    ],
)
def test_run_portfolio_variant_invalidates_malformed_or_out_of_root_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metadata_contents: str,
    expect_weights_rerun: bool,
) -> None:
    del expect_weights_rerun
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    variant_dir = tmp_path / "size_value_dividend_ttm"
    weights_dir = variant_dir / "weights"
    weights_dir.mkdir(parents=True)
    pd.DataFrame({"A": [0.7]}, index=["2024-01-31"]).to_csv(weights_dir / "target_weights.csv")
    pd.DataFrame({"A": [0.1]}, index=pd.to_datetime(["2024-01-31"])).to_parquet(weights_dir / "active_weights.parquet")
    pd.DataFrame({"active_share": [0.1], "active_share_pct": [10.0]}, index=pd.to_datetime(["2024-01-31"])).to_parquet(
        weights_dir / "active_share.parquet"
    )
    metadata_path = variant_dir / "backtest_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(metadata_contents, encoding="utf-8")

    run_emp008_calls: list[dict[str, object]] = []
    runner_runs: list[object] = []

    monkeypatch.setattr(comparison, "build_emp008_config", lambda **_: Emp008Config(factor_set="size_value_dividend_ttm"))

    def fake_run_emp008(**kwargs: object) -> SimpleNamespace:
        run_emp008_calls.append(kwargs)
        output_dir = Path(str(kwargs["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"A": [0.1]}, index=pd.to_datetime(["2024-01-31"])).to_parquet(output_dir / "active_weights.parquet")
        return SimpleNamespace(target_weights=pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-31"])))

    def fake_write_active_share(path: Path) -> dict[str, str]:
        frame = pd.DataFrame(
            {"active_share": [0.1], "active_share_pct": [10.0]},
            index=pd.to_datetime(["2024-01-31"]),
        )
        frame.index.name = "date"
        frame.to_parquet(path.parent / "active_share.parquet")
        frame.reset_index().to_csv(path.parent / "active_share.csv", index=False)
        return {
            "active_share_parquet": str(path.parent / "active_share.parquet"),
            "active_share_csv": str(path.parent / "active_share.csv"),
        }

    monkeypatch.setattr(comparison, "run_emp008", fake_run_emp008)
    monkeypatch.setattr(comparison, "write_target_weights_csv", lambda weights, path: weights.to_csv(path) or path)
    monkeypatch.setattr(comparison, "write_active_share", fake_write_active_share)
    monkeypatch.setattr(comparison, "active_share_summary", lambda _: {"mean_pct": 10.0})
    monkeypatch.setattr(comparison, "build_target_weight_spec", lambda **kwargs: kwargs)

    class FakeRunner:
        def __init__(self, **_: object) -> None:
            pass

        def resolve_spec(self, spec: object) -> object:
            return spec

        def run_spec(self, spec: object) -> SimpleNamespace:
            runner_runs.append(spec)
            rerun_dir = variant_dir / "backtests" / "rerun" / "series"
            rerun_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"date": ["2024-02-29"], "returns": [0.02]}).to_csv(rerun_dir / "returns.csv", index=False)
            return SimpleNamespace(output_dir=rerun_dir.parent)

    monkeypatch.setattr(comparison, "BacktestRunner", FakeRunner)

    run_portfolio_variant(
        factor_set=FactorSetId.SIZE_VALUE_DIVIDEND_TTM,
        parquet_dir=parquet_dir,
        variant_dir=variant_dir,
        start="2024-01-31",
        end="2024-02-29",
        tracking_error_annual=0.007,
        risk_model="factor_idio",
    )

    assert len(run_emp008_calls) == 1
    assert len(runner_runs) == 1


def test_run_size_value_measure_comparison_runs_selected_variants_and_writes_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    benchmark_path = parquet_dir / "qw_BM.parquet"
    benchmark_path.write_text("x", encoding="utf-8")
    selected = ("size_value_dividend_ttm", "size_value_fcf_tev")
    calls: list[dict[str, object]] = []
    benchmark_calls: list[tuple[Path, str, pd.Index | None]] = []

    def fake_run_portfolio_variant(**kwargs: object) -> VariantResult:
        calls.append(kwargs)
        factor_set = kwargs["factor_set"]
        assert isinstance(factor_set, FactorSetId)
        variant_dir = Path(str(kwargs["variant_dir"]))
        variant_dir.mkdir(parents=True, exist_ok=True)
        returns_csv = variant_dir / "returns.csv"
        pd.DataFrame({"date": ["2024-01-31", "2024-02-29"], "returns": [0.01, 0.02]}).to_csv(returns_csv, index=False)
        series = pd.Series([0.01, 0.02], index=pd.to_datetime(["2024-01-31", "2024-02-29"]), name="returns")
        return VariantResult(
            factor_set=factor_set,
            returns=series,
            returns_csv=returns_csv,
            weights_dir=variant_dir / "weights",
            active_share={"mean_pct": 12.5, "rows": 2},
        )

    monkeypatch.setattr(comparison, "run_portfolio_variant", fake_run_portfolio_variant)
    def fake_benchmark_returns(path: Path, code: str, comparison_index: pd.Index | None = None) -> pd.Series:
        benchmark_calls.append((path, code, comparison_index))
        return pd.Series([0.0, 0.001], index=pd.to_datetime(["2024-01-31", "2024-02-29"]))

    monkeypatch.setattr(comparison, "_benchmark_returns", fake_benchmark_returns)
    monkeypatch.setattr(
        comparison,
        "_load_historical_comparison_returns",
        lambda _: (
            pd.DataFrame(
                {
                    "existing_emp008": [0.0, 0.001],
                    "first_adjustment_emp008": [0.0, 0.001],
                    "second_adjustment_emp008": [0.0, 0.001],
                },
                index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
            ),
            pd.Series([0.0, 0.001], index=pd.to_datetime(["2024-01-31", "2024-02-29"]), name="benchmark_return"),
        ),
    )

    payload = run_size_value_measure_comparison(
        parquet_dir=parquet_dir,
        output_dir=tmp_path / "out",
        start="2024-01-31",
        end="2024-02-29",
        tracking_error_annual=0.007,
        risk_model="direct_covariance",
        variants=selected,
    )

    assert [call["factor_set"].value for call in calls] == list(selected)
    assert benchmark_calls[0][0:2] == (benchmark_path, "IKS200")
    assert benchmark_calls[0][2].equals(pd.to_datetime(["2024-01-31", "2024-02-29"]))
    for call in calls:
        assert call["parquet_dir"] == parquet_dir
        assert call["start"] == "2024-01-31"
        assert call["end"] == "2024-02-29"
        assert call["tracking_error_annual"] == 0.007
        assert call["risk_model"] == "direct_covariance"
        assert call["force"] is False
    manifest = json.loads(Path(payload["manifest_json"]).read_text(encoding="utf-8"))
    assert set(manifest) == {
        "start",
        "end",
        "tracking_error_annual",
        "risk_model",
        "fill_mode",
        "costs",
        "benchmark",
        "variants",
    }
    assert manifest["start"] == "2024-01-31"
    assert manifest["end"] == "2024-02-29"
    assert manifest["tracking_error_annual"] == 0.007
    assert manifest["risk_model"] == "direct_covariance"
    assert manifest["fill_mode"] == "close"
    assert manifest["costs"] == EXPECTED_COSTS
    assert manifest["benchmark"] == "IKS200"
    assert manifest["variants"] == [
        {
            "factor_set": "size_value_dividend_ttm",
            "factors": [factor.value for factor in get_factor_set_definition("size_value_dividend_ttm").factors],
            "datasets": ["qw_dps_ttm"],
        },
        {
            "factor_set": "size_value_fcf_tev",
            "factors": [factor.value for factor in get_factor_set_definition("size_value_fcf_tev").factors],
            "datasets": ["qw_fcf", "qw_int_bearing_liab_nfq0", "qw_quick_assets_nfq0"],
        },
    ]
    assert all(set(item) == {"factor_set", "factors", "datasets"} for item in manifest["variants"])
    assert payload["variants"] == list(selected)
    for key in (
        "performance_summary_csv",
        "performance_summary_xlsx",
        "performance_table_ko_csv",
        "daily_returns_csv",
        "performance_dashboard_png",
        "cumulative_returns_png",
        "cumulative_excess_returns_png",
        "yearly_excess_returns_png",
        "yearly_returns_csv",
        "yearly_excess_returns_csv",
        "yearly_performance_xlsx",
        "interpretation_md",
        "manifest_json",
    ):
        assert Path(payload[key]).exists()


def test_resolve_shared_end_uses_selected_variant_configs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_build_emp008_config(**kwargs: object) -> Emp008Config:
        return Emp008Config(factor_set=str(kwargs["factor_set"]), risk_model=str(kwargs["risk_model"]))

    def fake_latest_common_end(parquet_dir: Path, config: Emp008Config) -> str:
        del parquet_dir
        calls.append(config.factor_set.value)
        return {
            "size_value_fcf_tev": "2024-03-31",
            "size_value_dividend_ttm": "2024-02-29",
        }[config.factor_set.value]

    monkeypatch.setattr(comparison, "build_emp008_config", fake_build_emp008_config)
    monkeypatch.setattr(comparison, "latest_common_end", fake_latest_common_end)

    resolved = _resolve_shared_end(
        parquet_dir=tmp_path,
        factor_sets=(
            FactorSetId.SIZE_VALUE_FCF_TEV,
            FactorSetId.SIZE_VALUE_DIVIDEND_TTM,
        ),
        tracking_error_annual=0.007,
        risk_model="factor_idio",
    )

    assert resolved == "2024-02-29"
    assert calls == ["size_value_fcf_tev", "size_value_dividend_ttm"]


def test_parser_and_main_resolve_default_end_and_print_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    parsed = _parser().parse_args([])
    assert parsed.parquet_dir == Path("parquet")
    assert parsed.output_dir is None
    assert parsed.comparison_profile == "value"
    assert parsed.start == "2019-12-30"
    assert parsed.historical_returns_csv == Path(
        "results/emp008_reports/emp008_final_comparison_20260630/daily_net_returns.csv"
    )
    momentum = _parser().parse_args(["--comparison-profile", "momentum"])
    assert momentum.variants is None
    flow = _parser().parse_args(["--comparison-profile", "flow"])
    assert flow.variants is None
    with pytest.raises(SystemExit, match="2"):
        _parser().parse_args(["--variants", "baseline"])

    monkeypatch.setattr(comparison, "_resolve_shared_end", lambda **_: "2024-04-30")
    monkeypatch.setattr(
        comparison,
        "run_size_value_measure_comparison",
        lambda **kwargs: {
            "output_dir": str(kwargs["output_dir"]),
            "manifest_json": str(kwargs["output_dir"] / "manifest.json"),
            "variants": list(kwargs["variants"]),
            "start": kwargs["start"],
            "end": kwargs["end"],
        },
    )

    comparison.main(["--parquet-dir", str(tmp_path / "parquet"), "--output-dir", str(tmp_path / "out")])

    payload = json.loads(capsys.readouterr().out)
    assert payload["end"] == "2024-04-30"
    assert payload["variants"] == list(DEFAULT_VARIANTS)


def test_main_selects_flow_profile_and_reuses_size_only_cache(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "output_dir": str(kwargs["output_dir"]),
            "variants": list(kwargs["variants"]),
            "start": kwargs["start"],
            "end": kwargs["end"],
        }

    monkeypatch.setattr(comparison, "run_size_value_measure_comparison", fake_run)
    comparison.main(["--comparison-profile", "flow", "--end", "2026-06-30"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["output_dir"] == str(DEFAULT_FLOW_OUTPUT_DIR)
    assert payload["variants"] == list(FLOW_VARIANTS)
    assert captured["size_only_variant_dir"] == DEFAULT_OUTPUT_DIR / FactorSetId.SIZE_ONLY.value
