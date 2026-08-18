from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.axes import Axes

from backtesting.run import BacktestRunner
from backtesting.strategies.emp008.factor_registry import (
    FactorSetId,
    factor_definitions_for_set,
    get_factor_set_definition,
    parse_factor_set,
)
from backtesting.strategies.emp008.reports.comparison import _benchmark_returns, excess_summary_bps, performance_metrics
from backtesting.strategies.emp008.run_backtest import active_share_summary, build_target_weight_spec, write_active_share
from backtesting.strategies.emp008.run_weights import build_emp008_config, latest_common_end, write_target_weights_csv
from backtesting.strategies.emp008.strategy import run_emp008

DEFAULT_VARIANTS = (
    "size_only",
    "size_value_fcf_tev",
    "size_value_dividend_fy0",
    "size_value_dividend_ttm",
)
MOMENTUM_VARIANTS = (
    "size_only",
    "size_momentum_12m",
    "size_momentum_12_1m",
    "size_momentum_high",
    "size_earnings_momentum",
)
FLOW_VARIANTS = (
    "size_only",
    "size_retail_flow",
)
SUPPORTED_VARIANTS = tuple(dict.fromkeys((*DEFAULT_VARIANTS, *MOMENTUM_VARIANTS, *FLOW_VARIANTS)))
DEFAULT_OUTPUT_DIR = Path("backtesting/strategies/emp008/tests/size_value_measure_comparison")
DEFAULT_MOMENTUM_OUTPUT_DIR = Path("backtesting/strategies/emp008/tests/size_momentum_measure_comparison")
DEFAULT_FLOW_OUTPUT_DIR = Path("backtesting/strategies/emp008/tests/size_flow_measure_comparison")
DEFAULT_START = "2019-12-30"
DEFAULT_HISTORICAL_RETURNS_CSV = Path(
    "results/emp008_reports/emp008_final_comparison_20260630/daily_net_returns.csv"
)
DEFAULT_FEE = 0.0002
DEFAULT_SELL_TAX = 0.0015
DEFAULT_SLIPPAGE = 0.0005
_DISPLAY_LABELS = {
    "size_only": "Size-only Base",
    "size_value_fcf_tev": "Size + FCF/TEV",
    "size_value_dividend_fy0": "Size + Dividend FY0",
    "size_value_dividend_ttm": "Size + Dividend TTM",
    "size_momentum_12m": "Size + Momentum 12M",
    "size_momentum_12_1m": "Size + Momentum 12-1M",
    "size_momentum_high": "Size + 252D High",
    "size_earnings_momentum": "Size + OP Consensus Momentum",
    "size_retail_flow": "Size + Retail Flow",
}
_VARIANT_COLORS = {
    "size_only": "#7C3AED",
    "size_value_fcf_tev": "#0F766E",
    "size_value_dividend_fy0": "#D97706",
    "size_value_dividend_ttm": "#2563EB",
    "size_momentum_12m": "#0F766E",
    "size_momentum_12_1m": "#D97706",
    "size_momentum_high": "#2563EB",
    "size_earnings_momentum": "#DB2777",
    "size_retail_flow": "#0F766E",
}
_HISTORICAL_COLUMN_MAP = {
    "기존 EMP008": "existing_emp008",
    "1차수정 EMP008": "first_adjustment_emp008",
    "2차수정 EMP008": "second_adjustment_emp008",
}
_YEARLY_DISPLAY_LABELS = {
    "size_only": "사이즈 단독(Base)",
    "size_value_fcf_tev": "사이즈 + FCF/TEV",
    "size_value_dividend_fy0": "사이즈 + 배당 FY0",
    "size_value_dividend_ttm": "사이즈 + 배당 TTM",
    "size_momentum_12m": "사이즈 + 12개월 모멘텀",
    "size_momentum_12_1m": "사이즈 + 12-1개월 모멘텀",
    "size_momentum_high": "사이즈 + 252일 고가대비",
    "size_earnings_momentum": "사이즈 + 영업이익 컨센서스",
    "size_retail_flow": "사이즈 + 개인수급",
    "existing_emp008": "기존 EMP008",
    "first_adjustment_emp008": "1차수정 EMP008",
    "second_adjustment_emp008": "2차수정 EMP008",
}
_YEARLY_COLORS = {
    **_VARIANT_COLORS,
    "existing_emp008": "#475569",
    "first_adjustment_emp008": "#DC2626",
    "second_adjustment_emp008": "#0891B2",
}
_PLOT_DISPLAY_LABELS = {
    **_DISPLAY_LABELS,
    "existing_emp008": "기존 EMP008",
    "first_adjustment_emp008": "1차수정 EMP008",
    "second_adjustment_emp008": "2차수정 EMP008",
}


@dataclass(frozen=True, slots=True)
class VariantResult:
    factor_set: FactorSetId
    returns: pd.Series
    returns_csv: Path
    weights_dir: Path
    active_share: dict[str, object]


def validate_variants(variants: tuple[str, ...]) -> tuple[FactorSetId, ...]:
    if not variants:
        raise ValueError("variants must not be empty")
    if len(variants) != len(set(variants)):
        duplicates = tuple(name for name in dict.fromkeys(variants) if variants.count(name) > 1)
        raise ValueError(f"duplicate variants: {duplicates}")
    unknown = tuple(name for name in variants if name not in SUPPORTED_VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    return tuple(parse_factor_set(name) for name in variants)


def run_portfolio_variant(
    *,
    factor_set: FactorSetId,
    parquet_dir: Path,
    variant_dir: Path,
    start: str,
    end: str,
    tracking_error_annual: float,
    risk_model: str,
    fee: float = DEFAULT_FEE,
    sell_tax: float = DEFAULT_SELL_TAX,
    slippage: float = DEFAULT_SLIPPAGE,
    force: bool = False,
) -> VariantResult:
    config = build_emp008_config(
        tracking_error_annual=tracking_error_annual,
        risk_model=risk_model,
        factor_set=factor_set.value,
    )
    metadata_state = _load_backtest_metadata(
        variant_dir=variant_dir,
        expected={
            "start": start,
            "end": end,
            "tracking_error_annual": tracking_error_annual,
            "risk_model": risk_model,
            "factor_set": factor_set.value,
            "fill_mode": "close",
            "costs": {"fee": fee, "sell_tax": sell_tax, "slippage": slippage},
            "capital": 100_000_000.0,
            "allow_fractional": True,
        },
    )
    weights_dir = variant_dir / "weights"
    weights_csv = weights_dir / "target_weights.csv"
    active_weights_path = weights_dir / "active_weights.parquet"
    active_share_path = weights_dir / "active_share.parquet"
    weights_were_rebuilt = not _weights_cache_is_compatible(
        metadata_state.get("payload"),
        start=start,
        end=end,
        tracking_error_annual=tracking_error_annual,
        risk_model=risk_model,
        factor_set=factor_set.value,
    )

    if force or weights_were_rebuilt or not weights_csv.exists() or not active_weights_path.exists() or not active_share_path.exists():
        weights_were_rebuilt = True
        result = run_emp008(
            parquet_dir=parquet_dir,
            start=start,
            end=end,
            config=config,
            output_dir=weights_dir,
        )
        write_target_weights_csv(result.target_weights, weights_csv)
        write_active_share(active_weights_path)

    returns_csv = None if metadata_state["valid"] is False else metadata_state["returns_csv"]
    if force or weights_were_rebuilt or returns_csv is None:
        weight_dates = pd.to_datetime(pd.read_csv(weights_csv, index_col=0, usecols=[0]).index)
        dates = tuple(weight_dates[weight_dates <= pd.Timestamp(end)].strftime("%Y-%m-%d"))
        spec = build_target_weight_spec(
            name=f"emp008_{factor_set.value}",
            weights_csv=weights_csv,
            dates=dates,
            end=end,
            fill_mode="close",
            capital=100_000_000.0,
            fee=fee,
            sell_tax=sell_tax,
            slippage=slippage,
            allow_fractional=True,
        )
        runner = BacktestRunner(result_dir=variant_dir / "backtests", write_report_assets=False, profile=True)
        report = runner.run_spec(runner.resolve_spec(spec))
        output_dir = Path(str(report.output_dir))
        returns_csv = output_dir / "series" / "returns.csv"
        metadata = {
            "start": start,
            "end": end,
            "tracking_error_annual": tracking_error_annual,
            "risk_model": risk_model,
            "factor_set": factor_set.value,
            "fill_mode": "close",
            "costs": {"fee": fee, "sell_tax": sell_tax, "slippage": slippage},
            "capital": 100_000_000.0,
            "allow_fractional": True,
            "config": _json_ready(asdict(config)),
            "backtest_output_dir": str(output_dir),
            "returns_csv": str(returns_csv),
        }
        (variant_dir / "backtest_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    returns = pd.read_csv(returns_csv, parse_dates=["date"]).set_index("date")["returns"].astype(float).sort_index()
    returns.index.name = None
    returns.name = "returns"
    return VariantResult(
        factor_set=factor_set,
        returns=returns,
        returns_csv=returns_csv,
        weights_dir=weights_dir,
        active_share=active_share_summary(active_share_path),
    )


def run_size_value_measure_comparison(
    *,
    parquet_dir: Path,
    output_dir: Path,
    start: str,
    end: str,
    tracking_error_annual: float,
    risk_model: str,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    fee: float = DEFAULT_FEE,
    sell_tax: float = DEFAULT_SELL_TAX,
    slippage: float = DEFAULT_SLIPPAGE,
    historical_returns_csv: Path = DEFAULT_HISTORICAL_RETURNS_CSV,
    size_only_variant_dir: Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    factor_sets = validate_variants(variants)
    output_dir.mkdir(parents=True, exist_ok=True)

    returns_by_variant: dict[str, pd.Series] = {}
    active_share_by_variant: dict[str, dict[str, object]] = {}
    for factor_set in factor_sets:
        variant_dir = output_dir / factor_set.value
        if factor_set is FactorSetId.SIZE_ONLY and size_only_variant_dir is not None:
            variant_dir = size_only_variant_dir
        result = run_portfolio_variant(
            factor_set=factor_set,
            parquet_dir=parquet_dir,
            variant_dir=variant_dir,
            start=start,
            end=end,
            tracking_error_annual=tracking_error_annual,
            risk_model=risk_model,
            fee=fee,
            sell_tax=sell_tax,
            slippage=slippage,
            force=force,
        )
        returns_by_variant[factor_set.value] = result.returns
        active_share_by_variant[factor_set.value] = result.active_share

    summary, daily = build_comparison_tables(
        returns_by_variant=returns_by_variant,
        benchmark_returns=_benchmark_returns(
            parquet_dir / "qw_BM.parquet",
            "IKS200",
            next(iter(returns_by_variant.values())).index,
        ),
        active_share_by_variant=active_share_by_variant,
    )
    historical_portfolios, historical_benchmark = _load_historical_comparison_returns(historical_returns_csv)
    yearly_portfolios = pd.DataFrame(
        {variant: daily[f"{variant}_return"] for variant in returns_by_variant},
        index=daily.index,
    ).join(historical_portfolios, how="inner")
    yearly_benchmark = daily["benchmark_return"].reindex(yearly_portfolios.index)
    historical_benchmark = historical_benchmark.reindex(yearly_portfolios.index)
    if not yearly_benchmark.sub(historical_benchmark).abs().le(1e-12).all():
        raise ValueError("historical KOSPI200 BM does not match the comparison benchmark")
    manifest = _build_manifest(
        factor_sets=factor_sets,
        start=start,
        end=end,
        tracking_error_annual=tracking_error_annual,
        risk_model=risk_model,
        fee=fee,
        sell_tax=sell_tax,
        slippage=slippage,
    )
    output_payload = write_comparison_outputs(
        output_dir=output_dir,
        summary=summary,
        daily=daily,
        yearly_portfolio_returns=yearly_portfolios,
        manifest=manifest,
        shared_assumptions=(
            f"All variants use close fills with fee {fee:.4%}, sell tax {sell_tax:.4%}, and slippage {slippage:.4%}.",
            "All variants use the IKS200 benchmark on common aligned return dates.",
        ),
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    payload = {
        "output_dir": str(output_dir),
        **output_payload,
        "manifest_json": str(manifest_path),
        "variants": list(variants),
        "start": start,
        "end": end,
    }
    return payload


def build_comparison_tables(
    *,
    returns_by_variant: dict[str, pd.Series],
    benchmark_returns: pd.Series,
    active_share_by_variant: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variant_order = _validate_return_inputs(
        returns_by_variant=returns_by_variant,
        active_share_by_variant=active_share_by_variant,
    )

    aligned_benchmark = benchmark_returns.astype(float).dropna().copy()
    aligned_benchmark.index = pd.to_datetime(aligned_benchmark.index)
    common_index = pd.DatetimeIndex(aligned_benchmark.index)

    aligned_returns: dict[str, pd.Series] = {}
    for variant in variant_order:
        series = returns_by_variant[variant].astype(float).dropna().copy()
        series.index = pd.to_datetime(series.index)
        aligned_returns[variant] = series
        common_index = common_index.intersection(series.index)

    common_index = common_index.sort_values()
    if common_index.empty:
        raise ValueError("no common return dates")

    daily = pd.DataFrame(index=common_index)
    daily.index.name = "date"
    daily["benchmark_return"] = aligned_benchmark.reindex(common_index)

    excess_input: dict[str, pd.Series] = {}
    summary_rows: list[dict[str, float | str | object]] = []
    for variant in variant_order:
        strategy_returns = aligned_returns[variant].reindex(common_index)
        excess_returns = strategy_returns.sub(daily["benchmark_return"])
        daily[f"{variant}_return"] = strategy_returns
        daily[f"{variant}_excess_return"] = excess_returns
        excess_input[variant] = excess_returns

    daily = daily.dropna()
    if daily.empty:
        raise ValueError("no common return dates")
    common_index = pd.DatetimeIndex(daily.index)
    excess_summary = excess_summary_bps(pd.DataFrame(excess_input, index=common_index), periods_per_year=252)
    benchmark_wealth = (1.0 + daily["benchmark_return"]).cumprod()
    for variant in variant_order:
        strategy_metrics = performance_metrics(daily[f"{variant}_return"], periods_per_year=252)
        excess_metrics = excess_summary.loc[variant].to_dict()
        strategy_wealth = (1.0 + daily[f"{variant}_return"]).cumprod()
        cumulative_relative_excess_bp = float(
            (strategy_wealth.iloc[-1] / benchmark_wealth.iloc[-1] - 1.0) * 10_000.0
        )
        summary_rows.append(
            {
                "variant": variant,
                **strategy_metrics,
                **{key: float(value) for key, value in excess_metrics.items()},
                "cumulative_relative_excess_bp": cumulative_relative_excess_bp,
                "mean_active_share_pct": float(active_share_by_variant[variant]["mean_pct"]),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["annualized_excess_bp", "information_ratio"],
        ascending=[False, False],
        kind="mergesort",
    )
    summary = summary.reset_index(drop=True)
    return summary, daily


def write_comparison_outputs(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    yearly_portfolio_returns: pd.DataFrame | None = None,
    manifest: Mapping[str, object] | None = None,
    shared_assumptions: tuple[str, ...] = (),
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = _variants_from_summary(summary)
    _configure_plot_font()

    summary_csv = output_dir / "performance_summary.csv"
    summary_xlsx = output_dir / "performance_summary.xlsx"
    performance_table_ko_csv = output_dir / "performance_table_ko.csv"
    daily_csv = output_dir / "daily_returns.csv"
    performance_dashboard_png = output_dir / "performance_dashboard.png"
    cumulative_returns_png = output_dir / "cumulative_returns.png"
    cumulative_excess_returns_png = output_dir / "cumulative_excess_returns.png"
    yearly_excess_returns_png = output_dir / "yearly_excess_returns.png"
    yearly_returns_csv = output_dir / "yearly_returns_pct.csv"
    yearly_excess_returns_csv = output_dir / "yearly_excess_returns_bp.csv"
    yearly_performance_xlsx = output_dir / "yearly_performance.xlsx"
    interpretation_md = output_dir / "interpretation.md"

    annual_input = yearly_portfolio_returns
    if annual_input is None:
        annual_input = pd.DataFrame(
            {variant: daily[f"{variant}_return"] for variant in variants},
            index=daily.index,
        )
    plot_variants = tuple(str(column) for column in annual_input.columns)
    plot_daily = pd.DataFrame(
        {"benchmark_return": daily["benchmark_return"].reindex(annual_input.index)},
        index=annual_input.index,
    )
    for variant in plot_variants:
        plot_daily[f"{variant}_return"] = annual_input[variant]
    plot_daily = plot_daily.dropna()
    performance_table_ko = _performance_table_ko(
        portfolio_returns=annual_input,
        benchmark_returns=daily["benchmark_return"],
    )

    summary.to_csv(summary_csv, index=False)
    performance_table_ko.to_csv(performance_table_ko_csv, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_csv)
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        performance_table_ko.to_excel(writer, sheet_name="성과지표", index=False)
        daily.to_excel(writer, sheet_name="daily_returns")

    _plot_performance_dashboard(
        summary=summary,
        daily=daily,
        path=performance_dashboard_png,
    )
    _plot_cumulative_returns(
        daily=plot_daily,
        variants=plot_variants,
        path=cumulative_returns_png,
    )
    _plot_cumulative_excess_returns(
        daily=plot_daily,
        variants=plot_variants,
        path=cumulative_excess_returns_png,
    )
    yearly_frames = _yearly_cumulative_excess_bp_frames(
        portfolio_returns=annual_input,
        benchmark_returns=daily["benchmark_return"],
    )
    yearly_returns, yearly_excess = _yearly_performance_tables(
        portfolio_returns=annual_input,
        benchmark_returns=daily["benchmark_return"],
    )
    yearly_returns.to_csv(yearly_returns_csv)
    yearly_excess.to_csv(yearly_excess_returns_csv)
    with pd.ExcelWriter(yearly_performance_xlsx, engine="openpyxl") as writer:
        yearly_returns.to_excel(writer, sheet_name="yearly_returns_pct")
        yearly_excess.to_excel(writer, sheet_name="yearly_excess_bp")
    _plot_yearly_excess_returns(yearly_frames=yearly_frames, path=yearly_excess_returns_png)
    interpretation_md.write_text(
        _build_interpretation_markdown(
            summary=summary,
            manifest=manifest,
            shared_assumptions=shared_assumptions,
        ),
        encoding="utf-8",
    )

    return {
        "performance_summary_csv": str(summary_csv),
        "performance_summary_xlsx": str(summary_xlsx),
        "performance_table_ko_csv": str(performance_table_ko_csv),
        "daily_returns_csv": str(daily_csv),
        "performance_dashboard_png": str(performance_dashboard_png),
        "cumulative_returns_png": str(cumulative_returns_png),
        "cumulative_excess_returns_png": str(cumulative_excess_returns_png),
        "yearly_excess_returns_png": str(yearly_excess_returns_png),
        "yearly_returns_csv": str(yearly_returns_csv),
        "yearly_excess_returns_csv": str(yearly_excess_returns_csv),
        "yearly_performance_xlsx": str(yearly_performance_xlsx),
        "interpretation_md": str(interpretation_md),
    }


def _validate_return_inputs(
    *,
    returns_by_variant: dict[str, pd.Series],
    active_share_by_variant: dict[str, dict[str, object]],
) -> tuple[str, ...]:
    if not returns_by_variant:
        raise ValueError("missing return mappings")
    return_names = tuple(returns_by_variant)
    validate_variants(return_names)
    if set(returns_by_variant) != set(active_share_by_variant):
        raise ValueError("missing return mappings")
    return return_names


def _variants_from_summary(summary: pd.DataFrame) -> tuple[str, ...]:
    if "variant" not in summary.columns:
        raise ValueError("summary must include variant column")
    variants = tuple(summary["variant"].astype(str).tolist())
    validate_variants(variants)
    return variants


def _normalized_wealth_frame(daily: pd.DataFrame, variants: tuple[str, ...]) -> pd.DataFrame:
    returns = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(daily.index), name="date"))
    returns["benchmark"] = daily["benchmark_return"].astype(float)
    for variant in variants:
        returns[variant] = daily[f"{variant}_return"].astype(float)
    wealth = (1.0 + returns).cumprod()
    return wealth.div(wealth.iloc[0]).mul(100.0)


def _cumulative_excess_bp_frame(wealth: pd.DataFrame, variants: tuple[str, ...]) -> pd.DataFrame:
    excess = wealth.loc[:, list(variants)].div(wealth["benchmark"], axis=0).sub(1.0).mul(10_000.0)
    excess.index.name = "date"
    return excess


def _load_historical_comparison_returns(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path, parse_dates=["date"], encoding="utf-8").set_index("date").sort_index()
    required = (*_HISTORICAL_COLUMN_MAP, "KOSPI200 BM")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required historical comparison columns: {missing}")
    portfolios = frame.loc[:, list(_HISTORICAL_COLUMN_MAP)].rename(columns=_HISTORICAL_COLUMN_MAP).astype(float)
    benchmark = frame["KOSPI200 BM"].astype(float).rename("benchmark_return")
    return portfolios, benchmark


def _yearly_cumulative_excess_bp_frames(
    *,
    portfolio_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> dict[int, pd.DataFrame]:
    aligned = portfolio_returns.astype(float).join(
        benchmark_returns.astype(float).rename("__benchmark__"),
        how="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError("no common yearly excess-return dates")
    aligned.index = pd.to_datetime(aligned.index)
    aligned = aligned.sort_index()
    aligned = _drop_prior_year_zero_baseline(aligned)

    yearly: dict[int, pd.DataFrame] = {}
    for year, annual_returns in aligned.groupby(aligned.index.year, sort=True):
        wealth = (1.0 + annual_returns).cumprod()
        relative = wealth.drop(columns="__benchmark__").div(wealth["__benchmark__"], axis=0).sub(1.0).mul(10_000.0)
        prior_dates = aligned.index[aligned.index < annual_returns.index[0]]
        baseline_date = prior_dates[-1] if len(prior_dates) else annual_returns.index[0] - pd.Timedelta(days=1)
        baseline = pd.DataFrame(0.0, index=pd.DatetimeIndex([baseline_date]), columns=relative.columns)
        relative = pd.concat([baseline, relative])
        relative.index.name = "date"
        yearly[int(year)] = relative
    return yearly


def _yearly_performance_tables(
    *,
    portfolio_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = portfolio_returns.astype(float).join(
        benchmark_returns.astype(float).rename("__benchmark__"),
        how="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError("no common yearly performance dates")
    aligned.index = pd.to_datetime(aligned.index)
    aligned = aligned.sort_index()
    aligned = _drop_prior_year_zero_baseline(aligned)

    annual_return_rows: dict[int, pd.Series] = {}
    for year, annual_returns in aligned.groupby(aligned.index.year, sort=True):
        annual_return_rows[int(year)] = (1.0 + annual_returns).prod().sub(1.0).mul(100.0)

    yearly_returns = pd.DataFrame.from_dict(annual_return_rows, orient="index")
    yearly_returns = yearly_returns.rename(columns={**_YEARLY_DISPLAY_LABELS, "__benchmark__": "KOSPI200 BM"})
    yearly_returns = yearly_returns.loc[:, ["KOSPI200 BM", *[_YEARLY_DISPLAY_LABELS[column] for column in portfolio_returns]]]

    frames = _yearly_cumulative_excess_bp_frames(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
    )
    yearly_excess = pd.DataFrame(
        {year: frame.iloc[-1] for year, frame in frames.items()}
    ).T.rename(columns=_YEARLY_DISPLAY_LABELS)
    yearly_excess.insert(0, "KOSPI200 BM", 0.0)
    yearly_returns.index.name = "연도"
    yearly_excess.index.name = "연도"
    return yearly_returns, yearly_excess


def _drop_prior_year_zero_baseline(aligned: pd.DataFrame) -> pd.DataFrame:
    if (
        len(aligned) > 1
        and aligned.iloc[0].eq(0.0).all()
        and aligned.index[0].year < aligned.index[1].year
    ):
        return aligned.iloc[1:]
    return aligned


def _performance_table_ko(
    *,
    portfolio_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    aligned = portfolio_returns.astype(float).join(
        benchmark_returns.astype(float).rename("__benchmark__"),
        how="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError("no common performance-table dates")
    aligned.index = pd.to_datetime(aligned.index)
    aligned = aligned.sort_index()

    rows: list[dict[str, float | str]] = []
    for column in (*portfolio_returns.columns, "__benchmark__"):
        metrics = performance_metrics(aligned[column], periods_per_year=252)
        label = "KOSPI200 BM" if column == "__benchmark__" else _YEARLY_DISPLAY_LABELS[str(column)]
        rows.append(
            {
                "모델": label,
                "CAGR (%)": metrics["cagr_pct"],
                "연 변동성 (%)": metrics["annual_vol_pct"],
                "Sharpe": metrics["sharpe"],
                "MDD (%)": metrics["max_drawdown_pct"],
                "누적수익률 (%)": metrics["total_return_pct"],
            }
        )
    return pd.DataFrame(rows)


def _plot_yearly_excess_returns(*, yearly_frames: Mapping[int, pd.DataFrame], path: Path) -> None:
    years = tuple(yearly_frames)
    if not years:
        raise ValueError("yearly excess-return frames must not be empty")
    fig, axes = plt.subplots(4, 2, figsize=(16, 15), sharey=True)
    flat_axes = axes.ravel()
    for ax, year in zip(flat_axes, years, strict=False):
        frame = yearly_frames[year]
        for column in frame.columns:
            ax.plot(
                frame.index,
                frame[column],
                color=_YEARLY_COLORS.get(column, "#64748B"),
                linewidth=1.45,
                label=_YEARLY_DISPLAY_LABELS.get(column, column),
            )
        ax.axhline(0.0, color="#94A3B8", linewidth=0.8)
        ax.set_title(str(year), loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", labelrotation=20)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in flat_axes[len(years) :]:
        ax.remove()

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.suptitle("KOSPI200 대비 연도별 누적 초과성과 (매년 0bp 재설정)", fontsize=18, fontweight="bold")
    fig.supxlabel("날짜")
    fig.supylabel("누적 초과성과 (bp)")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=False)
    fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.92), h_pad=2.0)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_cumulative_returns(*, daily: pd.DataFrame, variants: tuple[str, ...], path: Path) -> None:
    wealth = _normalized_wealth_frame(daily, variants)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(wealth.index, wealth["benchmark"], color="#475569", linewidth=1.7, label="KOSPI200 BM")
    for variant in variants:
        ax.plot(
            wealth.index,
            wealth[variant],
            color=_YEARLY_COLORS[variant],
            linewidth=1.7,
            label=_PLOT_DISPLAY_LABELS[variant],
        )
    ax.set_title("EMP008 팩터 포트폴리오 누적수익")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV (시작=100)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_cumulative_excess_returns(*, daily: pd.DataFrame, variants: tuple[str, ...], path: Path) -> None:
    wealth = _normalized_wealth_frame(daily, variants)
    excess = _cumulative_excess_bp_frame(wealth, variants)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for variant in variants:
        values = excess[variant].astype(float)
        color = _YEARLY_COLORS[variant]
        ax.plot(values.index, values, color=color, linewidth=1.5, label=f"{_PLOT_DISPLAY_LABELS[variant]} vs BM")
        ax.fill_between(values.index, 0.0, values.to_numpy(), color=color, alpha=0.14)
    ax.set_title("KOSPI200 대비 누적 초과성과")
    ax.set_xlabel("Date")
    ax.set_ylabel("누적 초과성과 (bp)")
    ax.axhline(0.0, color="#94A3B8", linewidth=0.8)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _configure_plot_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"):
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False


def _plot_performance_dashboard(*, summary: pd.DataFrame, daily: pd.DataFrame, path: Path) -> None:
    variants = _variants_from_summary(summary)
    colors = [_VARIANT_COLORS[variant] for variant in variants]
    labels = [_DISPLAY_LABELS[variant] for variant in variants]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="#f4f6f8")
    fig.suptitle(
        _comparison_title(variants),
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#172033",
    )
    start = pd.Timestamp(daily.index.min()).date().isoformat()
    end = pd.Timestamp(daily.index.max()).date().isoformat()
    winner = _DISPLAY_LABELS[str(summary.iloc[0]["variant"])]
    fig.text(
        0.06,
        0.925,
        f"{start} to {end}  |  IKS200 benchmark  |  Net of configured costs  |  Best excess return: {winner}",
        ha="left",
        fontsize=10.5,
        color="#526174",
    )

    _plot_metric_bars(
        ax=axes[0, 0],
        labels=labels,
        values=summary["annualized_excess_bp"].astype(float).tolist(),
        colors=colors,
        title="Annualized excess return",
        value_format="{:.1f} bp",
    )
    _plot_metric_bars(
        ax=axes[0, 1],
        labels=labels,
        values=summary["information_ratio"].astype(float).tolist(),
        colors=colors,
        title="Information ratio",
        value_format="{:.3f}",
    )
    _plot_metric_bars(
        ax=axes[1, 0],
        labels=labels,
        values=summary["mean_active_share_pct"].astype(float).tolist(),
        colors=colors,
        title="Mean active share",
        value_format="{:.2f}%",
    )

    cumulative_ax = axes[1, 1]
    cumulative_excess = _cumulative_excess_bp_frame(_normalized_wealth_frame(daily, variants), variants)
    for variant in variants:
        cumulative_ax.plot(
            cumulative_excess.index,
            cumulative_excess[variant].to_numpy(),
            color=_VARIANT_COLORS[variant],
            linewidth=2.1,
            label=_DISPLAY_LABELS[variant],
        )
    cumulative_ax.axhline(0.0, color="#738094", linewidth=0.8)
    cumulative_ax.set_title("Cumulative excess return", loc="left", fontweight="bold", color="#26354a")
    cumulative_ax.set_ylabel("Basis points")
    cumulative_ax.grid(True, alpha=0.22)
    cumulative_ax.legend(frameon=False, fontsize=9, loc="best")
    cumulative_ax.tick_params(axis="x", labelrotation=20)

    for ax in axes.flat:
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("#d8dee8")
    fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.9), h_pad=3.0, w_pad=2.5)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_metric_bars(
    *,
    ax: Axes,
    labels: list[str],
    values: list[float],
    colors: list[str],
    title: str,
    value_format: str,
) -> None:
    bars = ax.barh(labels, values, color=colors, height=0.56)
    ax.invert_yaxis()
    ax.axvline(0.0, color="#738094", linewidth=0.8)
    ax.set_title(title, loc="left", fontweight="bold", color="#26354a")
    ax.grid(axis="x", alpha=0.18)
    ax.tick_params(axis="y", length=0)
    ax.margins(x=0.18)
    for bar, value in zip(bars, values, strict=True):
        offset = max(max(abs(item) for item in values) * 0.025, 0.01)
        ax.text(
            value + (offset if value >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            value_format.format(value),
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
            fontweight="bold",
            color="#26354a",
        )


def _build_interpretation_markdown(
    *,
    summary: pd.DataFrame,
    manifest: Mapping[str, object] | None,
    shared_assumptions: tuple[str, ...],
) -> str:
    top_excess = str(summary.iloc[0]["variant"])
    top_ir = str(summary.sort_values("information_ratio", ascending=False, kind="mergesort").iloc[0]["variant"])
    variants = ", ".join(_variants_from_summary(summary))
    title = _comparison_title(_variants_from_summary(summary)).removeprefix("EMP008 ")

    lines = [
        f"# {title}",
        "",
        "## Shared assumptions",
    ]
    if shared_assumptions:
        lines.extend(f"- {assumption}" for assumption in shared_assumptions)
    else:
        lines.append("- All variants are evaluated on the same common daily return dates.")
    if manifest:
        lines.append("")
        lines.append("## Manifest")
        lines.extend(f"- {key}: {value}" for key, value in manifest.items())
    lines.extend(
        [
            "",
            "## Variants",
            f"- {variants}",
            "",
            "## Ranked metrics",
            _frame_to_markdown_table(summary),
            "",
            f"Top annualized excess variant: `{top_excess}`",
            f"Top information ratio variant: `{top_ir}`",
            "",
            "This descriptive comparison does not claim statistical significance.",
            "",
        ]
    )
    return "\n".join(lines)


def _comparison_title(variants: tuple[str, ...]) -> str:
    if any(variant in MOMENTUM_VARIANTS[1:] for variant in variants):
        return "EMP008 Size + Momentum Measure Comparison"
    if any(variant in FLOW_VARIANTS[1:] for variant in variants):
        return "EMP008 Size + Flow Measure Comparison"
    return "EMP008 Size + Value Measure Comparison"


def _frame_to_markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in frame.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, float):
                rendered.append(f"{value:.6g}")
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, divider, *rows])


def _build_manifest(
    *,
    factor_sets: tuple[FactorSetId, ...],
    start: str,
    end: str,
    tracking_error_annual: float,
    risk_model: str,
    fee: float,
    sell_tax: float,
    slippage: float,
) -> dict[str, object]:
    variants: list[dict[str, object]] = []
    for factor_set in factor_sets:
        factor_set_definition = get_factor_set_definition(factor_set)
        datasets = sorted(
            {
                dataset_id.value
                for definition in factor_definitions_for_set(factor_set)
                for dataset_id in definition.datasets
            }
        )
        variants.append(
            {
                "factor_set": factor_set_definition.id.value,
                "factors": [factor.value for factor in factor_set_definition.factors],
                "datasets": datasets,
            }
        )
    return {
        "start": start,
        "end": end,
        "tracking_error_annual": tracking_error_annual,
        "risk_model": risk_model,
        "fill_mode": "close",
        "costs": {
            "fee": fee,
            "sell_tax": sell_tax,
            "slippage": slippage,
        },
        "benchmark": "IKS200",
        "variants": variants,
    }


def _load_backtest_metadata(
    *,
    variant_dir: Path,
    expected: Mapping[str, object],
) -> dict[str, object]:
    metadata_path = variant_dir / "backtest_metadata.json"
    if not metadata_path.exists():
        return {"valid": False, "returns_csv": None, "payload": None}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"valid": False, "returns_csv": None, "payload": None}
    if not isinstance(payload, dict):
        return {"valid": False, "returns_csv": None, "payload": None}
    returns_csv = payload.get("returns_csv")
    backtest_output_dir = payload.get("backtest_output_dir")
    if not isinstance(returns_csv, str) or not isinstance(backtest_output_dir, str):
        return {"valid": False, "returns_csv": None, "payload": payload}
    backtests_root = (variant_dir / "backtests").resolve()
    try:
        returns_path = Path(returns_csv).resolve()
        output_dir = Path(backtest_output_dir).resolve()
        returns_path.relative_to(backtests_root)
        output_dir.relative_to(backtests_root)
    except (OSError, RuntimeError, ValueError):
        return {"valid": False, "returns_csv": None, "payload": None}
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            return {"valid": False, "returns_csv": None, "payload": payload}
    if not returns_path.exists():
        return {"valid": False, "returns_csv": None, "payload": payload}
    return {"valid": True, "returns_csv": returns_path, "payload": payload}


def _weights_cache_is_compatible(
    payload: object,
    *,
    start: str,
    end: str,
    tracking_error_annual: float,
    risk_model: str,
    factor_set: str,
) -> bool:
    if not isinstance(payload, dict):
        return False
    expected = {
        "start": start,
        "tracking_error_annual": tracking_error_annual,
        "risk_model": risk_model,
        "factor_set": factor_set,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return False
    try:
        return pd.Timestamp(str(payload["end"])) >= pd.Timestamp(end)
    except (KeyError, TypeError, ValueError):
        return False


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        raw = getattr(value, "value")
        if isinstance(raw, str):
            return raw
    return value


def _resolve_shared_end(
    *,
    parquet_dir: Path,
    factor_sets: tuple[FactorSetId, ...],
    tracking_error_annual: float,
    risk_model: str,
) -> str:
    latest_ends = []
    for factor_set in factor_sets:
        config = build_emp008_config(
            tracking_error_annual=tracking_error_annual,
            risk_model=risk_model,
            factor_set=factor_set.value,
        )
        latest_ends.append(pd.Timestamp(latest_common_end(parquet_dir, config)))
    return min(latest_ends).date().isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare EMP008 size-plus-factor measure variants.")
    parser.add_argument("--parquet-dir", type=Path, default=Path("parquet"))
    parser.add_argument("--comparison-profile", choices=("value", "momentum", "flow"), default="value")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", help="Shared end date. Default: minimum latest_common_end across selected variants.")
    parser.add_argument("--tracking-error-annual", type=float, default=0.007)
    parser.add_argument("--risk-model", choices=("factor_idio", "direct_covariance"), default="factor_idio")
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE)
    parser.add_argument("--sell-tax", type=float, default=DEFAULT_SELL_TAX)
    parser.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE)
    parser.add_argument("--historical-returns-csv", type=Path, default=DEFAULT_HISTORICAL_RETURNS_CSV)
    parser.add_argument("--variants", nargs="+", choices=SUPPORTED_VARIANTS)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    profile_variants = {
        "value": DEFAULT_VARIANTS,
        "momentum": MOMENTUM_VARIANTS,
        "flow": FLOW_VARIANTS,
    }[args.comparison_profile]
    variants = tuple(args.variants) if args.variants else profile_variants
    output_dir = args.output_dir or {
        "value": DEFAULT_OUTPUT_DIR,
        "momentum": DEFAULT_MOMENTUM_OUTPUT_DIR,
        "flow": DEFAULT_FLOW_OUTPUT_DIR,
    }[args.comparison_profile]
    factor_sets = validate_variants(variants)
    end = args.end or _resolve_shared_end(
        parquet_dir=args.parquet_dir,
        factor_sets=factor_sets,
        tracking_error_annual=args.tracking_error_annual,
        risk_model=args.risk_model,
    )
    payload = run_size_value_measure_comparison(
        parquet_dir=args.parquet_dir,
        output_dir=output_dir,
        start=args.start,
        end=end,
        tracking_error_annual=args.tracking_error_annual,
        risk_model=args.risk_model,
        variants=variants,
        fee=args.fee,
        sell_tax=args.sell_tax,
        slippage=args.slippage,
        historical_returns_csv=args.historical_returns_csv,
        size_only_variant_dir=(
            DEFAULT_OUTPUT_DIR / FactorSetId.SIZE_ONLY.value
            if args.comparison_profile in {"momentum", "flow"}
            and (DEFAULT_OUTPUT_DIR / FactorSetId.SIZE_ONLY.value / "backtest_metadata.json").exists()
            else None
        ),
        force=args.force,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_FLOW_OUTPUT_DIR",
    "DEFAULT_MOMENTUM_OUTPUT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_FEE",
    "DEFAULT_HISTORICAL_RETURNS_CSV",
    "DEFAULT_SELL_TAX",
    "DEFAULT_SLIPPAGE",
    "DEFAULT_START",
    "DEFAULT_VARIANTS",
    "FLOW_VARIANTS",
    "MOMENTUM_VARIANTS",
    "SUPPORTED_VARIANTS",
    "VariantResult",
    "_parser",
    "_resolve_shared_end",
    "build_comparison_tables",
    "run_portfolio_variant",
    "run_size_value_measure_comparison",
    "validate_variants",
    "write_comparison_outputs",
]
