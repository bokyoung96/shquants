from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .mfbt_emp008 import (
    _common_month_end_dates,
    _neutralize_large_benchmark_weight_factor_exposures,
    _positive_benchmark_weights,
)
from .mfbt_emp008_data import MfbtEmp008Config, load_mfbt_emp008_market
from .mfbt_emp008_factors import build_raw_mfbt_factors
from .mfbt_emp008_preprocess import build_sector_active_exposures, combine_exposures, preprocess_factor_frame
from .mfbt_emp008_risk import fit_cross_sectional_factor_returns


@dataclass(frozen=True, slots=True)
class FactorAttributionResult:
    monthly_contribution: pd.DataFrame
    cumulative_contribution: pd.DataFrame
    yearly_contribution: pd.DataFrame
    factor_summary_bps: pd.DataFrame
    active_factor_exposure: pd.DataFrame
    realized_factor_return: pd.DataFrame
    reconciliation: pd.DataFrame


def factor_attribution_row(
    *,
    active_weights: pd.Series,
    exposures: pd.DataFrame,
    factor_returns: pd.Series,
    residuals: pd.Series,
    alpha_factor_names: list[str],
    sector_factor_names: list[str],
) -> pd.Series:
    common = active_weights.index.intersection(exposures.index)
    active = active_weights.reindex(common).fillna(0.0).astype(float)
    z = exposures.reindex(index=common).fillna(0.0).astype(float)
    active_exposure = z.mul(active, axis=0).sum(axis=0)
    factor_contribution = active_exposure.mul(factor_returns.reindex(active_exposure.index).fillna(0.0))
    specific = float(active.mul(residuals.reindex(common).fillna(0.0)).sum())

    alpha_total = float(factor_contribution.reindex(alpha_factor_names).fillna(0.0).sum())
    sector_total = float(factor_contribution.reindex(sector_factor_names).fillna(0.0).sum())
    model_active_return = alpha_total + sector_total + specific
    row = factor_contribution.reindex([*alpha_factor_names, *sector_factor_names]).fillna(0.0).astype(float)
    row["alpha_total"] = alpha_total
    row["sector_total"] = sector_total
    row["specific"] = specific
    row["model_active_return"] = model_active_return
    return row


def build_emp008_factor_attribution(
    *,
    parquet_dir: Path,
    run_root: Path,
    output_dir: Path | None = None,
    config: MfbtEmp008Config | None = None,
) -> dict[str, object]:
    active_weights = pd.read_parquet(run_root / "weights" / "active_weights.parquet").astype(float)
    active_weights.index = pd.to_datetime(active_weights.index)
    if active_weights.empty:
        raise ValueError("active weights are empty")

    active_config = config or MfbtEmp008Config()
    start = active_weights.index.min().date().isoformat()
    end = active_weights.index.max().date().isoformat()
    market = load_mfbt_emp008_market(parquet_dir=parquet_dir, start=start, end=end, config=active_config)
    raw_factors = build_raw_mfbt_factors(market, active_config)
    alpha_factor_names = list(raw_factors)
    monthly_dates = _common_month_end_dates(raw_factors)

    close = market.frames["close"].astype(float)
    float_mktcap = market.frames["float_market_cap"].reindex(index=close.index, columns=close.columns).astype(float)
    universe = market.frames["k200_yn"].reindex(index=close.index, columns=close.columns).fillna(0).astype(bool)
    sector = market.frames["sector_neutral_big"].reindex(index=close.index, columns=close.columns).ffill()
    bm_weights = market.frames["bm_weights"].reindex(index=close.index, columns=close.columns).astype(float)

    alpha_factors = {
        name: preprocess_factor_frame(
            frame,
            float_mktcap,
            universe,
            rank_transform=name in active_config.rank_transform_factors,
            winsor_quantile=active_config.value_raw_winsor_quantile if name == "value" else None,
            zscore_cap=active_config.value_zscore_cap if name == "value" else None,
        )
        for name, frame in raw_factors.items()
    }
    alpha_factors = _neutralize_large_benchmark_weight_factor_exposures(alpha_factors, bm_weights, active_config)
    sector_factors = build_sector_active_exposures(sector, float_mktcap, universe)
    sector_factor_names = list(sector_factors)

    result = _compute_attribution(
        close=close,
        bm_weights=bm_weights,
        alpha_factors=alpha_factors,
        sector_factors=sector_factors,
        active_weights=active_weights,
        monthly_dates=monthly_dates,
        alpha_factor_names=alpha_factor_names,
        sector_factor_names=sector_factor_names,
    )
    destination = output_dir or run_root / "factor_attribution"
    payload = write_factor_attribution(result, destination)
    payload["periods"] = int(len(result.monthly_contribution))
    payload["date_start"] = (
        result.monthly_contribution.index.min().date().isoformat() if not result.monthly_contribution.empty else None
    )
    payload["date_end"] = (
        result.monthly_contribution.index.max().date().isoformat() if not result.monthly_contribution.empty else None
    )
    (destination.parent / "factor_attribution_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def write_factor_attribution(result: FactorAttributionResult, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / "factor_attribution.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        result.monthly_contribution.to_excel(writer, sheet_name="monthly_contribution")
        result.cumulative_contribution.to_excel(writer, sheet_name="cumulative_contribution")
        result.yearly_contribution.to_excel(writer, sheet_name="yearly_contribution")
        result.factor_summary_bps.to_excel(writer, sheet_name="factor_summary_bps")
        result.active_factor_exposure.to_excel(writer, sheet_name="active_factor_exposure")
        result.realized_factor_return.to_excel(writer, sheet_name="realized_factor_return")
        result.reconciliation.to_excel(writer, sheet_name="reconciliation")

    cumulative_png = output_dir / "cumulative_factor_contribution.png"
    heatmap_png = output_dir / "monthly_factor_heatmap.png"
    yearly_png = output_dir / "yearly_factor_contribution.png"
    _plot_cumulative_contribution(result.cumulative_contribution, cumulative_png)
    _plot_monthly_factor_heatmap(result.monthly_contribution, heatmap_png)
    _plot_yearly_contribution(result.yearly_contribution, yearly_png)
    return {
        "excel": str(excel_path),
        "cumulative_factor_contribution_png": str(cumulative_png),
        "monthly_factor_heatmap_png": str(heatmap_png),
        "yearly_factor_contribution_png": str(yearly_png),
    }


def _compute_attribution(
    *,
    close: pd.DataFrame,
    bm_weights: pd.DataFrame,
    alpha_factors: dict[str, pd.DataFrame],
    sector_factors: dict[str, pd.DataFrame],
    active_weights: pd.DataFrame,
    monthly_dates: list[pd.Timestamp],
    alpha_factor_names: list[str],
    sector_factor_names: list[str],
) -> FactorAttributionResult:
    active_dates = set(pd.to_datetime(active_weights.index))
    contribution_rows: list[pd.Series] = []
    exposure_rows: list[pd.Series] = []
    factor_return_rows: list[pd.Series] = []
    contribution_dates: list[pd.Timestamp] = []
    reconciliation_rows: list[dict[str, float | str]] = []

    for factor_date, return_date in zip(monthly_dates[:-1], monthly_dates[1:], strict=True):
        if factor_date not in active_dates:
            continue
        if return_date > active_weights.index.max():
            continue
        exposures = combine_exposures(alpha_factors, sector_factors, factor_date)
        stock_returns = close.loc[return_date].divide(close.loc[factor_date]).sub(1.0)
        bm = _positive_benchmark_weights(bm_weights.reindex(index=[return_date], columns=stock_returns.index).iloc[0])
        excess_returns = stock_returns.sub(stock_returns.reindex(bm.index).mul(bm).sum())
        regression = fit_cross_sectional_factor_returns(exposures, excess_returns)
        active = active_weights.loc[factor_date]
        row = factor_attribution_row(
            active_weights=active,
            exposures=exposures,
            factor_returns=regression.factor_returns,
            residuals=regression.residuals,
            alpha_factor_names=alpha_factor_names,
            sector_factor_names=sector_factor_names,
        ).rename(return_date)
        contribution_rows.append(row)
        contribution_dates.append(return_date)
        active_exposure = exposures.mul(active.reindex(exposures.index).fillna(0.0), axis=0).sum(axis=0).rename(return_date)
        exposure_rows.append(active_exposure.reindex([*alpha_factor_names, *sector_factor_names]).fillna(0.0))
        factor_return_rows.append(regression.factor_returns.reindex([*alpha_factor_names, *sector_factor_names]).fillna(0.0).rename(return_date))
        actual_active = float(active.reindex(stock_returns.index).fillna(0.0).mul(stock_returns.fillna(0.0)).sum())
        reconciliation_rows.append(
            {
                "date": return_date,
                "active_weight_date": factor_date.date().isoformat(),
                "actual_active_return": actual_active,
                "model_active_return": float(row["model_active_return"]),
                "unexplained": actual_active - float(row["model_active_return"]),
            }
        )

    monthly = pd.DataFrame(contribution_rows, index=contribution_dates).fillna(0.0)
    exposures = pd.DataFrame(exposure_rows, index=contribution_dates).fillna(0.0)
    factor_returns = pd.DataFrame(factor_return_rows, index=contribution_dates).fillna(0.0)
    reconciliation = pd.DataFrame(reconciliation_rows)
    if not reconciliation.empty:
        reconciliation = reconciliation.set_index("date")
        reconciliation.index = pd.to_datetime(reconciliation.index)

    main_columns = [*alpha_factor_names, "sector_total", "specific"]
    cumulative = monthly.reindex(columns=main_columns).fillna(0.0).cumsum()
    yearly = monthly.reindex(columns=main_columns).fillna(0.0).groupby(monthly.index.year).sum()
    summary = _factor_summary_bps(monthly.reindex(columns=main_columns).fillna(0.0))
    return FactorAttributionResult(
        monthly_contribution=monthly,
        cumulative_contribution=cumulative,
        yearly_contribution=yearly,
        factor_summary_bps=summary,
        active_factor_exposure=exposures,
        realized_factor_return=factor_returns,
        reconciliation=reconciliation,
    )


def _factor_summary_bps(monthly_contribution: pd.DataFrame) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for column in monthly_contribution:
        series = monthly_contribution[column].astype(float)
        rows[column] = {
            "total_bp": float(series.sum() * 10_000.0),
            "mean_monthly_bp": float(series.mean() * 10_000.0),
            "best_month_bp": float(series.max() * 10_000.0),
            "worst_month_bp": float(series.min() * 10_000.0),
            "positive_month_rate_pct": float(series.gt(0.0).mean() * 100.0),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _plot_cumulative_contribution(cumulative: pd.DataFrame, path: Path) -> None:
    frame = cumulative * 10_000.0
    ax = frame.plot(figsize=(13, 6.8), linewidth=1.6)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Cumulative Factor Contribution")
    ax.set_ylabel("Contribution (bp)")
    ax.set_xlabel("Date")
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=160)
    plt.close(ax.figure)


def _plot_monthly_factor_heatmap(monthly: pd.DataFrame, path: Path) -> None:
    columns = [column for column in monthly.columns if column not in {"alpha_total", "model_active_return"}]
    frame = monthly.reindex(columns=columns).T * 10_000.0
    limit = float(frame.abs().max().max()) if not frame.empty else 1.0
    if limit == 0.0:
        limit = 1.0
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    image = ax.imshow(frame.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_title("Monthly Factor Contribution")
    ax.set_yticks(range(len(frame.index)))
    ax.set_yticklabels(frame.index)
    ax.set_xticks(range(len(frame.columns)))
    ax.set_xticklabels([date.strftime("%Y-%m") for date in frame.columns], rotation=90, fontsize=7)
    cbar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Contribution (bp)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_yearly_contribution(yearly: pd.DataFrame, path: Path) -> None:
    ax = (yearly * 10_000.0).plot(kind="bar", stacked=True, figsize=(12, 6.5), alpha=0.86)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Yearly Factor Contribution")
    ax.set_ylabel("Contribution (bp)")
    ax.set_xlabel("Year")
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=160)
    plt.close(ax.figure)
