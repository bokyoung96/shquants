from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def active_weight_abs_sum_frame(active_weights: pd.DataFrame) -> pd.DataFrame:
    active = active_weights.astype(float).copy()
    active.index = pd.to_datetime(active.index)
    result = pd.DataFrame(index=active.index.sort_values())
    active = active.loc[result.index]
    result["sum_abs_active_weight"] = active.abs().sum(axis=1)
    result["active_share"] = result["sum_abs_active_weight"] * 0.5
    result["sum_abs_active_weight_pct"] = result["sum_abs_active_weight"] * 100.0
    result["active_share_pct"] = result["active_share"] * 100.0
    result.index.name = "date"
    return result


def performance_metrics(returns: pd.Series, *, periods_per_year: int) -> dict[str, float]:
    clean = returns.astype(float).dropna()
    if clean.empty:
        return {
            "cagr_pct": 0.0,
            "annual_vol_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "total_return_pct": 0.0,
            "best_month_pct": 0.0,
            "worst_month_pct": 0.0,
            "positive_month_rate_pct": 0.0,
        }
    cumulative = (1.0 + clean).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0)
    years = len(clean) / periods_per_year
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    vol = float(clean.std(ddof=0) * (periods_per_year**0.5))
    sharpe = float((clean.mean() * periods_per_year) / vol) if vol > 0 else 0.0
    drawdown = cumulative.div(cumulative.cummax()).sub(1.0)
    return {
        "cagr_pct": cagr * 100.0,
        "annual_vol_pct": vol * 100.0,
        "sharpe": sharpe,
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "total_return_pct": total_return * 100.0,
        "best_month_pct": float(clean.max() * 100.0),
        "worst_month_pct": float(clean.min() * 100.0),
        "positive_month_rate_pct": float(clean.gt(0.0).mean() * 100.0),
    }


def monthly_compounded_returns(returns: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + returns.astype(float)).resample("ME").prod().sub(1.0)


def yearly_compounded_returns(returns: pd.DataFrame) -> pd.DataFrame:
    yearly = (1.0 + returns.astype(float)).groupby(returns.index.year).prod().sub(1.0)
    yearly.index.name = "year"
    return yearly


def pair_return_display_frame(returns: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "MFBT": _find_pair_column(returns, "mfbt", "gross"),
        "Origin": _find_pair_column(returns, "origin", "gross"),
        "KOSPI200 BM": "KOSPI200 BM",
    }
    return pd.DataFrame({name: returns[column].astype(float) for name, column in columns.items()}, index=returns.index)


def _find_pair_column(frame: pd.DataFrame, strategy: str, variant: str) -> str:
    matches = [
        column
        for column in frame.columns
        if strategy in str(column).lower()
        and variant in str(column).lower()
        and "excess" not in str(column).lower()
    ]
    if not matches:
        raise KeyError(f"missing {strategy} {variant} return column")
    return str(matches[0])


def excess_summary_bps(
    active_returns: pd.DataFrame,
    *,
    periods_per_year: int,
) -> pd.DataFrame:
    monthly = monthly_compounded_returns(active_returns)
    rows: dict[str, dict[str, float]] = {}
    for column in active_returns.columns:
        daily = active_returns[column].astype(float).dropna()
        monthly_series = monthly[column].astype(float).dropna()
        cumulative = (1.0 + daily).cumprod()
        total = float(cumulative.iloc[-1] - 1.0) if not cumulative.empty else 0.0
        years = len(daily) / periods_per_year if periods_per_year > 0 else 0.0
        annualized = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else 0.0
        annualized_vol = float(daily.std(ddof=0) * (periods_per_year**0.5)) if not daily.empty else 0.0
        information_ratio = annualized / annualized_vol if annualized_vol > 0 else 0.0
        rows[column] = {
            "total_excess_bp": total * 10_000.0,
            "annualized_excess_bp": annualized * 10_000.0,
            "monthly_mean_excess_bp": float(monthly_series.mean() * 10_000.0) if not monthly_series.empty else 0.0,
            "monthly_median_excess_bp": float(monthly_series.median() * 10_000.0) if not monthly_series.empty else 0.0,
            "best_month_excess_bp": float(monthly_series.max() * 10_000.0) if not monthly_series.empty else 0.0,
            "worst_month_excess_bp": float(monthly_series.min() * 10_000.0) if not monthly_series.empty else 0.0,
            "positive_month_rate_pct": float(monthly_series.gt(0.0).mean() * 100.0) if not monthly_series.empty else 0.0,
            "annualized_tracking_error_bp": annualized_vol * 10_000.0,
            "information_ratio": information_ratio,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def monthly_excess_heatmap_frame(monthly_active: pd.DataFrame, column: str) -> pd.DataFrame:
    series = monthly_active[column].astype(float).dropna() * 100.0
    frame = pd.DataFrame(
        {
            "year": series.index.year,
            "month": series.index.month,
            "value": series.to_numpy(),
        },
        index=series.index,
    )
    heatmap = frame.pivot(index="year", columns="month", values="value").sort_index()
    return heatmap.reindex(columns=range(1, 13))


def build_emp008_comparison(
    *,
    gross_run_dir: Path,
    costed_run_dir: Path,
    active_weights_parquet: Path,
    benchmark_parquet: Path,
    output_dir: Path,
    benchmark_code: str = "IKS200",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    gross_returns = _read_series(gross_run_dir / "series" / "returns.csv", "returns")
    costed_returns = _read_series(costed_run_dir / "series" / "returns.csv", "returns")
    bm_returns = _benchmark_returns(benchmark_parquet, benchmark_code, gross_returns.index)
    returns = pd.concat(
        [
            gross_returns.rename("Gross"),
            costed_returns.reindex(gross_returns.index).fillna(0.0).rename("Costed"),
            bm_returns.rename("KOSPI200 BM"),
        ],
        axis=1,
    ).fillna(0.0)
    cumulative = (1.0 + returns).cumprod().sub(1.0)
    drawdowns = (1.0 + returns).cumprod().div((1.0 + returns).cumprod().cummax()).sub(1.0)
    monthly_returns = monthly_compounded_returns(returns)

    active_returns = pd.DataFrame(
        {
            "Gross excess": returns["Gross"].sub(returns["KOSPI200 BM"]),
            "Costed excess": returns["Costed"].sub(returns["KOSPI200 BM"]),
        }
    )
    cumulative_active = (1.0 + active_returns).cumprod().sub(1.0)
    active_drawdowns = (1.0 + active_returns).cumprod().div((1.0 + active_returns).cumprod().cummax()).sub(1.0)
    monthly_active = monthly_compounded_returns(active_returns)

    active_weight_abs_sum = active_weight_abs_sum_frame(pd.read_parquet(active_weights_parquet))
    metrics = pd.DataFrame({name: performance_metrics(returns[name], periods_per_year=252) for name in returns}).T
    active_metrics = pd.DataFrame(
        {name: performance_metrics(active_returns[name], periods_per_year=252) for name in active_returns}
    ).T
    excess_bps = excess_summary_bps(active_returns, periods_per_year=252)

    excel_path = output_dir / "performance.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="metrics")
        returns.to_excel(writer, sheet_name="daily_returns")
        cumulative.to_excel(writer, sheet_name="cumulative_returns")
        drawdowns.to_excel(writer, sheet_name="drawdowns")
        monthly_returns.to_excel(writer, sheet_name="monthly_returns")
        active_returns.to_excel(writer, sheet_name="active_returns")
        cumulative_active.to_excel(writer, sheet_name="cumulative_active_returns")
        active_drawdowns.to_excel(writer, sheet_name="active_drawdowns")
        monthly_active.to_excel(writer, sheet_name="monthly_active_returns")
        active_metrics.to_excel(writer, sheet_name="active_metrics")
        excess_bps.to_excel(writer, sheet_name="excess_summary_bps")
        active_weight_abs_sum.to_excel(writer, sheet_name="active_weight_sum")

    active_weight_csv = output_dir / "active_weight_sum.csv"
    active_weight_xlsx = output_dir / "active_weight_sum.xlsx"
    active_weight_abs_sum.to_csv(active_weight_csv)
    with pd.ExcelWriter(active_weight_xlsx, engine="openpyxl") as writer:
        active_weight_abs_sum.to_excel(writer, sheet_name="active_weight_sum")

    paths = {
        "performance_xlsx": excel_path,
        "cumulative_png": output_dir / "cumulative_excess_drawdown.png",
        "monthly_excess_heatmap_png": output_dir / "monthly_excess_heatmap.png",
        "active_weight_sum_png": output_dir / "active_weight_sum.png",
        "active_weight_sum_csv": active_weight_csv,
        "active_weight_sum_xlsx": active_weight_xlsx,
    }
    _plot_cumulative_with_excess_and_drawdown(
        cumulative * 100.0,
        cumulative_active * 100.0,
        drawdowns * 100.0,
        paths["cumulative_png"],
    )
    _plot_monthly_excess_heatmap(monthly_active, paths["monthly_excess_heatmap_png"])
    _plot_active_weight_sum(active_weight_abs_sum, paths["active_weight_sum_png"])

    payload = {
        **{name: str(path) for name, path in paths.items()},
        "metrics": json.loads(metrics.to_json(orient="index")),
        "active_metrics": json.loads(active_metrics.to_json(orient="index")),
        "excess_summary_bps": json.loads(excess_bps.to_json(orient="index")),
        "active_weight_sum": {
            "rows": int(len(active_weight_abs_sum)),
            "sum_abs_active_weight_mean_pct": float(active_weight_abs_sum["sum_abs_active_weight_pct"].mean()),
            "sum_abs_active_weight_median_pct": float(active_weight_abs_sum["sum_abs_active_weight_pct"].median()),
            "sum_abs_active_weight_min_pct": float(active_weight_abs_sum["sum_abs_active_weight_pct"].min()),
            "sum_abs_active_weight_max_pct": float(active_weight_abs_sum["sum_abs_active_weight_pct"].max()),
            "active_share_mean_pct": float(active_weight_abs_sum["active_share_pct"].mean()),
            "active_share_median_pct": float(active_weight_abs_sum["active_share_pct"].median()),
            "active_share_min_pct": float(active_weight_abs_sum["active_share_pct"].min()),
            "active_share_max_pct": float(active_weight_abs_sum["active_share_pct"].max()),
        },
    }
    (output_dir.parent / "comparison_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def rebuild_emp008_pair_comparison_from_excel(
    *,
    excel_path: Path,
    output_dir: Path,
    benchmark_parquet: Path | None = None,
    benchmark_code: str = "IKS200",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = pd.read_excel(excel_path, sheet_name="run_config", index_col=0)
    returns = pd.read_excel(excel_path, sheet_name="daily_returns", index_col=0, parse_dates=True).astype(float)
    active_weight_sum = pd.read_excel(excel_path, sheet_name="active_weight_sum", index_col=0, parse_dates=True)
    if benchmark_parquet is not None:
        returns["KOSPI200 BM"] = _benchmark_returns(benchmark_parquet, benchmark_code, returns.index)

    cumulative = (1.0 + returns).cumprod().sub(1.0)
    drawdowns = (1.0 + returns).cumprod().div((1.0 + returns).cumprod().cummax()).sub(1.0)
    monthly_returns = monthly_compounded_returns(returns)
    yearly_returns = yearly_compounded_returns(pair_return_display_frame(returns))

    active_returns = _pair_active_returns(returns)
    cumulative_active = (1.0 + active_returns).cumprod().sub(1.0)
    monthly_active = monthly_compounded_returns(active_returns)
    yearly_active = yearly_compounded_returns(_pair_excess_display_frame(active_returns))

    metrics = pd.DataFrame({name: performance_metrics(returns[name], periods_per_year=252) for name in returns}).T
    excess_metrics = pd.DataFrame(
        {name: performance_metrics(active_returns[name], periods_per_year=252) for name in active_returns}
    ).T
    excess_bps = excess_summary_bps(active_returns, periods_per_year=252)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        run_config.to_excel(writer, sheet_name="run_config")
        metrics.to_excel(writer, sheet_name="performance_metrics")
        excess_metrics.to_excel(writer, sheet_name="excess_metrics")
        excess_bps.to_excel(writer, sheet_name="excess_summary_bps")
        returns.to_excel(writer, sheet_name="daily_returns")
        cumulative.to_excel(writer, sheet_name="cumulative_returns")
        drawdowns.to_excel(writer, sheet_name="drawdowns")
        monthly_returns.to_excel(writer, sheet_name="monthly_returns")
        yearly_returns.to_excel(writer, sheet_name="yearly_returns")
        active_returns.to_excel(writer, sheet_name="daily_excess_returns")
        cumulative_active.to_excel(writer, sheet_name="cumulative_excess")
        monthly_active.to_excel(writer, sheet_name="monthly_excess")
        yearly_active.to_excel(writer, sheet_name="yearly_excess")
        active_weight_sum.to_excel(writer, sheet_name="active_weight_sum")

    monthly_excess_csv = output_dir / "monthly_excess_returns.csv"
    active_weight_csv = output_dir / "active_weight_sum.csv"
    monthly_active.to_csv(monthly_excess_csv)
    active_weight_sum.to_csv(active_weight_csv)

    paths = {
        "excel": excel_path,
        "cumulative_png": output_dir / "cumulative_excess_drawdown.png",
        "yearly_performance_png": output_dir / "yearly_performance.png",
        "monthly_excess_heatmap_png": output_dir / "monthly_excess_heatmap.png",
        "active_weight_sum_png": output_dir / "active_weight_sum.png",
        "active_weight_sum_csv": active_weight_csv,
        "monthly_excess_csv": monthly_excess_csv,
    }
    _plot_pair_cumulative_with_excess_and_drawdown(returns, active_returns, drawdowns, paths["cumulative_png"])
    _plot_pair_yearly_performance(yearly_returns, yearly_active, paths["yearly_performance_png"])
    _plot_pair_monthly_excess_heatmap(monthly_active, paths["monthly_excess_heatmap_png"])
    _plot_pair_active_weight_sum(active_weight_sum, paths["active_weight_sum_png"])

    payload = {
        "output_dir": str(output_dir),
        **{name: str(path) for name, path in paths.items()},
        "common_start": returns.index.min().date().isoformat(),
        "common_end": returns.index.max().date().isoformat(),
        "metrics": json.loads(metrics.to_json(orient="index")),
        "excess_summary_bps": json.loads(excess_bps.to_json(orient="index")),
        "run_config": json.loads(run_config.to_json(orient="index")),
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def _pair_active_returns(returns: pd.DataFrame) -> pd.DataFrame:
    mfbt = _find_pair_column(returns, "mfbt", "gross")
    mfbt_costed = _find_pair_column(returns, "mfbt", "costed")
    origin = _find_pair_column(returns, "origin", "gross")
    origin_costed = _find_pair_column(returns, "origin", "costed")
    return pd.DataFrame(
        {
            "mfbt gross excess": returns[mfbt].sub(returns["KOSPI200 BM"]),
            "mfbt costed excess": returns[mfbt_costed].sub(returns["KOSPI200 BM"]),
            "origin gross excess": returns[origin].sub(returns["KOSPI200 BM"]),
            "origin costed excess": returns[origin_costed].sub(returns["KOSPI200 BM"]),
            "origin minus mfbt gross": returns[origin].sub(returns[mfbt]),
            "origin minus mfbt costed": returns[origin_costed].sub(returns[mfbt_costed]),
        },
        index=returns.index,
    )


def _pair_excess_display_frame(active_returns: pd.DataFrame) -> pd.DataFrame:
    return active_returns[["mfbt gross excess", "origin gross excess", "origin minus mfbt gross"]].rename(
        columns={
            "mfbt gross excess": "MFBT excess",
            "origin gross excess": "Origin excess",
            "origin minus mfbt gross": "Origin - MFBT",
        }
    )


def _read_series(path: Path, column: str) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return frame[column].astype(float).sort_index()


def _benchmark_returns(path: Path, code: str, comparison_index: pd.Index | None = None) -> pd.Series:
    frame = pd.read_parquet(path)
    if isinstance(frame.columns, pd.MultiIndex):
        close = frame[(code, "close")]
    elif code in frame.columns:
        close = frame[code]
    else:
        close = frame["close"]
    close = close.astype(float).replace(0.0, pd.NA).dropna().sort_index()
    if comparison_index is not None:
        close = close.reindex(pd.to_datetime(comparison_index))
    return close.pct_change(fill_method=None).fillna(0.0)


def _plot_pair_cumulative_with_excess_and_drawdown(
    returns: pd.DataFrame,
    active_returns: pd.DataFrame,
    drawdowns: pd.DataFrame,
    path: Path,
) -> None:
    display_returns = pair_return_display_frame(returns)
    cumulative = (1.0 + display_returns).cumprod().sub(1.0).mul(100.0)
    display_drawdowns = drawdowns[
        [
            _find_pair_column(returns, "mfbt", "gross"),
            _find_pair_column(returns, "origin", "gross"),
            "KOSPI200 BM",
        ]
    ].copy()
    display_drawdowns.columns = ["MFBT", "Origin", "KOSPI200 BM"]
    excess = (1.0 + _pair_excess_display_frame(active_returns)[["MFBT excess", "Origin excess"]]).cumprod().sub(1.0).mul(100.0)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0]},
    )
    ax = axes[0]
    ax_excess = ax.twinx()
    colors = {"MFBT": "#1f77b4", "Origin": "#d55e00", "KOSPI200 BM": "#4d4d4d"}
    fill_colors = {"MFBT excess": "#8ecae6", "Origin excess": "#ffb703"}

    for column in excess.columns:
        values = excess[column].astype(float)
        ax_excess.fill_between(
            values.index,
            0.0,
            values.to_numpy(),
            color=fill_colors[column],
            alpha=0.16,
            linewidth=0,
            label=column,
            zorder=1,
        )
    ax_excess.axhline(0.0, color="#7f8c8d", linewidth=0.8, alpha=0.8)
    ax_excess.set_ylabel("Cumulative excess (%)")
    ax_excess.grid(False)

    for column in cumulative.columns:
        ax.plot(
            cumulative.index,
            cumulative[column],
            color=colors[column],
            linewidth=2.1 if column != "KOSPI200 BM" else 1.8,
            label=column,
            zorder=3,
        )
    ax.set_title("Cumulative return with excess return background")
    ax.set_ylabel("Cumulative return (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax_excess.spines["top"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    excess_handles, excess_labels = ax_excess.get_legend_handles_labels()
    ax.legend(handles + excess_handles, labels + excess_labels, loc="upper left", ncol=3, frameon=False)

    bottom = axes[1]
    for column in display_drawdowns.columns:
        bottom.plot(
            display_drawdowns.index,
            display_drawdowns[column].mul(100.0),
            color=colors[column],
            linewidth=1.4,
            label=column,
        )
    bottom.set_ylabel("Drawdown (%)")
    bottom.set_xlabel("Date")
    bottom.grid(axis="y", alpha=0.25)
    bottom.spines[["top", "right"]].set_visible(False)
    bottom.legend(loc="lower left", ncol=3, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_pair_yearly_performance(
    yearly_returns: pd.DataFrame,
    yearly_active: pd.DataFrame,
    path: Path,
) -> None:
    nonzero_years = yearly_returns.abs().sum(axis=1).gt(0.0)
    yearly_returns = yearly_returns.loc[nonzero_years]
    yearly_active = yearly_active.reindex(yearly_returns.index)
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.4), sharex=True, gridspec_kw={"height_ratios": [1.35, 1.0]})
    colors = {"MFBT": "#1f77b4", "Origin": "#d55e00", "KOSPI200 BM": "#4d4d4d"}
    yearly_returns.mul(100.0).plot.bar(ax=axes[0], color=[colors[column] for column in yearly_returns.columns], width=0.74)
    axes[0].set_title("Yearly performance")
    axes[0].set_ylabel("Return (%)")
    axes[0].axhline(0.0, color="#7f8c8d", linewidth=0.8)
    axes[0].grid(axis="y", alpha=0.24)
    axes[0].legend(loc="upper left", ncol=3, frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)

    excess_colors = ["#8ecae6", "#ffb703", "#6c757d"]
    yearly_active.mul(100.0).plot.bar(ax=axes[1], color=excess_colors, width=0.74)
    axes[1].set_ylabel("Excess return (%)")
    axes[1].set_xlabel("Year")
    axes[1].axhline(0.0, color="#7f8c8d", linewidth=0.8)
    axes[1].grid(axis="y", alpha=0.24)
    axes[1].legend(loc="upper left", ncol=3, frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_pair_monthly_excess_heatmap(monthly_active: pd.DataFrame, path: Path) -> None:
    display = _pair_excess_display_frame(monthly_active).mul(100.0)
    limit = float(display.abs().max().max()) if not display.empty else 1.0
    if limit == 0.0:
        limit = 1.0
    fig, ax = plt.subplots(figsize=(13, 4.6))
    image = ax.imshow(display.T.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_title("Monthly excess returns")
    ax.set_yticks(range(len(display.columns)))
    ax.set_yticklabels(display.columns)
    step = max(1, len(display.index) // 12)
    ticks = list(range(0, len(display.index), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([display.index[idx].strftime("%Y-%m") for idx in ticks], rotation=45, ha="right")
    for row, column in enumerate(display.columns):
        for col, value in enumerate(display[column].to_numpy(dtype=float)):
            if pd.notna(value) and abs(value) >= max(limit * 0.35, 0.15):
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) > limit * 0.55 else "#202020",
                )
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, pad=0.02, label="Monthly excess return (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_pair_active_weight_sum(frame: pd.DataFrame, path: Path) -> None:
    pct_columns = [column for column in frame.columns if str(column).endswith("_sum_abs_active_weight_pct")]
    display = pd.DataFrame(index=frame.index)
    for column in pct_columns:
        label = "MFBT" if "mfbt" in str(column).lower() else "Origin"
        display[label] = frame[column].astype(float)
    if not display.empty:
        display = display.loc[display.gt(1.0).all(axis=1)]

    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    colors = {"MFBT": "#1f77b4", "Origin": "#d55e00"}
    for column in display.columns:
        ax.plot(display.index, display[column], linewidth=1.9, color=colors[column], label=column)
        ax.scatter(display.index, display[column], s=18, color=colors[column], alpha=0.8)
    ax.set_title("Active weight sum")
    ax.set_ylabel("Sum |active weight| (%)")
    ax.set_xlabel("Month")
    ax.grid(axis="y", alpha=0.24)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_active_weight_sum(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.bar(frame.index, frame["sum_abs_active_weight_pct"], width=22, alpha=0.82, color="#3867d6")
    ax.set_title("Active Weight Sum")
    ax.set_ylabel("Sum |active weight| (%)")
    ax.set_xlabel("Month")
    ax.grid(axis="y", alpha=0.28)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_monthly_excess_heatmap(monthly_active: pd.DataFrame, path: Path) -> None:
    heatmaps = {
        "Gross excess": monthly_excess_heatmap_frame(monthly_active, "Gross excess"),
        "Costed excess": monthly_excess_heatmap_frame(monthly_active, "Costed excess"),
    }
    values = pd.concat([frame.stack() for frame in heatmaps.values()]).dropna()
    limit = float(values.abs().max()) if not values.empty else 1.0
    if limit == 0.0:
        limit = 1.0

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.8), sharex=True, constrained_layout=True)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    image = None
    for ax, (title, frame) in zip(axes, heatmaps.items(), strict=True):
        masked = frame.to_numpy(dtype=float)
        image = ax.imshow(masked, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(f"{title} monthly excess return")
        ax.set_yticks(range(len(frame.index)))
        ax.set_yticklabels([str(year) for year in frame.index])
        ax.set_xticks(range(12))
        ax.set_xticklabels(month_labels)
        ax.set_ylabel("Year")
        for row_idx, year in enumerate(frame.index):
            for col_idx, month in enumerate(frame.columns):
                value = frame.loc[year, month]
                if pd.notna(value):
                    text_color = "white" if abs(value) > limit * 0.55 else "black"
                    ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)
    axes[-1].set_xlabel("Month")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.9, pad=0.02)
        cbar.set_label("Monthly excess return (%)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_cumulative_with_excess_and_drawdown(
    cumulative: pd.DataFrame,
    cumulative_active: pd.DataFrame,
    drawdowns: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]},
    )
    ax_top = axes[0]
    ax_excess = ax_top.twinx()

    fill_colors = {
        "Gross excess": "#7fb3d5",
        "Costed excess": "#f5b7b1",
    }
    for column in cumulative_active.columns:
        series = cumulative_active[column].astype(float)
        ax_excess.fill_between(
            series.index,
            0.0,
            series.to_numpy(),
            color=fill_colors.get(column, "#bdc3c7"),
            alpha=0.24,
            label=column,
            zorder=1,
        )
        ax_excess.plot(
            series.index,
            series.to_numpy(),
            color=fill_colors.get(column, "#7f8c8d"),
            alpha=0.72,
            linewidth=1.0,
            zorder=2,
        )
    ax_excess.axhline(0.0, color="#7f8c8d", linewidth=0.7, alpha=0.65)
    ax_excess.set_ylabel("Cumulative excess return (%)")
    ax_excess.grid(False)

    line_colors = {
        "Gross": "#1f4e79",
        "Costed": "#922b21",
        "KOSPI200 BM": "#2e7d32",
    }
    for column in cumulative.columns:
        ax_top.plot(
            cumulative.index,
            cumulative[column],
            linewidth=1.8,
            label=column,
            color=line_colors.get(column),
            zorder=3,
        )
    ax_top.set_title("Cumulative Return, Cumulative Excess Return, and Drawdown")
    ax_top.set_ylabel("Cumulative return (%)")
    ax_top.grid(alpha=0.25)

    top_handles, top_labels = ax_top.get_legend_handles_labels()
    excess_handles, excess_labels = ax_excess.get_legend_handles_labels()
    ax_top.legend(
        top_handles + excess_handles,
        top_labels + excess_labels,
        loc="upper left",
        ncol=2,
    )

    ax_bottom = axes[1]
    for column in drawdowns.columns:
        ax_bottom.plot(
            drawdowns.index,
            drawdowns[column],
            linewidth=1.5,
            label=column,
            color=line_colors.get(column),
        )
    ax_bottom.set_ylabel("Drawdown (%)")
    ax_bottom.set_xlabel("Date")
    ax_bottom.grid(alpha=0.25)
    ax_bottom.legend(loc="lower left", ncol=3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
