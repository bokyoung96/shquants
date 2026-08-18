from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.patches import Rectangle

from backtesting.strategies.emp008.reports.comparison import performance_metrics


DEFAULT_SUMMARY_CSV = Path(
    "backtesting/strategies/emp008/tests/factor_weight_grid_search_wics/performance_summary.csv"
)
DEFAULT_OUTPUT_PNG = Path(
    "backtesting/strategies/emp008/tests/factor_weight_grid_search_wics/performance_summary_heatmap.png"
)
DEFAULT_CURVE_OUTPUT_PNG = Path(
    "backtesting/strategies/emp008/tests/factor_weight_grid_search_wics/cumulative_performance_mdd.png"
)
DEFAULT_YEARLY_OUTPUT_PNG = Path(
    "backtesting/strategies/emp008/tests/factor_weight_grid_search_wics/yearly_cumulative_excess.png"
)

WEIGHT_COLUMNS = (
    "ln_market_cap_pct",
    "momentum_12m_pct",
    "earnings_momentum_pct",
    "value_pct",
)
WEIGHT_LABELS = ("사이즈", "12개월 모멘텀", "영업이익 컨센", "FCF/TEV")
METRIC_COLUMNS = (
    "cagr_pct",
    "total_return_pct",
    "cumulative_excess_pct_point",
    "information_ratio",
    "max_drawdown_pct",
)
METRIC_LABELS = ("CAGR", "누적수익률", "누적 초과수익", "IR", "MDD")


def relative_performance_scores(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = summary.loc[:, METRIC_COLUMNS].astype(float)
    scores = pd.DataFrame(index=metrics.index, columns=metrics.columns, dtype=float)
    for column in metrics:
        ranks = metrics[column].rank(method="average", ascending=True)
        denominator = max(len(ranks) - 1, 1)
        scores[column] = ranks.sub(1.0).div(denominator)
    return scores


def write_performance_summary_heatmap(
    *,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    output_png: Path = DEFAULT_OUTPUT_PNG,
) -> Path:
    summary = pd.read_csv(summary_csv)
    _validate_summary(summary)
    daily = _read_performance_daily(summary_csv.parent / "daily_returns.csv")
    display_metrics = _build_display_metrics(summary, daily)
    order = display_metrics.sort_values(
        ["cumulative_excess_pct_point", "information_ratio"],
        ascending=False,
        kind="mergesort",
    ).index
    summary = summary.loc[order].reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    display_metrics = display_metrics.loc[order].reset_index(drop=True)
    weights = summary.loc[:, WEIGHT_COLUMNS].astype(float)
    scores = relative_performance_scores(display_metrics)
    period = _performance_period(daily)
    benchmark_metrics = performance_metrics(daily["IKS200"], periods_per_year=252)

    _configure_font()
    figure, (weight_axis, metric_axis) = plt.subplots(
        ncols=2,
        figsize=(18, 8.7),
        gridspec_kw={"width_ratios": (1.0, 1.65), "wspace": 0.06},
    )
    figure.patch.set_facecolor("#F8FAFC")
    _draw_weight_panel(weight_axis, summary, weights)
    _draw_metric_panel(metric_axis, summary, display_metrics, scores, benchmark_metrics)
    figure.suptitle(
        f"EMP008 WICS 팩터 가중치 그리드서치 성과 요약 ({period})",
        x=0.5,
        y=0.975,
        fontsize=20,
        fontweight="bold",
        color="#0F172A",
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_png, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor()
    )
    plt.close(figure)
    return output_png


def write_cumulative_performance_mdd_chart(
    *,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    output_png: Path = DEFAULT_CURVE_OUTPUT_PNG,
) -> Path:
    summary = pd.read_csv(summary_csv)
    _validate_summary(summary)
    daily = _read_performance_daily(summary_csv.parent / "daily_returns.csv")
    display_metrics = _build_display_metrics(summary, daily)
    order = display_metrics.sort_values(
        ["cumulative_excess_pct_point", "information_ratio"],
        ascending=False,
        kind="mergesort",
    ).index
    summary = summary.loc[order].reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)

    candidate_ids = summary["candidate_id"].astype(str).tolist()
    columns = [*candidate_ids, "IKS200"]
    wealth = (1.0 + daily.loc[:, columns]).cumprod()
    cumulative = wealth.sub(1.0).mul(100.0)
    cumulative_vs_benchmark = cumulative.loc[:, candidate_ids].sub(
        cumulative["IKS200"], axis=0
    )
    drawdown = wealth.div(wealth.cummax()).sub(1.0).mul(100.0)

    _configure_font()
    figure, (return_axis, mdd_axis) = plt.subplots(
        nrows=2,
        figsize=(17.5, 10.5),
        sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.0), "hspace": 0.08},
    )
    figure.patch.set_facecolor("#F8FAFC")
    color_map = plt.get_cmap("tab10")
    row_labels = _row_labels(summary)
    for position, (candidate_id, label) in enumerate(
        zip(candidate_ids, row_labels, strict=True)
    ):
        color = color_map(position)
        width = 2.4 if position == 0 else 1.35
        alpha = 1.0 if position < 3 else 0.72
        return_axis.plot(
            cumulative.index,
            cumulative_vs_benchmark[candidate_id],
            label=label,
            color=color,
            linewidth=width,
            alpha=alpha,
        )
        mdd_axis.plot(
            drawdown.index,
            drawdown[candidate_id],
            color=color,
            linewidth=width,
            alpha=alpha,
        )
    return_axis.axhline(
        0.0,
        label="KOSPI200 BM 기준",
        color="#111827",
        linewidth=2.0,
        linestyle="--",
    )
    mdd_axis.plot(
        drawdown.index,
        drawdown["IKS200"],
        color="#111827",
        linewidth=2.4,
        linestyle="--",
    )

    period = _performance_period(daily)
    return_axis.set_title(
        f"EMP008 WICS 팩터 가중치별 BM 대비 누적수익 ({period})",
        fontsize=18,
        fontweight="bold",
        pad=16,
        color="#0F172A",
    )
    return_axis.set_ylabel("포트 누적수익률 - BM 누적수익률 (%p)")
    mdd_axis.set_ylabel("MDD 경로 (%)")
    mdd_axis.set_xlabel("날짜")
    for axis in (return_axis, mdd_axis):
        axis.set_facecolor("white")
        axis.grid(True, alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    return_axis.legend(
        loc="upper left",
        ncols=2,
        fontsize=9,
        frameon=False,
    )
    mdd_axis.axhline(0.0, color="#64748B", linewidth=0.8)
    figure.autofmt_xdate()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_png, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor()
    )
    plt.close(figure)
    return output_png


def write_yearly_cumulative_excess_chart(
    *,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    output_png: Path = DEFAULT_YEARLY_OUTPUT_PNG,
) -> Path:
    summary = pd.read_csv(summary_csv)
    _validate_summary(summary)
    daily = _read_performance_daily(summary_csv.parent / "daily_returns.csv")
    display_metrics = _build_display_metrics(summary, daily)
    order = display_metrics.sort_values(
        ["cumulative_excess_pct_point", "information_ratio"],
        ascending=False,
        kind="mergesort",
    ).index
    summary = summary.loc[order].reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    candidate_ids = summary["candidate_id"].astype(str).tolist()
    yearly = _yearly_cumulative_excess(daily, candidate_ids)

    _configure_font()
    figure, axes = plt.subplots(nrows=4, ncols=2, figsize=(17.5, 15.5))
    figure.patch.set_facecolor("#F8FAFC")
    flat_axes = list(axes.flat)
    color_map = plt.get_cmap("tab10")
    row_labels = _row_labels(summary)
    legend_handles = []
    legend_labels = []
    for panel_axis, (year, frame) in zip(flat_axes, yearly.items(), strict=False):
        for position, (candidate_id, label) in enumerate(
            zip(candidate_ids, row_labels, strict=True)
        ):
            (line,) = panel_axis.plot(
                frame.index,
                frame[candidate_id],
                color=color_map(position),
                linewidth=2.1 if position == 0 else 1.2,
                alpha=1.0 if position < 3 else 0.72,
            )
            if year == min(yearly):
                legend_handles.append(line)
                legend_labels.append(label)
        benchmark_line = panel_axis.axhline(
            0.0,
            color="#111827",
            linewidth=1.6,
            linestyle="--",
        )
        if year == min(yearly):
            legend_handles.append(benchmark_line)
            legend_labels.append("KOSPI200 BM 기준")
        panel_axis.set_title(str(year), fontsize=13, fontweight="bold")
        panel_axis.set_ylabel("누적 초과수익 (%p)")
        panel_axis.set_facecolor("white")
        panel_axis.grid(True, alpha=0.22)
        panel_axis.spines[["top", "right"]].set_visible(False)
        panel_axis.xaxis.set_major_locator(MonthLocator(interval=2))
        panel_axis.xaxis.set_major_formatter(DateFormatter("%m월"))
        panel_axis.tick_params(axis="x", rotation=0)
    for unused_axis in flat_axes[len(yearly) :]:
        unused_axis.set_visible(False)

    figure.suptitle(
        "EMP008 WICS 연도별 BM 대비 누적 초과수익",
        fontsize=20,
        fontweight="bold",
        y=0.995,
        color="#0F172A",
    )
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncols=5,
        fontsize=9,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), h_pad=2.0, w_pad=1.5)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_png, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor()
    )
    plt.close(figure)
    return output_png


def _yearly_cumulative_excess(
    daily: pd.DataFrame, candidate_ids: list[str]
) -> dict[int, pd.DataFrame]:
    yearly: dict[int, pd.DataFrame] = {}
    columns = [*candidate_ids, "IKS200"]
    for year, period in daily.loc[:, columns].groupby(daily.index.year):
        wealth = (1.0 + period).cumprod()
        cumulative = wealth.sub(1.0).mul(100.0)
        yearly[int(year)] = cumulative.loc[:, candidate_ids].sub(
            cumulative["IKS200"], axis=0
        )
    return yearly


def _draw_weight_panel(
    axis: Axes, summary: pd.DataFrame, weights: pd.DataFrame
) -> None:
    display_weights = np.vstack(
        [weights.to_numpy(), np.full((1, len(WEIGHT_COLUMNS)), np.nan)]
    )
    color_map = plt.get_cmap("Blues").copy()
    color_map.set_bad("#E2E8F0")
    image = axis.imshow(
        np.ma.masked_invalid(display_weights),
        cmap=color_map,
        vmin=0.0,
        vmax=50.0,
        aspect="auto",
    )
    axis.set_title(
        "팩터 가중치 (%)", fontsize=13, fontweight="bold", pad=14, color="#0F172A"
    )
    axis.set_xticks(np.arange(len(WEIGHT_LABELS)), labels=WEIGHT_LABELS)
    axis.set_yticks(
        np.arange(len(summary) + 1), labels=[*_row_labels(summary), "KOSPI200 BM"]
    )
    axis.tick_params(axis="x", labelrotation=0, labelsize=10)
    axis.tick_params(axis="y", labelsize=10, length=0)
    for row in range(len(weights)):
        for column in range(len(WEIGHT_COLUMNS)):
            value = float(weights.iat[row, column])
            color = "white" if value >= 37.5 else "#0F172A"
            axis.text(
                column,
                row,
                f"{value:.0f}%",
                ha="center",
                va="center",
                fontsize=10,
                color=color,
            )
    for column in range(len(WEIGHT_COLUMNS)):
        axis.text(column, len(summary), "-", ha="center", va="center", fontsize=10)
    _style_axis(axis, rows=len(summary) + 1, columns=len(WEIGHT_COLUMNS))
    _highlight_winner(axis, columns=len(WEIGHT_COLUMNS))
    image.set_clim(0.0, 50.0)


def _draw_metric_panel(
    axis: Axes,
    summary: pd.DataFrame,
    display_metrics: pd.DataFrame,
    scores: pd.DataFrame,
    benchmark_metrics: dict[str, float],
) -> None:
    display_scores = np.vstack(
        [scores.to_numpy(), np.full((1, len(METRIC_COLUMNS)), np.nan)]
    )
    color_map = plt.get_cmap("RdYlGn").copy()
    color_map.set_bad("#E2E8F0")
    axis.imshow(
        np.ma.masked_invalid(display_scores),
        cmap=color_map,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    axis.set_title("성과 지표", fontsize=13, fontweight="bold", pad=14, color="#0F172A")
    axis.set_xticks(np.arange(len(METRIC_LABELS)), labels=METRIC_LABELS)
    axis.set_yticks(np.arange(len(summary)), labels=())
    axis.tick_params(axis="x", labelrotation=0, labelsize=10)
    axis.tick_params(axis="y", length=0)
    for row in range(len(summary)):
        annotations = (
            f"{display_metrics.at[row, 'cagr_pct']:.2f}%",
            f"{display_metrics.at[row, 'total_return_pct']:.2f}%",
            f"{display_metrics.at[row, 'cumulative_excess_pct_point']:+.2f}%p",
            f"{display_metrics.at[row, 'information_ratio']:.3f}",
            f"{display_metrics.at[row, 'max_drawdown_pct']:.2f}%",
        )
        for column, label in enumerate(annotations):
            score = float(scores.iat[row, column])
            color = "white" if score <= 0.15 or score >= 0.85 else "#0F172A"
            axis.text(
                column, row, label, ha="center", va="center", fontsize=9.5, color=color
            )
    benchmark_annotations = (
        f"{benchmark_metrics['cagr_pct']:.2f}%",
        f"{benchmark_metrics['total_return_pct']:.2f}%",
        "+0.00%p",
        "-",
        f"{benchmark_metrics['max_drawdown_pct']:.2f}%",
    )
    for column, label in enumerate(benchmark_annotations):
        axis.text(
            column,
            len(summary),
            label,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#0F172A",
        )
    _style_axis(axis, rows=len(summary) + 1, columns=len(METRIC_COLUMNS))
    _highlight_winner(axis, columns=len(METRIC_COLUMNS))


def _style_axis(axis: Axes, *, rows: int, columns: int) -> None:
    axis.set_facecolor("white")
    axis.set_xticks(np.arange(-0.5, columns, 1.0), minor=True)
    axis.set_yticks(np.arange(-0.5, rows, 1.0), minor=True)
    axis.grid(which="minor", color="white", linewidth=2.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _highlight_winner(axis: Axes, *, columns: int) -> None:
    axis.add_patch(
        Rectangle(
            (-0.49, -0.49),
            columns - 0.02,
            0.98,
            fill=False,
            edgecolor="#F59E0B",
            linewidth=3.0,
            clip_on=False,
        )
    )


def _row_labels(summary: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for row in summary.itertuples(index=False):
        if row.candidate_id == "equal_25":
            label = "동일가중 25/25/25/25"
        else:
            label = (
                f"{row.ln_market_cap_pct:.0f}/{row.momentum_12m_pct:.0f}/"
                f"{row.earnings_momentum_pct:.0f}/{row.value_pct:.0f}"
            )
        labels.append(f"#{int(row.rank)}  {label}")
    return labels


def _validate_summary(summary: pd.DataFrame) -> None:
    required = {"rank", "candidate_id", *WEIGHT_COLUMNS}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"missing summary columns: {missing}")
    if summary.empty:
        raise ValueError("summary must not be empty")


def _read_performance_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    daily = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    if daily.empty:
        raise ValueError("daily returns must not be empty")
    daily = daily.astype(float)
    if len(daily) > 1 and daily.iloc[0].eq(0.0).all():
        daily = daily.iloc[1:]
    return daily


def _performance_period(daily: pd.DataFrame) -> str:
    start = pd.Timestamp(daily.index[0]).date().isoformat()
    end = pd.Timestamp(daily.index[-1]).date().isoformat()
    return f"{start} ~ {end}"


def _build_display_metrics(summary: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if "IKS200" not in daily.columns:
        raise ValueError("daily returns must include IKS200")
    benchmark = daily["IKS200"]
    benchmark_total_return_pct = performance_metrics(benchmark, periods_per_year=252)[
        "total_return_pct"
    ]
    rows: list[dict[str, float]] = []
    for candidate_id in summary["candidate_id"].astype(str):
        if candidate_id not in daily.columns:
            raise ValueError(f"daily returns missing candidate: {candidate_id}")
        returns = daily[candidate_id]
        metrics = performance_metrics(returns, periods_per_year=252)
        active = returns.sub(benchmark)
        tracking_error = float(active.std(ddof=1))
        information_ratio = (
            float(active.mean() / tracking_error * np.sqrt(252.0))
            if tracking_error > 0.0
            else 0.0
        )
        rows.append(
            {
                "cagr_pct": metrics["cagr_pct"],
                "total_return_pct": metrics["total_return_pct"],
                "cumulative_excess_pct_point": (
                    metrics["total_return_pct"] - benchmark_total_return_pct
                ),
                "information_ratio": information_ratio,
                "max_drawdown_pct": metrics["max_drawdown_pct"],
            }
        )
    return pd.DataFrame(rows, index=summary.index, columns=METRIC_COLUMNS)


def _configure_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    if "Malgun Gothic" in available:
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the EMP008 factor-weight performance heatmap."
    )
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument(
        "--curve-output-png", type=Path, default=DEFAULT_CURVE_OUTPUT_PNG
    )
    parser.add_argument(
        "--yearly-output-png", type=Path, default=DEFAULT_YEARLY_OUTPUT_PNG
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    heatmap_path = write_performance_summary_heatmap(
        summary_csv=args.summary_csv, output_png=args.output_png
    )
    curve_path = write_cumulative_performance_mdd_chart(
        summary_csv=args.summary_csv, output_png=args.curve_output_png
    )
    yearly_path = write_yearly_cumulative_excess_chart(
        summary_csv=args.summary_csv, output_png=args.yearly_output_png
    )
    print(heatmap_path)
    print(curve_path)
    print(yearly_path)


if __name__ == "__main__":
    main()
