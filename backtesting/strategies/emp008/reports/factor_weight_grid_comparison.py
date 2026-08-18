from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import font_manager


BENCHMARK_COLUMN = "IKS200"
PLOT_LABELS = {
    "candidate_title": "동일 후보별 누적수익률 차이: WICS - WI26",
    "yearly_title": "연도별 누적수익률 차이: WICS - WI26",
    "direction": "양수: WICS 우위 / 음수: WI26 우위",
    "paired_note": "매년 동일한 팩터 가중치 후보끼리 비교",
    "y_axis": "누적수익률 차이 (bp)",
    "x_axis": "날짜",
    "median": "후보 중앙값",
}
DEFAULT_WI26_CSV = Path(
    "backtesting/strategies/emp008/tests/factor_weight_grid_search_wi26/daily_returns.csv"
)
DEFAULT_WICS_CSV = Path(
    "backtesting/strategies/emp008/tests/factor_weight_grid_search_wics/daily_returns.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "backtesting/strategies/emp008/tests/factor_weight_grid_search_comparison"
)


def _configure_korean_font() -> None:
    malgun_path = Path("C:/Windows/Fonts/malgun.ttf")
    if malgun_path.is_file():
        font_manager.fontManager.addfont(malgun_path)
        family = font_manager.FontProperties(fname=malgun_path).get_name()
        plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False


_configure_korean_font()


def _validated_pair(
    wi26_returns: pd.DataFrame,
    wics_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    wi26 = wi26_returns.copy()
    wics = wics_returns.copy()
    wi26.index = pd.DatetimeIndex(pd.to_datetime(wi26.index), name="date")
    wics.index = pd.DatetimeIndex(pd.to_datetime(wics.index), name="date")
    wi26 = wi26.sort_index()
    wics = wics.sort_index()

    if not wi26.index.equals(wics.index):
        raise ValueError("WI26 and WICS return dates differ")
    if BENCHMARK_COLUMN not in wi26 or BENCHMARK_COLUMN not in wics:
        raise ValueError(f"both return panels must contain {BENCHMARK_COLUMN}")
    if not np.allclose(
        wi26[BENCHMARK_COLUMN].to_numpy(dtype=float),
        wics[BENCHMARK_COLUMN].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
        equal_nan=False,
    ):
        raise ValueError("benchmark returns differ between WI26 and WICS")

    candidates = tuple(
        column
        for column in wi26.columns
        if column != BENCHMARK_COLUMN and column in wics.columns
    )
    if not candidates:
        raise ValueError("no common candidate columns")
    paired_columns = (BENCHMARK_COLUMN, *candidates)
    wi26 = wi26.loc[:, list(paired_columns)].astype(float)
    wics = wics.loc[:, list(paired_columns)].astype(float)
    if wi26.isna().any().any() or wics.isna().any().any():
        raise ValueError("paired return panels must not contain missing values")
    return wi26, wics, candidates


def build_cumulative_pair_gap_bp(
    wi26_returns: pd.DataFrame,
    wics_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Return cumulative WICS-minus-WI26 return gaps in basis points."""
    wi26, wics, candidates = _validated_pair(wi26_returns, wics_returns)
    wi26_wealth = (1.0 + wi26.loc[:, list(candidates)]).cumprod()
    wics_wealth = (1.0 + wics.loc[:, list(candidates)]).cumprod()
    gap = wics_wealth.sub(wi26_wealth).mul(10_000.0)
    gap.index.name = "date"
    return gap


def build_yearly_pair_gap_bp(
    wi26_returns: pd.DataFrame,
    wics_returns: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """Return annual-reset cumulative WICS-minus-WI26 return gaps in basis points."""
    wi26, wics, _ = _validated_pair(wi26_returns, wics_returns)
    if (
        len(wi26) > 1
        and wi26.iloc[0].eq(0.0).all()
        and wics.iloc[0].eq(0.0).all()
        and wi26.index[0].year < wi26.index[1].year
    ):
        wi26 = wi26.iloc[1:]
        wics = wics.iloc[1:]

    yearly: dict[int, pd.DataFrame] = {}
    for year, positions in wi26.groupby(wi26.index.year, sort=True).indices.items():
        wi26_year = wi26.iloc[positions]
        wics_year = wics.loc[wi26_year.index]
        gap = build_cumulative_pair_gap_bp(wi26_year, wics_year)
        baseline_date = gap.index[0] - pd.Timedelta(days=1)
        baseline = pd.DataFrame(
            0.0,
            index=pd.DatetimeIndex([baseline_date], name="date"),
            columns=gap.columns,
        )
        yearly[int(year)] = pd.concat([baseline, gap])
    return yearly


def build_year_end_gap_table(
    yearly_frames: Mapping[int, pd.DataFrame],
) -> pd.DataFrame:
    if not yearly_frames:
        raise ValueError("yearly gap frames must not be empty")
    table = pd.DataFrame(
        {year: frame.iloc[-1] for year, frame in yearly_frames.items()}
    ).T
    table.index.name = "year"
    return table


def _plot_candidate_pair_gaps(gap: pd.DataFrame, path: Path) -> None:
    columns = tuple(gap.columns)
    column_count = min(3, len(columns))
    row_count = math.ceil(len(columns) / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.2 * column_count, 3.25 * row_count),
        squeeze=False,
        sharex=True,
    )
    flat_axes = axes.ravel()
    year_starts = gap.groupby(gap.index.year).head(1).index[1:]
    for ax, candidate in zip(flat_axes, columns, strict=False):
        values = gap[candidate]
        ax.plot(values.index, values, color="#243B53", linewidth=1.35)
        ax.fill_between(
            values.index,
            0.0,
            values.to_numpy(),
            where=values.to_numpy() >= 0.0,
            color="#0F9D7A",
            alpha=0.22,
            interpolate=True,
        )
        ax.fill_between(
            values.index,
            0.0,
            values.to_numpy(),
            where=values.to_numpy() < 0.0,
            color="#D95F59",
            alpha=0.22,
            interpolate=True,
        )
        for year_start in year_starts:
            ax.axvline(year_start, color="#CBD5E1", linewidth=0.65, zorder=0)
        ax.axhline(0.0, color="#64748B", linewidth=0.8)
        ax.set_title(candidate, loc="left", fontsize=10.5, fontweight="bold")
        ax.text(
            0.98,
            0.06,
            f"최종 {values.iloc[-1]:+.1f}bp",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="#334E68",
        )
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelrotation=25)
    for ax in flat_axes[len(columns) :]:
        ax.remove()
    fig.suptitle(
        PLOT_LABELS["candidate_title"],
        fontsize=16,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.94,
        PLOT_LABELS["direction"],
        ha="center",
        color="#52606D",
    )
    fig.supxlabel(PLOT_LABELS["x_axis"])
    fig.supylabel(PLOT_LABELS["y_axis"])
    fig.tight_layout(rect=(0.025, 0.03, 0.99, 0.90), h_pad=1.8)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_yearly_pair_gaps(
    yearly_frames: Mapping[int, pd.DataFrame],
    path: Path,
) -> None:
    years = tuple(yearly_frames)
    row_count = math.ceil(len(years) / 2)
    fig, axes = plt.subplots(
        row_count,
        2,
        figsize=(15.5, 3.5 * row_count),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    candidate_order = tuple(next(iter(yearly_frames.values())).columns)
    colors = plt.get_cmap("tab10").colors
    for ax, year in zip(flat_axes, years, strict=False):
        frame = yearly_frames[year]
        for position, candidate in enumerate(candidate_order):
            is_baseline = candidate == "equal_25"
            ax.plot(
                frame.index,
                frame[candidate],
                color="#111827" if is_baseline else colors[position % len(colors)],
                linewidth=2.2 if is_baseline else 1.0,
                alpha=1.0 if is_baseline else 0.72,
                label=candidate,
            )
        median = frame.median(axis=1)
        ax.plot(
            median.index,
            median,
            color="#F59E0B",
            linewidth=2.4,
            linestyle="--",
            label=PLOT_LABELS["median"],
        )
        ax.axhline(0.0, color="#64748B", linewidth=0.8)
        final_median = float(median.iloc[-1])
        winner = (
            "WICS 우위"
            if final_median > 0.0
            else "WI26 우위"
            if final_median < 0.0
            else "동률"
        )
        ax.set_title(
            f"{year}년  |  중앙값 {final_median:+.1f}bp ({winner})",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelrotation=25)
    for ax in flat_axes[len(years) :]:
        ax.remove()
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.suptitle(
        PLOT_LABELS["yearly_title"],
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.957,
        PLOT_LABELS["paired_note"],
        ha="center",
        color="#52606D",
    )
    fig.supxlabel(PLOT_LABELS["x_axis"])
    fig.supylabel(PLOT_LABELS["y_axis"])
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=5,
        frameon=False,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.025, 0.03, 0.99, 0.89), h_pad=2.0)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _read_return_panel(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"return panel not found: {path}")
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


def generate_pair_comparison_report(
    *,
    wi26_csv: Path,
    wics_csv: Path,
    output_dir: Path,
) -> dict[str, Path]:
    wi26_path = Path(wi26_csv)
    wics_path = Path(wics_csv)
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    wi26 = _read_return_panel(wi26_path)
    wics = _read_return_panel(wics_path)
    full_gap = build_cumulative_pair_gap_bp(wi26, wics)
    yearly_frames = build_yearly_pair_gap_bp(wi26, wics)
    year_end = build_year_end_gap_table(yearly_frames)

    outputs = {
        "candidate_pair_cumulative_gap_png": report_dir / "candidate_pair_cumulative_gap.png",
        "yearly_pair_cumulative_gap_png": report_dir / "yearly_pair_cumulative_gap.png",
        "yearly_pair_end_gap_csv": report_dir / "yearly_pair_end_gap_bp.csv",
        "manifest_json": report_dir / "manifest.json",
    }
    _plot_candidate_pair_gaps(full_gap, outputs["candidate_pair_cumulative_gap_png"])
    _plot_yearly_pair_gaps(yearly_frames, outputs["yearly_pair_cumulative_gap_png"])
    year_end.to_csv(outputs["yearly_pair_end_gap_csv"], encoding="utf-8-sig")
    manifest = {
        "definition": (
            "동일 팩터 가중치 후보의 누적수익률을 각각 계산한 뒤 WICS에서 "
            "WI26을 직접 차감한 값이다. 벤치마크 나눗셈은 사용하지 않으며, "
            "양수는 WICS 우위를 뜻한다."
        ),
        "language": "ko",
        "wi26_csv": str(wi26_path.resolve()),
        "wics_csv": str(wics_path.resolve()),
        "start": str(full_gap.index.min().date()),
        "end": str(full_gap.index.max().date()),
        "common_candidates": list(full_gap.columns),
        "outputs": {
            name: str(path.resolve())
            for name, path in outputs.items()
            if name != "manifest_json"
        },
    }
    outputs["manifest_json"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EMP008 WI26·WICS의 동일 그리드 후보를 페어로 비교합니다."
    )
    parser.add_argument("--wi26-csv", type=Path, default=DEFAULT_WI26_CSV)
    parser.add_argument("--wics-csv", type=Path, default=DEFAULT_WICS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = generate_pair_comparison_report(
        wi26_csv=args.wi26_csv,
        wics_csv=args.wics_csv,
        output_dir=args.output_dir,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
