from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    ROOT / "backtesting/strategies/emp008/tests/size_value_measure_comparison/daily_returns.csv",
    ROOT / "backtesting/strategies/emp008/tests/size_momentum_measure_comparison/daily_returns.csv",
    ROOT / "backtesting/strategies/emp008/tests/size_flow_measure_comparison/daily_returns.csv",
)
DEFAULT_OUTPUT_DIR = ROOT / "backtesting/strategies/emp008/tests/factor_contribution_heatmap"

FACTOR_COLUMNS = {
    "size_value_fcf_tev_return": "Value · FCF/TEV",
    "size_value_dividend_fy0_return": "Value · 배당 FY0",
    "size_value_dividend_ttm_return": "Value · 배당 TTM",
    "size_momentum_12m_return": "Momentum · 12개월",
    "size_momentum_12_1m_return": "Momentum · 12-1개월",
    "size_momentum_high_return": "Momentum · 252일 고가대비",
    "size_earnings_momentum_return": "Momentum · 영업이익 컨센서스",
    "size_retail_flow_return": "Flow · 개인수급",
}


def _configure_korean_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"):
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_daily_returns(paths: tuple[Path, ...]) -> pd.DataFrame:
    frames = [pd.read_csv(path, index_col=0, parse_dates=True).sort_index() for path in paths]
    reference = frames[0].loc[:, ["benchmark_return", "size_only_return"]]

    combined = reference.copy()
    for path, frame in zip(paths, frames, strict=True):
        common = reference.index.intersection(frame.index)
        for shared in ("benchmark_return", "size_only_return"):
            if not np.allclose(
                reference.loc[common, shared],
                frame.loc[common, shared],
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(f"{path}: {shared} does not match the shared baseline")
        for column in FACTOR_COLUMNS:
            if column in frame:
                combined[column] = frame[column]

    missing = [column for column in FACTOR_COLUMNS if column not in combined]
    if missing:
        raise ValueError(f"missing factor return columns: {missing}")

    combined = combined.loc[:, ["benchmark_return", "size_only_return", *FACTOR_COLUMNS]].dropna()
    if len(combined) > 1 and combined.iloc[0].eq(0.0).all() and combined.index[0].year < combined.index[1].year:
        combined = combined.iloc[1:]
    return combined


def period_contribution_bp(daily: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if granularity == "halfyear":
        period_labels = pd.Index(
            [f"{date.year} H{1 if date.month <= 6 else 2}" for date in daily.index],
            name="period",
        )
    elif granularity == "quarterly":
        period_labels = pd.Index(
            [f"{date.year} Q{date.quarter}" for date in daily.index],
            name="period",
        )
    else:
        raise ValueError(f"unsupported granularity: {granularity}")

    period_wealth = (1.0 + daily).groupby(period_labels, sort=False).prod()
    contribution = pd.DataFrame(index=period_wealth.index)
    for column, label in FACTOR_COLUMNS.items():
        contribution[label] = (
            period_wealth[column].divide(period_wealth["size_only_return"]).sub(1.0).mul(10_000.0)
        )
    return contribution.T


def plot_heatmap(values: pd.DataFrame, *, granularity: str, path: Path) -> None:
    _configure_korean_font()
    matrix = values.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(matrix)))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1.0

    cmap = LinearSegmentedColormap.from_list(
        "factor_contribution",
        ("#8F1D35", "#D9878F", "#F7F4EE", "#91CDBB", "#08776C"),
        N=256,
    )
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    width = 15.5 if granularity == "halfyear" else 21.0
    fig, ax = plt.subplots(figsize=(width, 6.6))
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    labels = values.columns.astype(str).tolist()
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(values.index)), labels=values.index)
    ax.tick_params(length=0)
    ax.set_xlabel("평가 구간")
    ax.set_ylabel("팩터 조합")
    title_period = "반기별" if granularity == "halfyear" else "분기별"
    ax.set_title(f"Size-only 대비 팩터 {title_period} 기여도", fontsize=17, fontweight="bold", pad=14)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if abs(value) >= limit * 0.48 else "#1F2937"
            ax.text(
                column,
                row,
                f"{value:+.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9.0 if granularity == "halfyear" else 7.6,
                fontweight="normal",
            )

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axhline(2.5, color="#64748B", linewidth=2.1)
    ax.axhline(6.5, color="#64748B", linewidth=2.1)

    colorbar = fig.colorbar(image, ax=ax, pad=0.018, fraction=0.025)
    colorbar.set_label("Size-only 대비 상대기여 (bp)")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot EMP008 factor contribution heatmaps versus Size-only.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    daily = load_daily_returns(DEFAULT_INPUTS)

    for granularity in ("halfyear", "quarterly"):
        contribution = period_contribution_bp(daily, granularity)
        contribution.to_csv(output_dir / f"factor_contribution_{granularity}_bp.csv", encoding="utf-8-sig")
        plot_heatmap(
            contribution,
            granularity=granularity,
            path=output_dir / f"factor_contribution_heatmap_{granularity}.png",
        )


if __name__ == "__main__":
    main()
