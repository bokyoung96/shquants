from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt

from backtesting.reporting.benchmarks import _load_display_name_maps


DEFAULT_WI26_WORKBOOK = Path(
    "backtesting/strategies/emp008/tests/factor_weight_grid_search_wi26/"
    "deliverables/emp008_rebalance_top_bottom_wi26.xlsx"
)
DEFAULT_WICS_WORKBOOK = Path(
    "backtesting/strategies/emp008/tests/factor_weight_grid_search_wics/"
    "deliverables/emp008_rebalance_top_bottom_wics.xlsx"
)
DEFAULT_OUTPUT_DIR = Path(
    "backtesting/strategies/emp008/tests/factor_weight_top_bottom_comparison"
)


def _configure_korean_font() -> None:
    malgun_path = Path("C:/Windows/Fonts/malgun.ttf")
    if malgun_path.is_file():
        font_manager.fontManager.addfont(malgun_path)
        family = font_manager.FontProperties(fname=malgun_path).get_name()
        plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False


_configure_korean_font()

def build_consensus_scores(
    *,
    top: pd.DataFrame,
    bottom: pd.DataFrame,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """Return date-by-ticker Top-count-minus-Bottom-count scores."""
    required = {"date", "candidate_id", "ticker"}
    for label, frame in (("top", top), ("bottom", bottom)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} rows missing columns: {missing}")

    top_rows = top.loc[:, list(required)].copy()
    bottom_rows = bottom.loc[:, list(required)].copy()
    top_rows["score"] = 1
    bottom_rows["score"] = -1
    combined = pd.concat([top_rows, bottom_rows], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.loc[combined["date"] >= pd.Timestamp(start)]
    if combined.empty:
        raise ValueError("no Top/Bottom rows on or after start")
    scores = combined.pivot_table(
        index="date",
        columns="ticker",
        values="score",
        aggfunc="sum",
        fill_value=0,
    ).astype(int)
    scores.index.name = "date"
    scores.columns.name = "ticker"
    return scores.sort_index().sort_index(axis=1)


def _load_top_bottom_rows(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(path)

    def read_sheet(sheet_name: str) -> pd.DataFrame:
        raw = pd.read_excel(path, sheet_name=sheet_name)
        if raw.shape[1] < 9:
            raise ValueError(f"{path} {sheet_name} sheet has fewer than 9 columns")
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(raw.iloc[:, 0]),
                "candidate_id": raw.iloc[:, 2].astype(str),
                "stock_rank": pd.to_numeric(raw.iloc[:, 7]),
                "ticker": raw.iloc[:, 8].astype(str),
            }
        )
        if frame.isna().any().any():
            raise ValueError(f"{path} {sheet_name} sheet contains missing values")
        return frame

    return read_sheet("Top 10"), read_sheet("Bottom 10")


def _select_display_tickers(
    score_frames: Sequence[pd.DataFrame],
    *,
    each_side: int,
) -> tuple[str, ...]:
    all_tickers = sorted(set().union(*(frame.columns for frame in score_frames)))
    positive = pd.Series(0.0, index=all_tickers)
    negative = pd.Series(0.0, index=all_tickers)
    for frame in score_frames:
        aligned = frame.reindex(columns=all_tickers, fill_value=0)
        positive = positive.add(aligned.clip(lower=0).sum(axis=0), fill_value=0.0)
        negative = negative.add(-aligned.clip(upper=0).sum(axis=0), fill_value=0.0)
    top = positive.sort_values(ascending=False, kind="mergesort").head(each_side).index
    bottom = negative.sort_values(ascending=False, kind="mergesort").head(each_side).index
    return tuple(dict.fromkeys((*top, *bottom)))


def _display_names(
    tickers: Sequence[str],
    stock_names: Mapping[str, str],
) -> list[str]:
    return [f"{stock_names.get(ticker, ticker)}  ({ticker})" for ticker in tickers]


def _aligned_heatmap_frames(
    wi26: pd.DataFrame,
    wics: pd.DataFrame,
    *,
    tickers: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = wi26.index.union(wics.index).sort_values()
    columns = list(tickers)
    return (
        wi26.reindex(index=dates, columns=columns, fill_value=0).T,
        wics.reindex(index=dates, columns=columns, fill_value=0).T,
    )


def _plot_heatmap_pair(
    *,
    wi26: pd.DataFrame,
    wics: pd.DataFrame,
    tickers: Sequence[str],
    stock_names: Mapping[str, str],
    title: str,
    subtitle: str,
    path: Path,
    xlabels: Sequence[str],
    annotate: bool,
) -> None:
    arrays = (wi26.to_numpy(dtype=float), wics.to_numpy(dtype=float))
    vmax = max(float(np.abs(values).max()) for values in arrays)
    vmax = max(vmax, 1.0)
    height = max(7.5, len(tickers) * 0.34)
    fig, axes = plt.subplots(1, 2, figsize=(20, height), sharey=True)
    images = []
    for ax, frame, label in zip(axes, (wi26, wics), ("WI26", "WICS"), strict=True):
        image = ax.imshow(
            frame.to_numpy(dtype=float),
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        images.append(image)
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_yticks(np.arange(len(tickers)))
        ax.set_yticklabels(_display_names(tickers, stock_names), fontsize=8.5)
        tick_count = min(len(xlabels), 13)
        positions = np.unique(
            np.linspace(0, len(xlabels) - 1, tick_count, dtype=int)
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [xlabels[position] for position in positions],
            rotation=45,
            ha="right",
            fontsize=8.5,
        )
        ax.set_xlabel("리밸런싱 시점")
        if annotate:
            for row in range(frame.shape[0]):
                for column in range(frame.shape[1]):
                    value = int(frame.iat[row, column])
                    ax.text(
                        column,
                        row,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(value) > vmax * 0.55 else "#1F2933",
                    )
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.94, subtitle, ha="center", color="#52606D", fontsize=10.5)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.87, bottom=0.17, wspace=0.08)
    colorbar_axis = fig.add_axes((0.34, 0.065, 0.32, 0.025))
    colorbar = fig.colorbar(images[0], cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("후보 합의 점수  (빨강: Top / 파랑: Bottom)")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _annual_scores(frame: pd.DataFrame) -> pd.DataFrame:
    annual = frame.groupby(frame.index.year).sum().T
    annual.columns = annual.columns.astype(str)
    annual.columns.name = "year"
    return annual


def generate_top_bottom_visual_report(
    *,
    wi26_workbook: Path,
    wics_workbook: Path,
    output_dir: Path,
    stock_names: Mapping[str, str],
    start: str = "2020-01-01",
) -> dict[str, Path]:
    wi26_top, wi26_bottom = _load_top_bottom_rows(Path(wi26_workbook))
    wics_top, wics_bottom = _load_top_bottom_rows(Path(wics_workbook))
    wi26_scores = build_consensus_scores(top=wi26_top, bottom=wi26_bottom, start=start)
    wics_scores = build_consensus_scores(top=wics_top, bottom=wics_bottom, start=start)
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "timeline_png": report_dir / "top_bottom_timeline_2020_plus.png",
        "yearly_frequency_png": report_dir / "top_bottom_yearly_frequency.png",
        "detail_2023_png": report_dir / "top_bottom_2023_detail.png",
        "wi26_scores_csv": report_dir / "wi26_consensus_scores.csv",
        "wics_scores_csv": report_dir / "wics_consensus_scores.csv",
        "manifest_json": report_dir / "manifest.json",
    }

    full_tickers = _select_display_tickers((wi26_scores, wics_scores), each_side=15)
    wi26_full, wics_full = _aligned_heatmap_frames(
        wi26_scores,
        wics_scores,
        tickers=full_tickers,
    )
    full_labels = [date.strftime("%Y-%m") for date in wi26_full.columns]
    _plot_heatmap_pair(
        wi26=wi26_full,
        wics=wics_full,
        tickers=full_tickers,
        stock_names=stock_names,
        title="리밸런싱별 Top 10 / Bottom 10 종목 흐름",
        subtitle="9개 팩터 가중치 후보의 합의 점수 · 2020년 이후",
        path=outputs["timeline_png"],
        xlabels=full_labels,
        annotate=False,
    )

    wi26_annual = _annual_scores(wi26_scores)
    wics_annual = _annual_scores(wics_scores)
    annual_tickers = _select_display_tickers(
        (wi26_annual.T, wics_annual.T),
        each_side=15,
    )
    years = wi26_annual.columns.union(wics_annual.columns).sort_values()
    wi26_annual_plot = wi26_annual.reindex(index=annual_tickers, columns=years, fill_value=0)
    wics_annual_plot = wics_annual.reindex(index=annual_tickers, columns=years, fill_value=0)
    _plot_heatmap_pair(
        wi26=wi26_annual_plot,
        wics=wics_annual_plot,
        tickers=annual_tickers,
        stock_names=stock_names,
        title="연도별 Top 10 / Bottom 10 반복 등장 종목",
        subtitle="셀 값은 해당 연도의 Top 등장 횟수 - Bottom 등장 횟수",
        path=outputs["yearly_frequency_png"],
        xlabels=list(years),
        annotate=True,
    )

    wi26_2023 = wi26_scores.loc[wi26_scores.index.year == 2023]
    wics_2023 = wics_scores.loc[wics_scores.index.year == 2023]
    if wi26_2023.empty or wics_2023.empty:
        raise ValueError("both runs must contain 2023 Top/Bottom rows")
    detail_tickers = _select_display_tickers((wi26_2023, wics_2023), each_side=15)
    wi26_detail, wics_detail = _aligned_heatmap_frames(
        wi26_2023,
        wics_2023,
        tickers=detail_tickers,
    )
    detail_labels = [date.strftime("%Y-%m-%d") for date in wi26_detail.columns]
    _plot_heatmap_pair(
        wi26=wi26_detail,
        wics=wics_detail,
        tickers=detail_tickers,
        stock_names=stock_names,
        title="2023년 리밸런싱별 Top 10 / Bottom 10 상세",
        subtitle="성과 차이가 컸던 2023년의 후보 합의 포지션 변화",
        path=outputs["detail_2023_png"],
        xlabels=detail_labels,
        annotate=True,
    )

    wi26_scores.to_csv(outputs["wi26_scores_csv"], encoding="utf-8-sig")
    wics_scores.to_csv(outputs["wics_scores_csv"], encoding="utf-8-sig")
    manifest = {
        "definition": "리밸런싱일·종목별 Top 10 등장 후보 수에서 Bottom 10 등장 후보 수를 차감",
        "start": start,
        "end": str(max(wi26_scores.index.max(), wics_scores.index.max()).date()),
        "candidate_count": int(
            max(wi26_top["candidate_id"].nunique(), wics_top["candidate_id"].nunique())
        ),
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EMP008 WI26·WICS Top/Bottom 종목 히트맵을 생성합니다."
    )
    parser.add_argument("--wi26-workbook", type=Path, default=DEFAULT_WI26_WORKBOOK)
    parser.add_argument("--wics-workbook", type=Path, default=DEFAULT_WICS_WORKBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stock-name-map", type=Path, default=Path("raw/map.xlsx"))
    parser.add_argument("--start", default="2020-01-01")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    _sector_names, stock_names = _load_display_name_maps(args.stock_name_map)
    outputs = generate_top_bottom_visual_report(
        wi26_workbook=args.wi26_workbook,
        wics_workbook=args.wics_workbook,
        output_dir=args.output_dir,
        stock_names=stock_names,
        start=args.start,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
