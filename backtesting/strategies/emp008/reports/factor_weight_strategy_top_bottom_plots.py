from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd


def _configure_korean_font() -> None:
    for font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if font in {item.name for item in __import__("matplotlib").font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_top_bottom_rows(path: Path, start: str = "2020-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    def read(sheet_name: str) -> pd.DataFrame:
        raw = pd.read_excel(path, sheet_name=sheet_name)
        if raw.shape[1] < 10:
            raise ValueError(f"{path.name} {sheet_name} 시트의 열이 부족합니다")
        rows = raw.iloc[:, [0, 2, 7, 8, 9]].copy()
        rows.columns = ["date", "candidate_id", "rank", "ticker", "stock_name"]
        rows["date"] = pd.to_datetime(rows["date"])
        rows = rows.loc[rows["date"] >= pd.Timestamp(start)].copy()
        rows["candidate_id"] = rows["candidate_id"].astype(str)
        rows["ticker"] = rows["ticker"].astype(str)
        return rows.sort_values(["candidate_id", "date", "rank"]).reset_index(drop=True)

    top, bottom = read("Top 10"), read("Bottom 10")
    if top.empty or bottom.empty:
        raise ValueError(f"{start} 이후 Top/Bottom 데이터가 없습니다: {path}")
    return top, bottom


def _candidate_set(frame: pd.DataFrame) -> set[str]:
    return set(frame["candidate_id"].astype(str).unique())


def validate_paired_rows(
    wi26_top: pd.DataFrame,
    wi26_bottom: pd.DataFrame,
    wics_top: pd.DataFrame,
    wics_bottom: pd.DataFrame,
) -> None:
    named_frames = {
        "WI26 Top": wi26_top,
        "WI26 Bottom": wi26_bottom,
        "WICS Top": wics_top,
        "WICS Bottom": wics_bottom,
    }
    strategy_sets = {name: _candidate_set(frame) for name, frame in named_frames.items()}
    expected = strategy_sets["WI26 Top"]
    if any(strategies != expected for strategies in strategy_sets.values()):
        detail = "; ".join(f"{name}={sorted(values)}" for name, values in strategy_sets.items())
        raise ValueError(f"전략 집합 불일치: {detail}")

    for candidate in sorted(expected):
        date_sets = {
            name: set(pd.to_datetime(frame.loc[frame["candidate_id"].eq(candidate), "date"]))
            for name, frame in named_frames.items()
        }
        reference = date_sets["WI26 Top"]
        if any(dates != reference for dates in date_sets.values()):
            detail = "; ".join(
                f"{name}={[str(date.date()) for date in sorted(values)]}"
                for name, values in date_sets.items()
            )
            raise ValueError(f"리밸런싱일 불일치 ({candidate}): {detail}")

    for name, frame in named_frames.items():
        counts = frame.groupby(["candidate_id", "date"])["ticker"].nunique()
        invalid = counts.loc[counts.ne(10)]
        if not invalid.empty:
            candidate, date = invalid.index[0]
            raise ValueError(
                f"{name}의 {candidate} {pd.Timestamp(date).date()} 종목 수가 "
                f"10개가 아닙니다: {int(invalid.iloc[0])}개"
            )


def build_strategy_state_matrix(
    top: pd.DataFrame,
    bottom: pd.DataFrame,
    *,
    candidate_id: str,
) -> pd.DataFrame:
    top_rows = top.loc[top["candidate_id"].eq(candidate_id), ["date", "ticker"]].assign(state=1)
    bottom_rows = bottom.loc[bottom["candidate_id"].eq(candidate_id), ["date", "ticker"]].assign(state=-1)
    selected = pd.concat([top_rows, bottom_rows], ignore_index=True)
    if selected.empty:
        raise ValueError(f"전략 데이터가 없습니다: {candidate_id}")
    matrix = selected.pivot_table(
        index="date", columns="ticker", values="state", aggfunc="sum", fill_value=0
    ).astype("int8")
    matrix.index = pd.to_datetime(matrix.index)
    matrix.index.name = "date"
    matrix.columns.name = "ticker"
    return matrix.sort_index().sort_index(axis=1)


def calculate_replacement_counts(
    wi26_top: pd.DataFrame,
    wi26_bottom: pd.DataFrame,
    wics_top: pd.DataFrame,
    wics_bottom: pd.DataFrame,
) -> pd.DataFrame:
    validate_paired_rows(wi26_top, wi26_bottom, wics_top, wics_bottom)
    records: list[dict[str, object]] = []
    for candidate in sorted(_candidate_set(wi26_top)):
        candidate_dates = sorted(
            pd.to_datetime(wi26_top.loc[wi26_top["candidate_id"].eq(candidate), "date"].unique())
        )
        for date in candidate_dates:
            def tickers(frame: pd.DataFrame) -> set[str]:
                mask = frame["candidate_id"].eq(candidate) & frame["date"].eq(date)
                return set(frame.loc[mask, "ticker"].astype(str))

            top_replacements = 10 - len(tickers(wi26_top) & tickers(wics_top))
            bottom_replacements = 10 - len(tickers(wi26_bottom) & tickers(wics_bottom))
            records.append(
                {
                    "candidate_id": candidate,
                    "date": pd.Timestamp(date),
                    "top_replacements": top_replacements,
                    "bottom_replacements": bottom_replacements,
                    "total_replacements": top_replacements + bottom_replacements,
                }
            )
    return pd.DataFrame(records).set_index(["candidate_id", "date"]).sort_index()


def _strategy_label(candidate_id: str) -> str:
    if candidate_id == "equal_25":
        return "equal_25 — Size 25% · 모멘텀 25% · 이익 25% · 가치 25%"
    parts = candidate_id.split("_")
    if len(parts) != 4:
        return candidate_id
    try:
        values = {part[0]: int(part[1:]) for part in parts}
        return (
            f"{candidate_id} — Size {values['s']}% · 모멘텀 {values['m']}% · "
            f"이익 {values['e']}% · 가치 {values['v']}%"
        )
    except (KeyError, ValueError):
        return candidate_id


def _year_ticks(dates: Iterable[pd.Timestamp]) -> tuple[list[int], list[str], list[float]]:
    index = pd.DatetimeIndex(dates)
    positions: list[int] = []
    labels: list[str] = []
    boundaries: list[float] = []
    for year in sorted(index.year.unique()):
        year_positions = np.flatnonzero(index.year == year)
        positions.append(int(year_positions[len(year_positions) // 2]))
        labels.append(str(year))
        if year_positions[0] > 0:
            boundaries.append(float(year_positions[0]) - 0.5)
    return positions, labels, boundaries


def _plot_summary(counts: pd.DataFrame, path: Path) -> None:
    matrix = counts["total_replacements"].unstack("date")
    figure_height = max(5.8, len(matrix) * 0.72)
    fig, ax = plt.subplots(figsize=(18, figure_height))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Reds", vmin=0, vmax=20)
    ax.set_yticks(range(len(matrix.index)), [_strategy_label(value) for value in matrix.index], fontsize=9)
    positions, labels, boundaries = _year_ticks(matrix.columns)
    ax.set_xticks(positions, labels)
    for boundary in boundaries:
        ax.axvline(boundary, color="#555555", linewidth=0.7, alpha=0.65)
    ax.set_title("WI26와 WICS의 전략별 Top/Bottom 교체 종목 수", fontsize=16, pad=20)
    ax.set_xlabel("리밸런싱 연도")
    ax.set_ylabel("팩터 가중치 전략")
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.14, fraction=0.045)
    colorbar.set_label("교체 종목 수 (Top 0~10 + Bottom 0~10)")
    fig.subplots_adjust(left=0.33, right=0.98, top=0.90, bottom=0.20)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _select_detail_tickers(wi26: pd.DataFrame, wics: pd.DataFrame, each_side: int = 15) -> list[str]:
    columns = sorted(set(wi26.columns) | set(wics.columns))
    left = wi26.reindex(columns=columns, fill_value=0)
    right = wics.reindex(columns=columns, fill_value=0)
    positive = ((left > 0).sum() + (right > 0).sum()).sort_values(ascending=False, kind="stable")
    negative = ((left < 0).sum() + (right < 0).sum()).sort_values(ascending=False, kind="stable")

    def ordered(series: pd.Series) -> list[str]:
        return sorted(series.index, key=lambda ticker: (-int(series[ticker]), str(ticker)))

    top = [ticker for ticker in ordered(positive) if positive[ticker] > 0][:each_side]
    bottom = [ticker for ticker in ordered(negative) if negative[ticker] > 0 and ticker not in top][:each_side]
    return top + bottom


def _plot_strategy_detail(
    wi26: pd.DataFrame,
    wics: pd.DataFrame,
    *,
    candidate_id: str,
    stock_names: dict[str, str],
    path: Path,
) -> None:
    tickers = _select_detail_tickers(wi26, wics)
    dates = wi26.index.union(wics.index).sort_values()
    frames = [
        wi26.reindex(index=dates, columns=tickers, fill_value=0).T,
        wics.reindex(index=dates, columns=tickers, fill_value=0).T,
    ]
    cmap = ListedColormap(["#2166ac", "#f7f7f7", "#b2182b"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(19, max(8.5, len(tickers) * 0.34)), sharey=True)
    images = []
    for ax, frame, panel_title in zip(axes, frames, ("WI26", "WICS")):
        image = ax.imshow(frame.to_numpy(), aspect="auto", cmap=cmap, norm=norm)
        images.append(image)
        positions, labels, boundaries = _year_ticks(frame.columns)
        ax.set_xticks(positions, labels)
        ax.set_title(panel_title, fontsize=13)
        ax.set_xlabel("리밸런싱 연도")
        for boundary in boundaries:
            ax.axvline(boundary, color="#777777", linewidth=0.6, alpha=0.65)
    axes[0].set_yticks(
        range(len(tickers)),
        [f"{stock_names.get(ticker, ticker)} ({ticker})" for ticker in tickers],
        fontsize=8,
    )
    axes[0].set_ylabel("반복 등장 종목")
    fig.suptitle(f"전략별 Top 10 / Bottom 10 종목 흐름\n{_strategy_label(candidate_id)}", fontsize=15, y=0.98)
    colorbar_axis = fig.add_axes((0.38, 0.055, 0.28, 0.025))
    colorbar = fig.colorbar(images[0], cax=colorbar_axis, orientation="horizontal", ticks=[-1, 0, 1])
    colorbar.ax.set_xticklabels(["Bottom 10", "미등장", "Top 10"])
    fig.subplots_adjust(left=0.20, right=0.98, top=0.90, bottom=0.13, wspace=0.06)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _stock_name_map(*frames: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for frame in frames:
        for ticker, name in frame[["ticker", "stock_name"]].drop_duplicates().itertuples(index=False):
            if pd.notna(name) and str(name).strip():
                result[str(ticker)] = str(name)
    return result


def generate_strategy_top_bottom_report(
    *,
    wi26_workbook: Path,
    wics_workbook: Path,
    output_dir: Path,
    start: str = "2020-01-01",
) -> dict[str, object]:
    _configure_korean_font()
    wi26_top, wi26_bottom = load_top_bottom_rows(Path(wi26_workbook), start)
    wics_top, wics_bottom = load_top_bottom_rows(Path(wics_workbook), start)
    counts = calculate_replacement_counts(wi26_top, wi26_bottom, wics_top, wics_bottom)
    stock_names = _stock_name_map(wi26_top, wi26_bottom, wics_top, wics_bottom)
    output_dir = Path(output_dir)
    detail_dir = output_dir / "details"
    state_dir = output_dir / "state_matrices"
    detail_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    summary_png = output_dir / "strategy_replacement_timeline_2020_plus.png"
    replacement_csv = output_dir / "strategy_replacement_counts.csv"
    manifest_json = output_dir / "manifest.json"
    _plot_summary(counts, summary_png)
    counts.reset_index().to_csv(replacement_csv, index=False, encoding="utf-8-sig")

    detail_pngs: list[Path] = []
    state_csvs: list[Path] = []
    strategies = sorted(_candidate_set(wi26_top))
    for candidate in strategies:
        wi26_state = build_strategy_state_matrix(wi26_top, wi26_bottom, candidate_id=candidate)
        wics_state = build_strategy_state_matrix(wics_top, wics_bottom, candidate_id=candidate)
        wi26_csv = state_dir / f"{candidate}_wi26.csv"
        wics_csv = state_dir / f"{candidate}_wics.csv"
        wi26_state.to_csv(wi26_csv, encoding="utf-8-sig")
        wics_state.to_csv(wics_csv, encoding="utf-8-sig")
        state_csvs.extend((wi26_csv, wics_csv))
        detail_png = detail_dir / f"{candidate}.png"
        _plot_strategy_detail(
            wi26_state,
            wics_state,
            candidate_id=candidate,
            stock_names=stock_names,
            path=detail_png,
        )
        detail_pngs.append(detail_png)

    dates = counts.index.get_level_values("date")
    manifest = {
        "definition": "Top 교체 수 + Bottom 교체 수 (전략별, 후보 간 합산 없음)",
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "strategy_count": len(strategies),
        "strategies": {candidate: _strategy_label(candidate) for candidate in strategies},
        "outputs": {
            "summary_png": str(summary_png.resolve()),
            "replacement_csv": str(replacement_csv.resolve()),
            "detail_pngs": [str(path.resolve()) for path in detail_pngs],
            "state_csvs": [str(path.resolve()) for path in state_csvs],
        },
    }
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "summary_png": summary_png,
        "replacement_csv": replacement_csv,
        "detail_pngs": detail_pngs,
        "state_csvs": state_csvs,
        "manifest_json": manifest_json,
    }


def _parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[4]
    test_root = repo_root / "backtesting" / "strategies" / "emp008" / "tests"
    parser = argparse.ArgumentParser(description="EMP008 WI26/WICS 전략별 Top/Bottom 비교")
    parser.add_argument(
        "--wi26-workbook",
        type=Path,
        default=test_root / "factor_weight_grid_search_wi26" / "deliverables" / "emp008_rebalance_top_bottom_wi26.xlsx",
    )
    parser.add_argument(
        "--wics-workbook",
        type=Path,
        default=test_root / "factor_weight_grid_search_wics" / "deliverables" / "emp008_rebalance_top_bottom_wics.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=test_root / "factor_weight_top_bottom_comparison" / "by_strategy",
    )
    parser.add_argument("--start", default="2020-01-01")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    outputs = generate_strategy_top_bottom_report(
        wi26_workbook=args.wi26_workbook,
        wics_workbook=args.wics_workbook,
        output_dir=args.output_dir,
        start=args.start,
    )
    print(
        json.dumps(
            {
                "summary_png": str(outputs["summary_png"]),
                "replacement_csv": str(outputs["replacement_csv"]),
                "detail_png_count": len(outputs["detail_pngs"]),
                "state_csv_count": len(outputs["state_csvs"]),
                "manifest_json": str(outputs["manifest_json"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
