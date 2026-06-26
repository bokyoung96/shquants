from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etc.next_day_down_event_study import (
    DEFAULT_CODE,
    DEFAULT_THRESHOLDS_PCT,
    build_iks200_daily_frame,
    build_intraday_next_day_paths,
    build_next_day_reactions,
    build_yearly_threshold_matrix,
    down_bucket_label,
    mark_down_day_events,
    summarize_intraday_paths,
    summarize_overall,
    summarize_yearly,
)


DEFAULT_INPUT = Path("parquet/qw_BM.parquet")
DEFAULT_MINUTE_INPUT = Path("parquet/KOSPI200_1m.parquet")
DEFAULT_SP500_INPUT = Path("parquet/SP500_1d.parquet")
DEFAULT_OUT_DIR = Path("results/next_day_down_event_study")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KOSPI200 next-day event study after down days.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--minute-input", type=Path, default=DEFAULT_MINUTE_INPUT)
    parser.add_argument("--sp500-input", type=Path, default=DEFAULT_SP500_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--code", default=DEFAULT_CODE)
    parser.add_argument("--thresholds", type=int, nargs="+", default=list(DEFAULT_THRESHOLDS_PCT))
    return parser.parse_args()


def load_qw_bm(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_futures_minutes(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"ts", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"minute parquet missing required columns: {sorted(missing)}")
    out = df.copy()
    ts_kst = pd.to_datetime(out["ts"], utc=True).dt.tz_convert("Asia/Seoul")
    if "trade_date_kst" not in out.columns:
        out["trade_date_kst"] = ts_kst.dt.date
    if "hhmm_kst" not in out.columns:
        out["hhmm_kst"] = ts_kst.dt.strftime("%H%M")
    keep_cols = ["ts", "trade_date_kst", "hhmm_kst", "close"]
    if "open" in out.columns:
        keep_cols.insert(3, "open")
    return out[keep_cols].dropna(subset=["trade_date_kst", "hhmm_kst", "close"])


def load_sp500_daily(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"date", "open", "high", "low", "close_pr"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"SP500 parquet missing required columns: {sorted(missing)}")
    out = df[["date", "open", "high", "low", "close_pr"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in ["open", "high", "low", "close_pr"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close_pr"]).sort_values("date")
    out["sp500_ret_cc"] = out["close_pr"].pct_change()
    out["sp500_ret_oc"] = out["close_pr"] / out["open"] - 1.0
    out["sp500_range"] = out["high"] / out["low"] - 1.0
    return out.rename(
        columns={
            "date": "sp500_date",
            "open": "sp500_open",
            "high": "sp500_high",
            "low": "sp500_low",
            "close_pr": "sp500_close",
        }
    )


def write_excel_report(
    path: Path,
    daily: pd.DataFrame,
    events: pd.DataFrame,
    reactions: pd.DataFrame,
    overall: pd.DataFrame,
    yearly: pd.DataFrame,
    intraday_paths: pd.DataFrame,
    intraday_summary: pd.DataFrame,
    sp500_reactions: pd.DataFrame | None = None,
    sp500_summary: pd.DataFrame | None = None,
) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="overall_summary", index=False)
        yearly.to_excel(writer, sheet_name="yearly_summary", index=False)
        reactions.to_excel(writer, sheet_name="event_reactions", index=False)
        events.to_excel(writer, sheet_name="down_day_events", index=False)
        daily.reset_index().to_excel(writer, sheet_name="iks200_daily", index=False)
        build_yearly_threshold_matrix(yearly, "mean_gap_ret").to_excel(writer, sheet_name="matrix_mean_gap")
        build_yearly_threshold_matrix(yearly, "mean_next_close_ret").to_excel(
            writer, sheet_name="matrix_mean_close"
        )
        build_yearly_threshold_matrix(yearly, "gap_up_rate").to_excel(writer, sheet_name="matrix_gap_up_probability")
        intraday_summary.to_excel(writer, sheet_name="intraday_1m_summary", index=False)
        intraday_paths.to_excel(writer, sheet_name="intraday_1m_paths", index=False)
        if sp500_reactions is not None and not sp500_reactions.empty:
            sp500_reactions.to_excel(writer, sheet_name="sp500_event_reactions", index=False)
        if sp500_summary is not None and not sp500_summary.empty:
            sp500_summary.to_excel(writer, sheet_name="sp500_condition_summary", index=False)


def safe_write_excel_report(
    path: Path,
    daily: pd.DataFrame,
    events: pd.DataFrame,
    reactions: pd.DataFrame,
    overall: pd.DataFrame,
    yearly: pd.DataFrame,
    intraday_paths: pd.DataFrame,
    intraday_summary: pd.DataFrame,
    sp500_reactions: pd.DataFrame | None = None,
    sp500_summary: pd.DataFrame | None = None,
) -> Path:
    try:
        write_excel_report(path, daily, events, reactions, overall, yearly, intraday_paths, intraday_summary, sp500_reactions, sp500_summary)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{dt.datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        write_excel_report(fallback, daily, events, reactions, overall, yearly, intraday_paths, intraday_summary, sp500_reactions, sp500_summary)
        return fallback


def write_markdown(
    path: Path,
    *,
    code: str,
    daily: pd.DataFrame,
    events: pd.DataFrame,
    reactions: pd.DataFrame,
    overall: pd.DataFrame,
    yearly: pd.DataFrame,
    intraday_paths: pd.DataFrame,
    excel_filename: str,
    sp500_summary: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# 코스피200 급락일 다음날 반응 이벤트 스터디",
        "",
        "## 설정",
        "",
        f"- 일봉 원천: `parquet/qw_BM.parquet`, 코드 `{code}`.",
        "- 이벤트 수익률: 이벤트일 종가 / 전일 종가 - 1.",
        "- 하락 기준은 상호배타 구간 방식입니다. 예를 들어 -3.4% 하락일은 `-3%~-4% 미만` 표본에만 포함됩니다.",
        "- 매매 관점: 이벤트일 종가 매수 후 다음 거래일 시가 갭, 고가/저가, 종가까지의 흐름을 측정합니다.",
        "- 1분봉 이벤트 경로: `parquet/KOSPI200_1m.parquet`의 선물/지수 1분봉 종가를 사용합니다.",
        "- S&P500 조건부 분석: 한국 이벤트일 `T` 장마감 이후 열리는 미국 정규장 `SP500 date = T` 수익률을 한국 `T+1` 반응에 붙였습니다.",
        "- 다음날 숏 로직은 포함하지 않았습니다.",
        "",
        "## 산출물",
        "",
        f"- `{excel_filename}`: 수치 리포트.",
        "- `01_yearly_subplots_gap_close.png`: year-level subplots by threshold.",
        "- `02_year_threshold_heatmaps.png`: heatmaps for gap-up probability and close return.",
        "- `03_intraday_1m_event_study_subplots.png`: per-year futures/index 1-minute event-study paths by down threshold.",
        "- `04_2026_extreme_intraday_examples.png`: 2026년 -5%~-8% 구간별 실제 예시를 이벤트일 시가부터 T+2 종가까지 표시.",
        "- `05_sp500_evening_condition_heatmaps.png`: 한국 급락일 저녁 S&P500 정규장 수익률별 다음날 반응.",
        "",
        "## 분석 범위",
        "",
        f"- 시작일: `{daily.index.min().date()}`",
        f"- 종료일: `{daily.index.max().date()}`",
        f"- 거래일 수: `{len(daily)}`",
        f"- 하락 기준별 이벤트 행 수: `{len(events)}`",
        f"- 다음 거래일이 존재하는 이벤트 행 수: `{len(reactions)}`",
        f"- 1분봉 이벤트 경로 행 수: `{len(intraday_paths)}`",
        "",
        "## 전체 요약",
        "",
        "| 하락 구간 | 표본 수 | 갭상승 확률 | 평균 갭 | 다음날 평균 고점 | 다음날 평균 저점 | 다음날 평균 종가 | 종가 승률 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"| {_row_bucket_label(row)} | {int(row['n'])} | {pct(row['gap_up_rate'])} | "
            f"{pct(row['mean_gap_ret'])} | {pct(row['mean_next_high_ret'])} | "
            f"{pct(row['mean_next_low_ret'])} | {pct(row['mean_next_close_ret'])} | "
            f"{pct(row['next_close_win_rate'])} |"
        )
    yearly_2026 = yearly[yearly["event_year"].eq(2026)].copy()
    if not yearly_2026.empty:
        lines.extend(
            [
                "",
                "## 2026년 요약",
                "",
                "| 하락 구간 | 표본 수 | 갭상승 확률 | 평균 갭 | 다음날 평균 고점 | 다음날 평균 저점 | 다음날 평균 종가 | 종가 승률 |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in yearly_2026.sort_values("threshold_pct").iterrows():
            lines.append(
                f"| {_row_bucket_label(row)} | {int(row['n'])} | {pct(row['gap_up_rate'])} | "
                f"{pct(row['mean_gap_ret'])} | {pct(row['mean_next_high_ret'])} | "
                f"{pct(row['mean_next_low_ret'])} | {pct(row['mean_next_close_ret'])} | "
                f"{pct(row['next_close_win_rate'])} |"
            )
    if sp500_summary is not None and not sp500_summary.empty:
        lines.extend(
            [
                "",
                "## S&P500 저녁장 조건부 요약",
                "",
                "| 하락 구간 | S&P500 당일 정규장 | 표본 수 | 평균 S&P500 | 갭상승 확률 | 평균 갭 | 다음날 평균 종가 | 종가 승률 |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        focus = sp500_summary[sp500_summary["threshold_pct"].ge(5)].copy()
        if focus.empty:
            focus = sp500_summary.copy()
        for _, row in focus.sort_values(["threshold_pct", "sp500_condition_order"]).iterrows():
            lines.append(
                f"| {_row_bucket_label(row)} | {row['sp500_condition']} | {int(row['n'])} | "
                f"{pct(row['mean_sp500_ret_cc'])} | {pct(row['gap_up_rate'])} | "
                f"{pct(row['mean_gap_ret'])} | {pct(row['mean_next_close_ret'])} | "
                f"{pct(row['next_close_win_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## 해석 메모",
            "",
            "- `평균 갭`은 급락일 종가 매수 후 다음날 시가 부근에서 얼마나 떠서 시작했는지 보는 값입니다.",
            "- `다음날 평균 종가`는 급락일 종가 매수 후 다음날 종가까지 보유했을 때의 평균 수익률입니다.",
            "- Excel의 `mean_open_to_close_ret`은 갭 이후 장중에 추가 상승했는지, 아니면 반납했는지 보는 값입니다.",
            "- 3번 그림의 open 지점은 이벤트일 선물 종가 대비 다음날 첫 1분봉 위치라서 갭상승분을 포함합니다.",
            "- 4번 그림은 이벤트일(T) 시가부터 T+2 종가까지 이어지는 실제 1분봉 경로입니다.",
            "- S&P500 조건부 분석은 미국 휴장일처럼 같은 달력일 S&P500 정규장 데이터가 없는 이벤트를 제외합니다.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_yearly_subplots(out_path: Path, yearly: pd.DataFrame) -> None:
    if yearly.empty:
        return
    years = sorted(int(year) for year in yearly["event_year"].unique())
    ncols = 4
    nrows = int(np.ceil(len(years) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, max(7, nrows * 3.1)), dpi=160, squeeze=False)
    fig.subplots_adjust(hspace=0.46, wspace=0.25, top=0.90)
    fig.suptitle("KOSPI200 Down-Day Close Buy: Next-Day Gap and Close by Year", x=0.02, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.02, 0.935, "Bars show mean return from event-day close. Blue: next open gap. Green: next close.", fontsize=10, color="#4b5563")

    thresholds = sorted(int(t) for t in yearly["threshold_pct"].unique())
    y_min, y_max = _symmetric_limits(yearly[["mean_gap_ret", "mean_next_close_ret"]].to_numpy().ravel() * 100.0)
    for idx, year in enumerate(years):
        ax = axes[idx // ncols][idx % ncols]
        panel = yearly[yearly["event_year"] == year].set_index("threshold_pct").reindex(thresholds)
        x = np.arange(len(thresholds))
        width = 0.34
        ax.bar(x - width / 2, panel["mean_gap_ret"] * 100.0, width=width, color="#3a6ea5", label="gap")
        ax.bar(x + width / 2, panel["mean_next_close_ret"] * 100.0, width=width, color="#4f8f58", label="next close")
        for xpos, n in zip(x, panel["n"].fillna(0).astype(int), strict=True):
            if n:
                ax.text(xpos, y_max * 0.88, str(n), ha="center", va="top", fontsize=7, color="#374151")
        _style_bar_axis(ax, thresholds, y_min, y_max)
        ax.set_title(str(year), loc="left", fontsize=10, fontweight="bold")

    for idx in range(len(years), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncols=2, bbox_to_anchor=(0.985, 0.985))
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_year_threshold_heatmaps(out_path: Path, yearly: pd.DataFrame) -> None:
    if yearly.empty:
        return
    gap_up = build_yearly_threshold_matrix(yearly, "gap_up_rate") * 100.0
    next_close = build_yearly_threshold_matrix(yearly, "mean_next_close_ret") * 100.0
    open_to_close = build_yearly_threshold_matrix(yearly, "mean_open_to_close_ret") * 100.0

    fig, axes = plt.subplots(1, 3, figsize=(19, 7.5), dpi=160)
    fig.suptitle("KOSPI200 Next-Day Reaction Heatmaps", x=0.025, ha="left", fontsize=17, fontweight="bold")
    _heatmap(axes[0], gap_up, "Gap-up probability (%)", cmap="BuGn", center_zero=False, vmin=0.0, vmax=100.0)
    _heatmap(axes[1], next_close, "Mean close-to-next-close (%)", cmap="BrBG", center_zero=True)
    _heatmap(axes[2], open_to_close, "Mean next open-to-close (%)", cmap="BrBG", center_zero=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_yearly_intraday_event_study_subplots(out_path: Path, intraday_summary: pd.DataFrame) -> None:
    if intraday_summary.empty:
        return
    years = sorted(int(year) for year in intraday_summary["event_year"].unique())
    thresholds = sorted(int(t) for t in intraday_summary["threshold_pct"].unique())
    ncols = 4
    nrows = int(np.ceil(len(years) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, max(7, nrows * 3.1)), dpi=160, squeeze=False)
    fig.subplots_adjust(hspace=0.44, wspace=0.24, top=0.90)
    fig.suptitle(
        "Futures 1-Minute Next-Day Event Study by Year and Down Threshold",
        x=0.02,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.935,
        "Events are classified from qw_BM IKS200 daily returns. Lines use KOSPI200 1-minute close, measured from the event-day futures close. The point at open includes the overnight gap.",
        fontsize=10,
        color="#4b5563",
    )
    y_min, y_max = _intraday_path_limits(intraday_summary)
    colors = plt.get_cmap("tab10").colors
    for idx, year in enumerate(years):
        ax = axes[idx // ncols][idx % ncols]
        year_data = intraday_summary[intraday_summary["event_year"] == year]
        for color_idx, threshold in enumerate(thresholds):
            group = year_data[year_data["threshold_pct"] == threshold].sort_values("minute_from_open")
            if group.empty:
                continue
            ax.plot(
                group["minute_from_open"],
                group["mean_ret_from_futures_event_close"] * 100.0,
                linewidth=1.45 if threshold <= 4 else 1.0,
                alpha=0.90 if threshold <= 4 else 0.62,
                color=colors[(threshold - 1) % len(colors)],
                label=f"{_bucket_plot_label(threshold, thresholds)} n={int(group['n'].max())}",
            )
        _style_intraday_path_axis(ax, y_min, y_max)
        ax.set_title(str(year), loc="left", fontsize=10, fontweight="bold")

    for idx in range(len(years), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, ncols=4, bbox_to_anchor=(0.985, 0.985), fontsize=8)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_2026_extreme_intraday_examples(out_path: Path, futures_minutes: pd.DataFrame, reactions: pd.DataFrame) -> None:
    subset = reactions[(reactions["event_year"] == 2026) & (reactions["threshold_pct"].between(5, 8))].copy()
    if subset.empty:
        return
    minute_data = _normalize_plot_minutes(futures_minutes)
    by_day = {
        trade_date: frame.reset_index(drop=True)
        for trade_date, frame in minute_data.groupby("trade_date_kst", sort=True)
    }
    trade_dates = sorted(by_day)
    trade_date_pos = {trade_date: idx for idx, trade_date in enumerate(trade_dates)}
    thresholds = [5, 6, 7, 8]
    panels: dict[int, list[dict[str, object]]] = {}
    y_values: list[float] = []
    for threshold in thresholds:
        panel: list[dict[str, object]] = []
        for _, event in subset[subset["threshold_pct"] == threshold].sort_values("event_date").iterrows():
            event_date = pd.Timestamp(event["event_date"]).date()
            next_date = pd.Timestamp(event["next_date"]).date()
            event_day = by_day.get(event_date)
            next_day = by_day.get(next_date)
            next_pos = trade_date_pos.get(next_date)
            t_plus_two = by_day.get(trade_dates[next_pos + 1]) if next_pos is not None and next_pos + 1 < len(trade_dates) else None
            if event_day is None or event_day.empty or next_day is None or next_day.empty or t_plus_two is None or t_plus_two.empty:
                continue
            path = _three_day_example_path(event_day, next_day, t_plus_two)
            if path.empty:
                continue
            label = f"{pd.Timestamp(event['event_date']).strftime('%m-%d')} ({float(event['event_ret_cc']) * 100:.1f}%)"
            y_values.extend(path["ret_pct"].tolist())
            panel.append({"path": path, "label": label})
        panels[threshold] = panel

    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5), dpi=160, squeeze=False)
    fig.subplots_adjust(hspace=0.34, wspace=0.22, top=0.88)
    fig.suptitle("2026 Extreme Down-Day Examples: Event-Day Open to T+2 Close", x=0.02, ha="left", fontsize=17, fontweight="bold")
    fig.text(
        0.02,
        0.925,
        "Each line is an actual 2026 event. Buckets are non-overlapping. Paths run from event-day open through T+2 close.",
        fontsize=10,
        color="#4b5563",
    )
    y_min, y_max = _example_limits(np.asarray(y_values, dtype=float))
    colors = plt.get_cmap("tab20").colors
    for idx, threshold in enumerate(thresholds):
        ax = axes[idx // 2][idx % 2]
        panel = panels.get(threshold, [])
        for line_idx, item in enumerate(panel):
            path = item["path"]
            ax.plot(path["x"], path["ret_pct"], linewidth=1.35, color=colors[line_idx % len(colors)], label=str(item["label"]))
        _style_two_day_example_axis(ax, y_min, y_max)
        ax.set_title(f"{_bucket_plot_label(threshold, thresholds)} | n={len(panel)}", loc="left", fontsize=11, fontweight="bold")
        if panel:
            ax.legend(frameon=False, fontsize=6.5, ncols=2, loc="best")
        else:
            ax.text(0.5, 0.5, "no matched intraday events", transform=ax.transAxes, ha="center", va="center", color="#6b7280")
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def attach_sp500_evening_reactions(reactions: pd.DataFrame, sp500_daily: pd.DataFrame) -> pd.DataFrame:
    if reactions.empty or sp500_daily.empty:
        return pd.DataFrame()
    sp = sp500_daily.copy()
    sp["event_date"] = pd.to_datetime(sp["sp500_date"])
    merged = reactions.merge(sp, on="event_date", how="left")
    merged["sp500_condition"] = merged["sp500_ret_cc"].map(_sp500_condition)
    merged["sp500_condition_order"] = merged["sp500_condition"].map(_sp500_condition_order)
    return merged


def summarize_sp500_conditions(sp500_reactions: pd.DataFrame) -> pd.DataFrame:
    matched = sp500_reactions.dropna(subset=["sp500_ret_cc", "sp500_condition"]).copy()
    if matched.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["threshold_pct", "sp500_condition"]
    for (threshold_pct, condition), group in matched.groupby(group_cols, sort=True):
        close = group["next_close_ret"].astype(float)
        row = {
            "threshold_pct": int(threshold_pct),
            "bucket_label": str(group["bucket_label"].iloc[0]),
            "bucket_floor_pct": int(group["bucket_floor_pct"].iloc[0]),
            "bucket_ceiling_pct": group["bucket_ceiling_pct"].iloc[0],
            "sp500_condition": str(condition),
            "sp500_condition_order": int(group["sp500_condition_order"].iloc[0]),
            "n": int(len(group)),
            "mean_sp500_ret_cc": float(group["sp500_ret_cc"].mean()),
            "median_sp500_ret_cc": float(group["sp500_ret_cc"].median()),
            "mean_sp500_ret_oc": float(group["sp500_ret_oc"].mean()),
            "gap_up_rate": float(group["gap_up"].mean()),
            "mean_gap_ret": float(group["gap_ret"].mean()),
            "mean_next_close_ret": float(close.mean()),
            "next_close_win_rate": float((close > 0.0).mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["threshold_pct", "sp500_condition_order"]).reset_index(drop=True)


def plot_sp500_condition_heatmaps(out_path: Path, sp500_summary: pd.DataFrame) -> None:
    if sp500_summary.empty:
        return
    data = sp500_summary.copy()
    labels = data.drop_duplicates("threshold_pct").sort_values("threshold_pct")
    threshold_labels = [int(value) for value in labels["threshold_pct"].tolist()]
    y_labels = [_bucket_plot_label(threshold, threshold_labels) for threshold in threshold_labels]
    conditions = ["<=-2%", "-2%~-1%", "-1%~0%", "0%~1%", ">=1%"]

    def matrix(value_col: str) -> pd.DataFrame:
        pivot = data.pivot(index="threshold_pct", columns="sp500_condition", values=value_col)
        return pivot.reindex(index=labels["threshold_pct"].tolist(), columns=conditions)

    close = matrix("mean_next_close_ret") * 100.0
    gap = matrix("mean_gap_ret") * 100.0
    count = matrix("n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.3), dpi=160)
    fig.suptitle("KOSPI200 Next-Day Reaction by Same-Evening S&P500 Return", x=0.025, ha="left", fontsize=16, fontweight="bold")
    fig.text(
        0.025,
        0.925,
        "Korea event date T is matched to S&P500 regular-session date T, which closes before Korea T+1 opens.",
        fontsize=9.5,
        color="#4b5563",
    )
    _condition_heatmap(axes[0], close, y_labels, "Mean KOSPI200 T+1 close (%)", cmap="BrBG", center_zero=True)
    _condition_heatmap(axes[1], gap, y_labels, "Mean KOSPI200 T+1 gap (%)", cmap="BrBG", center_zero=True)
    _condition_heatmap(axes[2], count, y_labels, "Matched event count", cmap="YlGnBu", center_zero=False)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.3f}%"


def _sp500_condition(value: float | int | None) -> str | float:
    if value is None or pd.isna(value):
        return np.nan
    ret = float(value)
    if ret <= -0.02:
        return "<=-2%"
    if ret <= -0.01:
        return "-2%~-1%"
    if ret < 0.0:
        return "-1%~0%"
    if ret < 0.01:
        return "0%~1%"
    return ">=1%"


def _sp500_condition_order(value: object) -> int | float:
    order = {"<=-2%": 0, "-2%~-1%": 1, "-1%~0%": 2, "0%~1%": 3, ">=1%": 4}
    return order.get(value, np.nan)


def _row_bucket_label(row: pd.Series) -> str:
    label = row.get("bucket_label")
    if pd.notna(label):
        return str(label)
    ceiling = row.get("bucket_ceiling_pct")
    return down_bucket_label(int(row["threshold_pct"]), int(ceiling) if pd.notna(ceiling) else None)


def _bucket_plot_label(threshold: int, thresholds: list[int]) -> str:
    ordered = sorted(thresholds)
    idx = ordered.index(int(threshold))
    if idx + 1 >= len(ordered):
        return f"<=-{threshold}%"
    return f"-{threshold}~-{ordered[idx + 1]}%"


def _bucket_axis_label(threshold: int, thresholds: list[int]) -> str:
    ordered = sorted(thresholds)
    idx = ordered.index(int(threshold))
    if idx + 1 >= len(ordered):
        return f"<=\n-{threshold}%"
    return f"-{threshold}\n~-{ordered[idx + 1]}%"


def _style_bar_axis(ax: plt.Axes, thresholds: list[int], y_min: float, y_max: float) -> None:
    ax.axhline(0, color="#1f2933", linewidth=0.8)
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels([_bucket_axis_label(threshold, thresholds) for threshold in thresholds], fontsize=7)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color="#d8dee4", alpha=0.8)
    ax.set_facecolor("#f7f8fa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    *,
    cmap: str,
    center_zero: bool,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    values = data.to_numpy(dtype=float)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#ffffff")
    if center_zero:
        finite = values[np.isfinite(values)]
        limit = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite)))) if len(finite) else 1.0
        image = ax.imshow(values, aspect="auto", cmap=cmap_obj, vmin=-limit, vmax=limit)
    else:
        image = ax.imshow(values, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels([str(int(year)) for year in data.index], fontsize=8)
    ax.set_xticks(np.arange(len(data.columns)))
    thresholds = [int(threshold) for threshold in data.columns]
    ax.set_xticklabels([_bucket_plot_label(threshold, thresholds) for threshold in thresholds], fontsize=8)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                text_color = _heatmap_text_color(value, image.norm)
                ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", fontsize=6.5, color=text_color)
    plt.colorbar(image, ax=ax, fraction=0.030, pad=0.02)


def _heatmap_text_color(value: float, norm) -> str:
    normalized = norm(value)
    if np.ma.is_masked(normalized):
        return "#111827"
    return "#ffffff" if float(normalized) > 0.78 else "#111827"


def _condition_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    y_labels: list[str],
    title: str,
    *,
    cmap: str,
    center_zero: bool,
) -> None:
    values = data.to_numpy(dtype=float)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#ffffff")
    if center_zero:
        finite = values[np.isfinite(values)]
        limit = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite)))) if len(finite) else 1.0
        image = ax.imshow(values, aspect="auto", cmap=cmap_obj, vmin=-limit, vmax=limit)
    else:
        image = ax.imshow(values, aspect="auto", cmap=cmap_obj)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(list(data.columns), fontsize=8, rotation=20, ha="right")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                text_color = _heatmap_text_color(value, image.norm)
                fmt = "{:.0f}" if "count" in title.lower() else "{:.2f}"
                ax.text(col_idx, row_idx, fmt.format(value), ha="center", va="center", fontsize=7, color=text_color)
    plt.colorbar(image, ax=ax, fraction=0.035, pad=0.02)


def _style_intraday_path_axis(ax: plt.Axes, y_min: float, y_max: float) -> None:
    ax.axhline(0, color="#1f2933", linewidth=0.8)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, 390)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax.set_xticklabels(["open", "+60", "+120", "+180", "+240", "+300", "+360"], fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(color="#d8dee4", alpha=0.8)
    ax.set_facecolor("#f7f8fa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _normalize_plot_minutes(minutes: pd.DataFrame) -> pd.DataFrame:
    data = minutes.copy()
    data["trade_date_kst"] = pd.to_datetime(data["trade_date_kst"]).dt.date
    data["hhmm_kst"] = data["hhmm_kst"].astype(str).str.zfill(4)
    if "open" not in data.columns:
        data["open"] = data["close"]
    sort_cols = ["trade_date_kst", "hhmm_kst"]
    if "ts" in data.columns:
        data["ts"] = pd.to_datetime(data["ts"], utc=True)
        sort_cols.append("ts")
    return data.sort_values(sort_cols).reset_index(drop=True)


def _three_day_example_path(event_day: pd.DataFrame, next_day: pd.DataFrame, t_plus_two: pd.DataFrame) -> pd.DataFrame:
    if event_day.empty or next_day.empty or t_plus_two.empty:
        return pd.DataFrame()
    base = float(event_day.iloc[0]["open"])
    if not np.isfinite(base) or np.isclose(base, 0.0):
        return pd.DataFrame()
    event = event_day.copy()
    event["x"] = np.arange(len(event))
    event["ret_pct"] = event["close"].astype(float) / base * 100.0 - 100.0
    event.loc[event.index[0], "ret_pct"] = float(event.iloc[0]["open"]) / base * 100.0 - 100.0
    next_ = next_day.copy()
    gap = 12
    next_["x"] = np.arange(len(next_)) + int(event["x"].max()) + gap
    next_["ret_pct"] = next_["close"].astype(float) / base * 100.0 - 100.0
    next_.loc[next_.index[0], "ret_pct"] = float(next_.iloc[0]["open"]) / base * 100.0 - 100.0
    third = t_plus_two.copy()
    third["x"] = np.arange(len(third)) + int(next_["x"].max()) + gap
    third["ret_pct"] = third["close"].astype(float) / base * 100.0 - 100.0
    third.loc[third.index[0], "ret_pct"] = float(third.iloc[0]["open"]) / base * 100.0 - 100.0
    return pd.concat([event[["x", "ret_pct"]], next_[["x", "ret_pct"]], third[["x", "ret_pct"]]], ignore_index=True)


def _style_two_day_example_axis(ax: plt.Axes, y_min: float, y_max: float) -> None:
    ax.axhline(0, color="#1f2933", linewidth=0.8)
    ax.axvline(390, color="#6b7280", linewidth=0.8, linestyle="--")
    ax.axvline(792, color="#6b7280", linewidth=0.8, linestyle="--")
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([0, 195, 390, 597, 792, 999, 1194])
    ax.set_xticklabels(["T open", "T mid", "T close\nT+1 open", "T+1 mid", "T+1 close\nT+2 open", "T+2 mid", "T+2 close"], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(color="#d8dee4", alpha=0.8)
    ax.set_facecolor("#f7f8fa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _example_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return -10.0, 10.0
    low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    pad = max((high - low) * 0.12, 1.0)
    return low - pad, high + pad


def _intraday_path_limits(intraday_summary: pd.DataFrame) -> tuple[float, float]:
    if intraday_summary.empty:
        return -5.0, 5.0
    arr = intraday_summary["mean_ret_from_futures_event_close"].to_numpy(dtype=float) * 100.0
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return -5.0, 5.0
    low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    pad = max((high - low) * 0.15, 1.0)
    return float(low - pad), float(high + pad)


def _symmetric_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return -1.0, 1.0
    limit = float(np.nanpercentile(np.abs(finite), 98))
    limit = max(limit * 1.25, 0.5)
    return -limit, limit


def _clear_output_dir(path: Path) -> None:
    for child in path.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except PermissionError:
            continue


def main() -> None:
    args = parse_args()
    if args.out_dir.resolve().is_relative_to(Path.cwd().resolve()) and args.out_dir.exists():
        _clear_output_dir(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    qw_bm = load_qw_bm(args.input)
    futures_minutes = load_futures_minutes(args.minute_input)
    sp500_daily = load_sp500_daily(args.sp500_input)
    daily = build_iks200_daily_frame(qw_bm, code=args.code)
    events = mark_down_day_events(daily, thresholds_pct=args.thresholds)
    reactions = build_next_day_reactions(daily, events)
    overall = summarize_overall(reactions)
    yearly = summarize_yearly(reactions)
    intraday_paths = build_intraday_next_day_paths(futures_minutes, reactions)
    intraday_summary = summarize_intraday_paths(intraday_paths)
    sp500_reactions = attach_sp500_evening_reactions(reactions, sp500_daily)
    sp500_summary = summarize_sp500_conditions(sp500_reactions)

    excel_path = safe_write_excel_report(
        args.out_dir / "next_day_down_event_study.xlsx",
        daily,
        events,
        reactions,
        overall,
        yearly,
        intraday_paths,
        intraday_summary,
        sp500_reactions,
        sp500_summary,
    )
    write_markdown(
        args.out_dir / "next_day_down_event_study.md",
        code=args.code,
        daily=daily,
        events=events,
        reactions=reactions,
        overall=overall,
        yearly=yearly,
        intraday_paths=intraday_paths,
        excel_filename=excel_path.name,
        sp500_summary=sp500_summary,
    )
    plot_yearly_subplots(args.out_dir / "01_yearly_subplots_gap_close.png", yearly)
    plot_year_threshold_heatmaps(args.out_dir / "02_year_threshold_heatmaps.png", yearly)
    plot_yearly_intraday_event_study_subplots(args.out_dir / "03_intraday_1m_event_study_subplots.png", intraday_summary)
    plot_2026_extreme_intraday_examples(args.out_dir / "04_2026_extreme_intraday_examples.png", futures_minutes, reactions)
    plot_sp500_condition_heatmaps(args.out_dir / "05_sp500_evening_condition_heatmaps.png", sp500_summary)

    print(overall.to_string(index=False))
    print(f"out={args.out_dir}")


if __name__ == "__main__":
    main()
