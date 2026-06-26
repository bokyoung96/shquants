from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etc.next_day_down_event_study import (
    DEFAULT_THRESHOLDS_PCT,
    build_next_day_reactions,
    build_yearly_threshold_matrix,
    mark_down_day_events,
    summarize_overall,
    summarize_yearly,
)


INPUT = Path("parquet/KOSPI200_1m.parquet")
INDEX_REPORT = Path("results/next_day_down_event_study/next_day_down_event_study.xlsx")
OUT_DIR = Path("results/next_day_down_event_study_futures_1m_daily_2025_2026")
START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2026-12-31")


def load_futures_minutes(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path)
    required = {"ts", "open", "high", "low", "close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"minute parquet missing required columns: {sorted(missing)}")
    out = data[list(required) + (["volume"] if "volume" in data.columns else [])].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out["date"] = out["ts"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None).dt.normalize()
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["date", "open", "high", "low", "close"]).sort_values(["date", "ts"])


def build_daily_from_minutes(minutes: pd.DataFrame) -> pd.DataFrame:
    grouped = minutes.groupby("date", sort=True)
    daily = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "minute_count": grouped["close"].size(),
        }
    )
    if "volume" in minutes.columns:
        daily["volume"] = grouped["volume"].sum()
    daily.index.name = "date"
    daily["ret_cc"] = daily["close"].pct_change()
    daily["ret_oc"] = daily["close"] / daily["open"] - 1.0
    return daily


def restrict_event_window(daily: pd.DataFrame) -> pd.DataFrame:
    window = daily[(daily.index >= START) & (daily.index <= END)].copy()
    window["date_index"] = np.arange(len(window), dtype=int)
    return window


def build_index_vs_futures_comparison(index_report: Path, futures_overall: pd.DataFrame) -> pd.DataFrame:
    if not index_report.exists() or futures_overall.empty:
        return pd.DataFrame()
    index_reactions = pd.read_excel(index_report, sheet_name="event_reactions")
    event_dates = pd.to_datetime(index_reactions["event_date"])
    index_reactions = index_reactions[event_dates.between(START, END)].copy()
    rows: list[dict[str, object]] = []
    for (threshold, label), group in index_reactions.groupby(["threshold_pct", "bucket_label"], sort=True):
        close = group["next_close_ret"].astype(float)
        rows.append(
            {
                "threshold_pct": int(threshold),
                "bucket_label": label,
                "source": "IKS200 daily",
                "n": int(len(group)),
                "gap_up_rate": float(group["gap_up"].mean()),
                "mean_gap_ret": float(group["gap_ret"].mean()),
                "mean_next_close_ret": float(close.mean()),
                "next_close_win_rate": float((close > 0.0).mean()),
            }
        )
    futures = futures_overall[
        ["threshold_pct", "bucket_label", "n", "gap_up_rate", "mean_gap_ret", "mean_next_close_ret", "next_close_win_rate"]
    ].copy()
    futures["source"] = "Futures 1m daily"
    return pd.concat([pd.DataFrame(rows), futures], ignore_index=True).sort_values(["threshold_pct", "source"])


def write_excel(
    path: Path,
    daily: pd.DataFrame,
    events: pd.DataFrame,
    reactions: pd.DataFrame,
    overall: pd.DataFrame,
    yearly: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="overall_summary", index=False)
        yearly.to_excel(writer, sheet_name="yearly_summary", index=False)
        reactions.to_excel(writer, sheet_name="event_reactions", index=False)
        events.to_excel(writer, sheet_name="down_day_events", index=False)
        daily.reset_index().to_excel(writer, sheet_name="futures_daily_from_1m", index=False)
        if not comparison.empty:
            comparison.to_excel(writer, sheet_name="index_vs_futures", index=False)


def write_markdown(
    path: Path,
    daily: pd.DataFrame,
    events: pd.DataFrame,
    reactions: pd.DataFrame,
    overall: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    lines = [
        "# 선물 1분봉 집계 일봉 기준 급락일 다음날 반응",
        "",
        "## 설정",
        "",
        "- 원천: `parquet/KOSPI200_1m.parquet`.",
        "- 1분봉 `ts`를 KST로 변환한 뒤, 날짜별 첫 open / max high / min low / 마지막 close로 일봉을 만들었습니다.",
        "- 이벤트 기간: 2025-01-01부터 2026-12-31까지입니다. 현재 파일은 2026-06-25까지 있습니다.",
        "- 하락 구간은 상호배타 방식입니다. 예: `-5%~-6% 미만`, `-8% 이상 하락`.",
        "- 수익률 기준은 선물 집계 일봉의 `T 종가 / T-1 종가 - 1`입니다.",
        "",
        "## 산출물",
        "",
        "- `futures_1m_daily_2025_2026.xlsx`: 전체 수치.",
        "- `01_futures_daily_gap_close_by_year.png`: 연도별 다음날 갭/종가 반응.",
        "- `02_futures_daily_year_threshold_heatmaps.png`: 연도별/하락구간별 히트맵.",
        "- Excel의 `index_vs_futures` 시트: 기존 IKS200 일봉 결과와 선물 1분봉 집계 일봉 결과 비교.",
        "",
        "## 범위",
        "",
        f"- 집계 일봉 시작일: `{daily.index.min().date()}`",
        f"- 집계 일봉 종료일: `{daily.index.max().date()}`",
        f"- 집계 일봉 수: `{len(daily)}`",
        f"- 이벤트 수: `{len(events)}`",
        f"- 다음 거래일 반응 수: `{len(reactions)}`",
        "",
        "## 전체 요약",
        "",
        "| 하락 구간 | 표본 수 | 갭상승 확률 | 평균 갭 | 다음날 평균 고점 | 다음날 평균 저점 | 다음날 평균 종가 | 종가 승률 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"| {row['bucket_label']} | {int(row['n'])} | {_pct(row['gap_up_rate'])} | "
            f"{_pct(row['mean_gap_ret'])} | {_pct(row['mean_next_high_ret'])} | "
            f"{_pct(row['mean_next_low_ret'])} | {_pct(row['mean_next_close_ret'])} | "
            f"{_pct(row['next_close_win_rate'])} |"
        )
    if not comparison.empty:
        lines.extend(
            [
                "",
                "## 기존 IKS200 일봉 대비 선물 1분봉 집계 일봉 비교",
                "",
                "- 같은 2025~2026 이벤트 기간으로 맞춰 비교했습니다.",
                "- 전체 방향은 큰 급락 후 `T 종가 -> T+1 종가` 반등이 나타나는 구간이 많다는 점에서 유사합니다.",
                "- 다만 선물 1분봉 집계 기준은 종가 산정, 연결 방식, 장마감/마감 단일가성 데이터 차이 때문에 버킷 경계가 바뀌는 날이 있습니다.",
                "- 특히 `-8% 이상 하락`은 IKS200 일봉 평균 `+7.49%`보다 선물 1분봉 집계 평균 `+2.17%`로 더 보수적입니다.",
                "- 선물 기준에서는 2026-03-03이 `-8% 이상`으로 들어가며 다음날 추가 급락이 포함되어 극단 구간 평균을 낮춥니다.",
                "",
                "| 하락 구간 | 기준 | 표본 수 | 갭상승 확률 | 평균 갭 | 다음날 평균 종가 | 종가 승률 |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in comparison.iterrows():
            lines.append(
                f"| {row['bucket_label']} | {row['source']} | {int(row['n'])} | "
                f"{_pct(row['gap_up_rate'])} | {_pct(row['mean_gap_ret'])} | "
                f"{_pct(row['mean_next_close_ret'])} | {_pct(row['next_close_win_rate'])} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_yearly_gap_close(path: Path, yearly: pd.DataFrame) -> None:
    if yearly.empty:
        return
    years = sorted(yearly["event_year"].astype(int).unique())
    thresholds = list(DEFAULT_THRESHOLDS_PCT)
    fig, axes = plt.subplots(1, len(years), figsize=(7.2 * len(years), 4.8), dpi=160, squeeze=False)
    fig.suptitle("Futures 1m-Derived Daily Down Events: Next-Day Gap and Close", x=0.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.91, "Event dates: 2025-2026. Bars are returns from event-day futures daily close.", fontsize=9.5, color="#4b5563")
    values = yearly[["mean_gap_ret", "mean_next_close_ret"]].to_numpy(dtype=float) * 100.0
    limit = max(float(np.nanpercentile(np.abs(values[np.isfinite(values)]), 98)) * 1.25, 1.0) if np.isfinite(values).any() else 1.0
    for idx, year in enumerate(years):
        ax = axes[0][idx]
        panel = yearly[yearly["event_year"].eq(year)].set_index("threshold_pct").reindex(thresholds)
        x = np.arange(len(thresholds))
        width = 0.34
        ax.bar(x - width / 2, panel["mean_gap_ret"] * 100.0, width=width, color="#3a6ea5", label="gap")
        ax.bar(x + width / 2, panel["mean_next_close_ret"] * 100.0, width=width, color="#4f8f58", label="next close")
        for xpos, n in zip(x, panel["n"].fillna(0).astype(int), strict=True):
            if n:
                ax.text(xpos, limit * 0.88, str(n), ha="center", va="top", fontsize=8, color="#374151")
        ax.axhline(0, color="#1f2933", linewidth=0.8)
        ax.set_ylim(-limit, limit)
        ax.set_xticks(x)
        ax.set_xticklabels([_bucket_axis_label(t, thresholds) for t in thresholds], fontsize=8)
        ax.grid(axis="y", color="#d8dee4", alpha=0.8)
        ax.set_facecolor("#f7f8fa")
        ax.set_title(str(year), loc="left", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncols=2)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_year_threshold_heatmaps(path: Path, yearly: pd.DataFrame) -> None:
    if yearly.empty:
        return
    thresholds = list(DEFAULT_THRESHOLDS_PCT)
    gap_up = build_yearly_threshold_matrix(yearly, "gap_up_rate").reindex(columns=thresholds) * 100.0
    next_close = build_yearly_threshold_matrix(yearly, "mean_next_close_ret").reindex(columns=thresholds) * 100.0
    open_to_close = build_yearly_threshold_matrix(yearly, "mean_open_to_close_ret").reindex(columns=thresholds) * 100.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), dpi=160)
    fig.suptitle("Futures 1m-Derived Daily Event Heatmaps", x=0.025, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.025, 0.91, "Rows are event years. Columns are non-overlapping down buckets.", fontsize=9.5, color="#4b5563")
    _heatmap(axes[0], gap_up, "Gap-up probability (%)", cmap="BuGn", center_zero=False, vmin=0.0, vmax=100.0)
    _heatmap(axes[1], next_close, "Mean close-to-next-close (%)", cmap="BrBG", center_zero=True)
    _heatmap(axes[2], open_to_close, "Mean next open-to-close (%)", cmap="BrBG", center_zero=True)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


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
    thresholds = [int(threshold) for threshold in data.columns]
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels([str(int(year)) for year in data.index], fontsize=8)
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels([_bucket_plot_label(threshold, thresholds) for threshold in thresholds], fontsize=8, rotation=20, ha="right")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                color = _heatmap_text_color(value, image.norm)
                ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(image, ax=ax, fraction=0.045, pad=0.02)


def _heatmap_text_color(value: float, norm) -> str:
    normalized = norm(value)
    if np.ma.is_masked(normalized):
        return "#111827"
    return "#ffffff" if float(normalized) > 0.78 else "#111827"


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


def _pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.3f}%"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    minutes = load_futures_minutes(INPUT)
    daily_all = build_daily_from_minutes(minutes)
    daily = restrict_event_window(daily_all)
    events = mark_down_day_events(daily, thresholds_pct=DEFAULT_THRESHOLDS_PCT)
    reactions = build_next_day_reactions(daily, events)
    overall = summarize_overall(reactions)
    yearly = summarize_yearly(reactions)
    comparison = build_index_vs_futures_comparison(INDEX_REPORT, overall)

    write_excel(OUT_DIR / "futures_1m_daily_2025_2026.xlsx", daily, events, reactions, overall, yearly, comparison)
    write_markdown(OUT_DIR / "futures_1m_daily_2025_2026.md", daily, events, reactions, overall, comparison)
    plot_yearly_gap_close(OUT_DIR / "01_futures_daily_gap_close_by_year.png", yearly)
    plot_year_threshold_heatmaps(OUT_DIR / "02_futures_daily_year_threshold_heatmaps.png", yearly)
    print(overall.to_string(index=False))
    print(f"out={OUT_DIR}")


if __name__ == "__main__":
    main()
