"""Event study for forced-liquidation-like intraday pressure after large selloffs.

The input data is KOSPI200 futures/index 1-minute parquet with KST trade-date and
HHMM columns. The script does not observe forced-liquidation orders directly.
Instead it tests whether T+1/T+2 after large down days show repeatable pressure
and rebound around the time windows commonly associated with liquidation flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("parquet/KOSPI200_1m.parquet")
OUT_DIR = Path("results/forced_liquidation_event_test")


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start: str
    end: str
    exit_hhmm: str
    note: str


WINDOWS = [
    WindowSpec("open_1st_0900", "0900", "0915", "1000", "1차: 미수·신용·담보대출 시초가 충격"),
    WindowSpec("cfd_1000", "1000", "1030", "1100", "2차: CFD 반대매매 가능 구간"),
    WindowSpec("cfd_follow_1030", "1030", "1100", "1400", "10시 CFD 후행/아시아장 영향 가능 구간"),
    WindowSpec("stockloan_1400", "1400", "1430", "1500", "3차: 스탁론·연계신용 가능 구간"),
    WindowSpec("avoidance_1500", "1500", "1530", "1545", "익일 반대매매 회피성 매도 가능 구간"),
]


def _first_at_or_after(day: pd.DataFrame, hhmm: str) -> pd.Series | None:
    subset = day[day["hhmm_kst"] >= hhmm]
    if subset.empty:
        return None
    return subset.iloc[0]


def _last_at_or_before(day: pd.DataFrame, hhmm: str) -> pd.Series | None:
    subset = day[day["hhmm_kst"] <= hhmm]
    if subset.empty:
        return None
    return subset.iloc[-1]


def load_data(path: Path) -> pd.DataFrame:
    cols = [
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "buy_vol",
        "sell_vol",
        "net_buy_vol",
        "ofi",
        "trade_date_kst",
        "hhmm_kst",
    ]
    df = pd.read_parquet(path, columns=cols)
    df = df.sort_values(["trade_date_kst", "hhmm_kst", "ts"]).reset_index(drop=True)
    df["trade_date_kst"] = pd.to_datetime(df["trade_date_kst"]).dt.date
    return df


def daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("trade_date_kst", sort=True)
    first = grouped.head(1).set_index("trade_date_kst")
    last = grouped.tail(1).set_index("trade_date_kst")
    daily = pd.DataFrame(
        {
            "open": first["open"],
            "close": last["close"],
            "first_hhmm": first["hhmm_kst"],
            "last_hhmm": last["hhmm_kst"],
            "bar_count": grouped.size(),
        }
    )
    daily["ret_cc"] = daily["close"].pct_change()
    daily["ret_oc"] = daily["close"] / daily["open"] - 1.0
    daily["date_index"] = np.arange(len(daily))
    return daily


def mark_events(daily: pd.DataFrame, cc_threshold: float = -0.025, oc_threshold: float = -0.020) -> pd.DataFrame:
    events = daily[(daily["ret_cc"] <= cc_threshold) | (daily["ret_oc"] <= oc_threshold)].copy()
    events["event_date"] = events.index
    events["event_idx"] = events["date_index"].astype(int)
    events["event_reason"] = np.where(
        (events["ret_cc"] <= cc_threshold) & (events["ret_oc"] <= oc_threshold),
        "cc_and_oc",
        np.where(events["ret_cc"] <= cc_threshold, "close_to_close", "open_to_close"),
    )
    return events.reset_index(drop=True)


def measure_window(day: pd.DataFrame, spec: WindowSpec) -> dict[str, object] | None:
    w = day[(day["hhmm_kst"] >= spec.start) & (day["hhmm_kst"] <= spec.end)]
    if w.empty:
        return None
    start_row = _first_at_or_after(day, spec.start)
    end_row = _last_at_or_before(day, spec.end)
    exit_row = _last_at_or_before(day, spec.exit_hhmm)
    close_row = _last_at_or_before(day, "1545")
    if start_row is None or end_row is None or exit_row is None or close_row is None:
        return None

    low_idx = w["low"].idxmin()
    low_row = day.loc[low_idx]
    high_idx = w["high"].idxmax()
    high_row = day.loc[high_idx]
    start_px = float(start_row["close"])
    end_px = float(end_row["close"])
    low_px = float(low_row["low"])
    high_px = float(high_row["high"])
    exit_px = float(exit_row["close"])
    close_px = float(close_row["close"])

    buy_vol = float(w["buy_vol"].sum(skipna=True))
    sell_vol = float(w["sell_vol"].sum(skipna=True))
    total_aggr = buy_vol + sell_vol

    return {
        "window": spec.name,
        "window_note": spec.note,
        "window_start": spec.start,
        "window_end": spec.end,
        "exit_hhmm": spec.exit_hhmm,
        "start_px": start_px,
        "end_px": end_px,
        "low_px": low_px,
        "low_hhmm": low_row["hhmm_kst"],
        "high_px": high_px,
        "high_hhmm": high_row["hhmm_kst"],
        "exit_px": exit_px,
        "close_px": close_px,
        "window_ret_start_to_end": end_px / start_px - 1.0,
        "pressure_start_to_low": low_px / start_px - 1.0,
        "ideal_low_to_exit": exit_px / low_px - 1.0,
        "ideal_low_to_close": close_px / low_px - 1.0,
        "end_to_exit": exit_px / end_px - 1.0,
        "end_to_close": close_px / end_px - 1.0,
        "aggressive_sell_share": sell_vol / total_aggr if total_aggr else np.nan,
        "volume": float(w["volume"].sum()),
        "bar_count": int(len(w)),
    }


def build_samples(df: pd.DataFrame, daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    by_day = {date: frame for date, frame in df.groupby("trade_date_kst", sort=True)}
    dates = list(daily.index)
    rows: list[dict[str, object]] = []

    event_targets: dict[tuple[object, int], list[dict[str, object]]] = {}
    for _, event in events.iterrows():
        event_idx = int(event["event_idx"])
        for lag in (1, 2):
            target_idx = event_idx + lag
            if target_idx >= len(dates):
                continue
            target_date = dates[target_idx]
            event_targets.setdefault((target_date, lag), []).append(event.to_dict())

    for date in dates:
        day = by_day.get(date)
        if day is None:
            continue
        matched_lags = [lag for lag in (1, 2) if (date, lag) in event_targets]
        sample_lags = matched_lags if matched_lags else [0]
        for lag in sample_lags:
            event_info = event_targets[(date, lag)][-1] if lag else {}
            for spec in WINDOWS:
                measured = measure_window(day, spec)
                if measured is None:
                    continue
                measured.update(
                    {
                        "trade_date": date,
                        "lag": lag,
                        "sample": f"T+{lag}" if lag else "baseline",
                        "event_date": event_info.get("event_date"),
                        "event_ret_cc": event_info.get("ret_cc"),
                        "event_ret_oc": event_info.get("ret_oc"),
                        "event_reason": event_info.get("event_reason"),
                    }
                )
                rows.append(measured)

    return pd.DataFrame(rows)


def summarize(samples: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "window_ret_start_to_end",
        "pressure_start_to_low",
        "ideal_low_to_exit",
        "ideal_low_to_close",
        "end_to_exit",
        "end_to_close",
        "aggressive_sell_share",
    ]
    grouped = samples.groupby(["window", "sample"], sort=False)
    summary = grouped[metrics].agg(["count", "mean", "median", "std"])
    summary.columns = ["__".join(col) for col in summary.columns]
    summary = summary.reset_index()

    win_rates = grouped[["ideal_low_to_exit", "end_to_exit", "end_to_close"]].agg(lambda x: float((x > 0).mean()))
    win_rates = win_rates.rename(columns={c: f"{c}__win_rate" for c in win_rates.columns}).reset_index()
    summary = summary.merge(win_rates, on=["window", "sample"], how="left")
    return summary


def compare_to_baseline(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ["pressure_start_to_low", "ideal_low_to_exit", "end_to_exit", "end_to_close", "aggressive_sell_share"]
    rng = np.random.default_rng(20260625)
    for window in samples["window"].unique():
        base = samples[(samples["window"] == window) & (samples["sample"] == "baseline")]
        for sample in ("T+1", "T+2"):
            target = samples[(samples["window"] == window) & (samples["sample"] == sample)]
            if target.empty or base.empty:
                continue
            for metric in metrics:
                x = target[metric].dropna().to_numpy()
                y = base[metric].dropna().to_numpy()
                if len(x) < 3 or len(y) < 10:
                    continue
                observed = float(np.mean(x) - np.mean(y))
                # Two-sided permutation test over means.
                pooled = np.concatenate([x, y])
                n = len(x)
                sims = []
                for _ in range(3000):
                    perm = rng.permutation(pooled)
                    sims.append(float(np.mean(perm[:n]) - np.mean(perm[n:])))
                sims_arr = np.asarray(sims)
                p_value = float((np.abs(sims_arr) >= abs(observed)).mean())
                rows.append(
                    {
                        "window": window,
                        "sample": sample,
                        "metric": metric,
                        "n_event": len(x),
                        "n_baseline": len(y),
                        "event_mean": float(np.mean(x)),
                        "baseline_mean": float(np.mean(y)),
                        "diff_event_minus_baseline": observed,
                        "permutation_p_value": p_value,
                    }
                )
    return pd.DataFrame(rows)


def fmt_pct(x: float | int | None) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x) * 100:.3f}%"


def write_markdown(
    path: Path,
    daily: pd.DataFrame,
    events: pd.DataFrame,
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Forced-Liquidation Intraday Event Test")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Input: `{INPUT.as_posix()}`")
    lines.append(f"- Date range: `{daily.index.min()}` to `{daily.index.max()}`")
    lines.append(f"- Trading days: `{len(daily):,}`")
    lines.append("- Event definition: `close-to-close <= -2.5%` OR `open-to-close <= -2.0%`")
    lines.append(f"- Event days found: `{len(events):,}`")
    lines.append("- Test samples: event day `T` then next trading days `T+1`, `T+2`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This is not a direct forced-liquidation order identifier. It tests whether the commonly cited "
        "liquidation windows show stronger sell pressure and better post-dip rebound after large selloff days."
    )
    lines.append("")
    lines.append("Key metrics:")
    lines.append("- `pressure_start_to_low`: window start to window low. More negative means stronger intraday pressure.")
    lines.append("- `ideal_low_to_exit`: buying the window low and exiting at the designated later time. This is an upper bound.")
    lines.append("- `end_to_exit`: buying at the end of the pressure window and exiting later. This is more tradable.")
    lines.append("- `end_to_close`: buying at the end of the pressure window and exiting same-day close/auction.")
    lines.append("")

    best = comparison[comparison["metric"].isin(["end_to_exit", "end_to_close"])].copy()
    if not best.empty:
        best = best.sort_values("diff_event_minus_baseline", ascending=False).head(12)
        lines.append("## Best Event-vs-Baseline Improvements")
        lines.append("")
        lines.append("| window | sample | metric | event mean | baseline mean | diff | p-value |")
        lines.append("|---|---:|---|---:|---:|---:|---:|")
        for _, row in best.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["window"]),
                        str(row["sample"]),
                        str(row["metric"]),
                        fmt_pct(row["event_mean"]),
                        fmt_pct(row["baseline_mean"]),
                        fmt_pct(row["diff_event_minus_baseline"]),
                        f"{row['permutation_p_value']:.3f}",
                    ]
                )
                + " |"
            )
        lines.append("")

    pressure = comparison[comparison["metric"] == "pressure_start_to_low"].copy()
    if not pressure.empty:
        pressure = pressure.sort_values("diff_event_minus_baseline").head(10)
        lines.append("## Strongest Extra Pressure After Selloff")
        lines.append("")
        lines.append("| window | sample | event pressure | baseline pressure | extra pressure | p-value |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in pressure.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["window"]),
                        str(row["sample"]),
                        fmt_pct(row["event_mean"]),
                        fmt_pct(row["baseline_mean"]),
                        fmt_pct(row["diff_event_minus_baseline"]),
                        f"{row['permutation_p_value']:.3f}",
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.append("## Event Days")
    lines.append("")
    lines.append("| date | ret_cc | ret_oc | reason |")
    lines.append("|---|---:|---:|---|")
    for _, row in events.sort_values("event_date").iterrows():
        lines.append(
            f"| {row['event_date']} | {fmt_pct(row['ret_cc'])} | {fmt_pct(row['ret_oc'])} | {row['event_reason']} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_ko(
    path: Path,
    daily: pd.DataFrame,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    window_desc = {
        "open_1st_0900": "09:00 1차: 09:00~09:15",
        "cfd_1000": "10:00 CFD: 10:00~10:30",
        "cfd_follow_1030": "10:30~11:00 후행/아시아",
        "stockloan_1400": "14:00 스탁론: 14:00~14:30",
        "avoidance_1500": "15:00 회피매도: 15:00~15:30",
    }
    candidates = [
        ("open_1st_0900", "T+1", "end_to_exit", "09:15", "10:00", "가장 강함"),
        ("open_1st_0900", "T+1", "end_to_close", "09:15", "종가", "강함"),
        ("cfd_1000", "T+2", "end_to_exit", "10:30", "11:00", "작지만 유의"),
        ("cfd_1000", "T+1", "end_to_exit", "10:30", "11:00", "약함"),
        ("stockloan_1400", "T+1", "end_to_exit", "14:30", "15:00", "약함"),
    ]

    lines: list[str] = []
    lines.append("# 반대매매 가설: 지수선물 1분봉 초기 테스트 요약")
    lines.append("")
    lines.append("## 테스트 정의")
    lines.append("")
    lines.append(f"- 데이터: `{INPUT.as_posix()}`")
    lines.append(f"- 기간: {daily.index.min()} ~ {daily.index.max()}, {len(daily):,} 거래일")
    lines.append("- 급락일 T: 전일 종가 대비 -2.5% 이하 또는 당일 시가 대비 종가 -2.0% 이하")
    lines.append(f"- 급락 이벤트 수: {len(events)}개")
    lines.append("- 검정 대상: 급락일 이후 `T+1`, `T+2`의 09시, 10시, 10:30, 14시, 15시 구간")
    lines.append("")
    lines.append("## 결론")
    lines.append("")
    lines.append(
        "선물 1분봉으로 이 가설은 테스트 가능하다. 초기 결과는 **T+1 09시 1차 눌림 후 매수**가 "
        "가장 뚜렷하다. 10시 CFD 구간은 눌림 압력은 강하지만 매수 후 수익성은 약하고, "
        "14시 스탁론 구간은 T+1에서만 약한 개선이 있다. T+2 14시와 15시 이후 매수는 현재 "
        "기준으로는 불리하게 나왔다."
    )
    lines.append("")
    lines.append("## 매매 가능한 후보")
    lines.append("")
    lines.append("| 후보 | 표본 | 진입 가정 | 청산 | 평균수익 | baseline | 차이 | 승률 | p-value | 판단 |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---|")
    for window, sample, metric, entry, exit_label, judgement in candidates:
        row_match = comparison[
            (comparison["window"] == window)
            & (comparison["sample"] == sample)
            & (comparison["metric"] == metric)
        ]
        if row_match.empty:
            continue
        row = row_match.iloc[0]
        srow = summary[(summary["window"] == window) & (summary["sample"] == sample)].iloc[0]
        win_rate = srow.get(f"{metric}__win_rate", np.nan)
        lines.append(
            f"| {window_desc[window]} 눌림 후 | {sample} | {entry} | {exit_label} | "
            f"{fmt_pct(row['event_mean'])} | {fmt_pct(row['baseline_mean'])} | "
            f"{fmt_pct(row['diff_event_minus_baseline'])} | {win_rate * 100:.1f}% | "
            f"{row['permutation_p_value']:.3f} | {judgement} |"
        )
    lines.append("")
    lines.append("## 눌림 압력 자체는 어디서 강했나")
    lines.append("")
    lines.append("| 구간 | 표본 | 이벤트 이후 평균 눌림 | 평상시 평균 눌림 | 추가 눌림 | p-value |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    pressure = comparison[comparison["metric"] == "pressure_start_to_low"].sort_values(
        "diff_event_minus_baseline"
    ).head(10)
    for _, row in pressure.iterrows():
        lines.append(
            f"| {window_desc[row['window']]} | {row['sample']} | {fmt_pct(row['event_mean'])} | "
            f"{fmt_pct(row['baseline_mean'])} | {fmt_pct(row['diff_event_minus_baseline'])} | "
            f"{row['permutation_p_value']:.3f} |"
        )
    lines.append("")
    lines.append("## 읽는 법")
    lines.append("")
    lines.append(
        "- `눌림 압력`: 해당 시간대 시작가에서 시간대 저가까지의 하락률이다. "
        "더 음수일수록 급락 이후 해당 시간대에 추가 하방 압력이 강했다는 뜻이다."
    )
    lines.append(
        "- `진입 가정`: 시간대 저점 매수는 비현실적이므로, 위 후보 표는 더 보수적으로 "
        "“압력 구간 종료 시점 매수”를 매매 가능한 후보로 봤다."
    )
    lines.append(
        "- 선물 1분봉은 실제 반대매매 주문 라벨이 없으므로, 결과는 반대매매 직접 검출이 아니라 "
        "시간대별 반복 패턴 검정이다."
    )
    lines.append(
        "- 거래비용, 슬리피지, 증거금, 체결 가능성은 아직 반영하지 않았다. 특히 0.05~0.10% "
        "수준의 edge는 비용 반영 후 사라질 수 있다."
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(INPUT)
    daily = daily_frame(df)
    events = mark_events(daily)
    samples = build_samples(df, daily, events)
    summary = summarize(samples)
    comparison = compare_to_baseline(samples)

    daily.to_csv(OUT_DIR / "daily_returns.csv", encoding="utf-8-sig")
    events.to_csv(OUT_DIR / "event_days.csv", index=False, encoding="utf-8-sig")
    samples.to_csv(OUT_DIR / "event_window_samples.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "window_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT_DIR / "event_vs_baseline.csv", index=False, encoding="utf-8-sig")
    write_markdown(OUT_DIR / "summary.md", daily, events, samples, summary, comparison)
    write_markdown_ko(OUT_DIR / "summary_ko.md", daily, events, summary, comparison)

    print(f"events={len(events)}")
    print(f"samples={len(samples)}")
    print(f"out={OUT_DIR}")


if __name__ == "__main__":
    main()
