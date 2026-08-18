"""Robust timing study for KOSPI buy-sidecar events.

The current sidecar pipeline labels one-minute bars by their ending minute. This
study keeps the legacy floor-close result for comparison, but bases its main
conclusion on the first minute boundary strictly after each requested delay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from etc.sell_sidecar_economics import (
    StrategyRule,
    build_rule_trades,
    pair_sidecar_events,
)


DEFAULT_PARQUET_DIR = Path("etc/data/sidecar/parquet")
DEFAULT_OUT_DIR = Path("results/buy_sidecar_entry_study")
DEFAULT_ENTRY_DELAYS = tuple(range(0, 6))
DEFAULT_RELEASE_DELAYS = tuple(range(0, 11))
DEFAULT_COST_BPS = (5, 10, 20)
BASELINE_ENTRY_DELAY = 3
BASELINE_EXIT_DELAY = 3
BOOTSTRAP_SEED = 20260710


def build_candidate_rules(
    *,
    entry_delays: Sequence[int] = DEFAULT_ENTRY_DELAYS,
    release_delays: Sequence[int] = DEFAULT_RELEASE_DELAYS,
) -> list[StrategyRule]:
    rules: list[StrategyRule] = []
    for entry_delay in entry_delays:
        for release_delay in release_delays:
            rules.append(
                StrategyRule(
                    name=f"a{entry_delay}_r{release_delay}",
                    economic_role="release_continuation",
                    entry_anchor="activation",
                    entry_delay_minutes=entry_delay,
                    exit_anchor="release",
                    exit_delay_minutes=release_delay,
                    thesis="Enter during the buy-sidecar halt and exit after program bids resume.",
                )
            )
        rules.append(
            StrategyRule(
                name=f"a{entry_delay}_close",
                economic_role="day_continuation",
                entry_anchor="activation",
                entry_delay_minutes=entry_delay,
                exit_anchor="close",
                exit_delay_minutes=0,
                thesis="Treat the buy-sidecar as an intraday upside regime through the close.",
            )
        )
    return rules


def build_execution_sensitivity(
    pairs: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    entry_delays: Sequence[int] = DEFAULT_ENTRY_DELAYS,
    release_delays: Sequence[int] = DEFAULT_RELEASE_DELAYS,
) -> pd.DataFrame:
    """Bracket executable returns for end-labeled one-minute OHLC bars."""
    if pairs.empty or prices.empty:
        return pd.DataFrame()
    data = prices.copy()
    data["dt"] = pd.to_datetime(data["dt"])
    lookup = data.drop_duplicates("dt", keep="last").set_index("dt").sort_index()

    rows: list[dict[str, object]] = []
    for _, pair in pairs.sort_values("activation_dt").iterrows():
        activation = pd.Timestamp(pair["activation_dt"])
        release = pd.Timestamp(pair["release_dt"])
        for entry_delay in entry_delays:
            entry_target = activation + pd.Timedelta(minutes=entry_delay)
            entry_close_bar = _next_minute_boundary(entry_target)
            entry_open_bar = entry_close_bar + pd.Timedelta(minutes=1)
            for release_delay in release_delays:
                exit_target = release + pd.Timedelta(minutes=release_delay)
                exit_close_bar = _next_minute_boundary(exit_target)
                exit_open_bar = exit_close_bar + pd.Timedelta(minutes=1)
                rule = f"a{entry_delay}_r{release_delay}"
                candidates = (
                    (
                        "first_complete_close",
                        entry_close_bar,
                        exit_close_bar,
                        "close",
                        "close",
                    ),
                    (
                        "first_boundary_open",
                        entry_open_bar,
                        exit_open_bar,
                        "open",
                        "open",
                    ),
                    (
                        "containing_bar_worst",
                        entry_close_bar,
                        exit_close_bar,
                        "high",
                        "low",
                    ),
                )
                for fill_method, entry_bar, exit_bar, entry_col, exit_col in candidates:
                    entry_price = _value_at(lookup, entry_bar, entry_col)
                    exit_price = _value_at(lookup, exit_bar, exit_col)
                    if not np.isfinite(entry_price) or not np.isfinite(exit_price):
                        continue
                    rows.append(
                        {
                            "trade_date": pair["trade_date"],
                            "rule": rule,
                            "entry_delay_m": entry_delay,
                            "exit_kind": "release",
                            "exit_delay_m": release_delay,
                            "fill_method": fill_method,
                            "activation_dt": activation,
                            "release_dt": release,
                            "entry_target_dt": entry_target,
                            "exit_target_dt": exit_target,
                            "entry_bar_dt": entry_bar,
                            "exit_bar_dt": exit_bar,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "ret": exit_price / entry_price - 1.0,
                        }
                    )
    return pd.DataFrame(rows)


def build_boundary_open_close_trades(
    pairs: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    entry_delays: Sequence[int] = DEFAULT_ENTRY_DELAYS,
) -> pd.DataFrame:
    if pairs.empty or prices.empty:
        return pd.DataFrame()
    data = prices.copy()
    data["dt"] = pd.to_datetime(data["dt"])
    data["trade_date"] = data["dt"].dt.date
    lookup = data.drop_duplicates("dt", keep="last").set_index("dt").sort_index()
    close_by_date = {
        date: pd.Timestamp(day.loc[day["dt"].le(pd.Timestamp(f"{date} 15:30:00")), "dt"].max())
        for date, day in data.groupby("trade_date", sort=False)
    }

    rows: list[dict[str, object]] = []
    for _, pair in pairs.sort_values("activation_dt").iterrows():
        activation = pd.Timestamp(pair["activation_dt"])
        trade_date = pair["trade_date"]
        exit_bar = close_by_date.get(trade_date)
        if exit_bar is None or pd.isna(exit_bar):
            continue
        for entry_delay in entry_delays:
            entry_target = activation + pd.Timedelta(minutes=entry_delay)
            entry_bar = _next_minute_boundary(entry_target) + pd.Timedelta(minutes=1)
            entry_price = _value_at(lookup, entry_bar, "open")
            exit_price = _value_at(lookup, exit_bar, "close")
            if not np.isfinite(entry_price) or not np.isfinite(exit_price):
                continue
            rows.append(
                {
                    "trade_date": trade_date,
                    "rule": f"a{entry_delay}_close",
                    "entry_delay_m": entry_delay,
                    "exit_kind": "close",
                    "exit_delay_m": pd.NA,
                    "fill_method": "first_boundary_open",
                    "activation_dt": activation,
                    "release_dt": pd.Timestamp(pair["release_dt"]),
                    "entry_target_dt": entry_target,
                    "entry_bar_dt": entry_bar,
                    "exit_bar_dt": exit_bar,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "ret": exit_price / entry_price - 1.0,
                }
            )
    return pd.DataFrame(rows)


def summarize_candidates(
    trades: pd.DataFrame,
    *,
    cost_bps: Sequence[int] = DEFAULT_COST_BPS,
    bootstrap_samples: int = 20_000,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for rule in sorted(trades["rule"].astype(str).unique()):
        group = trades[trades["rule"].astype(str).eq(rule)].copy()
        group["trade_date"] = pd.to_datetime(group["trade_date"])
        group = group.sort_values("trade_date")
        returns = group["ret"].astype(float).dropna().to_numpy()
        if not len(returns):
            continue
        bootstrap = returns[
            rng.integers(0, len(returns), size=(max(1, bootstrap_samples), len(returns)))
        ].mean(axis=1)
        loo = (
            np.array([np.delete(returns, index).mean() for index in range(len(returns))])
            if len(returns) > 1
            else np.array([returns.mean()])
        )
        split = max(1, len(returns) // 2)
        second_half = returns[split:] if split < len(returns) else returns
        equity = np.cumprod(1.0 + returns)
        drawdown = equity / np.maximum.accumulate(equity) - 1.0
        months = group.loc[group["ret"].notna(), "trade_date"].dt.to_period("M")
        lomo = [returns[months.ne(month).to_numpy()].mean() for month in months.unique() if months.ne(month).any()]
        positive = returns[returns > 0.0]
        row: dict[str, object] = {
            "rule": rule,
            "n": len(returns),
            "wins": int((returns > 0.0).sum()),
            "win_rate": float((returns > 0.0).mean()),
            "mean_ret": float(returns.mean()),
            "median_ret": float(np.median(returns)),
            "p10_ret": float(np.quantile(returns, 0.10)),
            "min_ret": float(returns.min()),
            "max_ret": float(returns.max()),
            "compound_ret": float(equity[-1] - 1.0),
            "max_drawdown": float(drawdown.min()),
            "bootstrap_mean_ci_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_mean_ci_high": float(np.quantile(bootstrap, 0.975)),
            "loo_mean_min": float(loo.min()),
            "loo_mean_max": float(loo.max()),
            "first_half_mean": float(returns[:split].mean()),
            "second_half_mean": float(second_half.mean()),
            "lomo_mean_min": float(min(lomo)) if lomo else np.nan,
            "lomo_mean_max": float(max(lomo)) if lomo else np.nan,
            "positive_pnl_concentration": (
                float(positive.max() / positive.sum()) if positive.size and positive.sum() else np.nan
            ),
        }
        for column in ("entry_delay_m", "exit_kind", "exit_delay_m", "fill_method"):
            if column in group.columns:
                row[column] = group.iloc[0][column]
        for bps in cost_bps:
            net = returns - bps / 10_000.0
            row[f"net_{bps}bps_mean_ret"] = float(net.mean())
            row[f"net_{bps}bps_win_rate"] = float((net > 0.0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_execution_sensitivity(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    keys = ["entry_delay_m", "exit_delay_m", "fill_method"]
    for values, group in trades.groupby(keys, sort=True):
        entry_delay, exit_delay, fill_method = values
        returns = group["ret"].astype(float).dropna()
        rows.append(
            {
                "entry_delay_m": int(entry_delay),
                "exit_delay_m": int(exit_delay),
                "fill_method": fill_method,
                "n": int(len(returns)),
                "mean_ret": float(returns.mean()),
                "median_ret": float(returns.median()),
                "win_rate": float((returns > 0.0).mean()),
                "min_ret": float(returns.min()),
                "max_ret": float(returns.max()),
            }
        )
    return pd.DataFrame(rows)


def select_robust_entry_window(
    candidate_summary: pd.DataFrame,
    execution_summary: pd.DataFrame,
    *,
    exit_delay_minutes: int = BASELINE_EXIT_DELAY,
) -> pd.DataFrame:
    candidates = candidate_summary[
        candidate_summary["exit_kind"].eq("release")
        & pd.to_numeric(candidate_summary["exit_delay_m"], errors="coerce").eq(exit_delay_minutes)
    ].copy()
    boundary = execution_summary[
        execution_summary["exit_delay_m"].eq(exit_delay_minutes)
        & execution_summary["fill_method"].eq("first_boundary_open")
    ][["entry_delay_m", "mean_ret"]].rename(columns={"mean_ret": "boundary_open_mean_ret"})
    worst = execution_summary[
        execution_summary["exit_delay_m"].eq(exit_delay_minutes)
        & execution_summary["fill_method"].eq("containing_bar_worst")
    ][["entry_delay_m", "mean_ret"]].rename(columns={"mean_ret": "bar_worst_mean_ret"})
    candidates = candidates.merge(boundary, on="entry_delay_m", how="left").merge(
        worst, on="entry_delay_m", how="left"
    )
    keep = (
        candidates["win_rate"].ge(0.90)
        & candidates["bootstrap_mean_ci_low"].gt(0.0)
        & candidates["net_10bps_mean_ret"].gt(0.0)
        & candidates["boundary_open_mean_ret"].gt(0.0)
        & candidates["bar_worst_mean_ret"].gt(0.0)
    )
    return candidates.loc[keep].sort_values("entry_delay_m").reset_index(drop=True)


def paired_rule_comparison(
    trades: pd.DataFrame,
    *,
    baseline_rule: str,
    candidate_rules: Iterable[str],
    bootstrap_samples: int = 20_000,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    pivot = trades.pivot_table(index="trade_date", columns="rule", values="ret", aggfunc="last")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for candidate in candidate_rules:
        if candidate == baseline_rule or candidate not in pivot or baseline_rule not in pivot:
            continue
        paired = pivot[[candidate, baseline_rule]].dropna()
        diff = (paired[candidate] - paired[baseline_rule]).to_numpy(float)
        if not len(diff):
            continue
        bootstrap = diff[
            rng.integers(0, len(diff), size=(max(1, bootstrap_samples), len(diff)))
        ].mean(axis=1)
        rows.append(
            {
                "candidate_rule": candidate,
                "baseline_rule": baseline_rule,
                "n": len(diff),
                "mean_diff": float(diff.mean()),
                "bootstrap_diff_ci_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_diff_ci_high": float(np.quantile(bootstrap, 0.975)),
                "candidate_better_events": int((diff > 0.0).sum()),
                "candidate_worse_events": int((diff < 0.0).sum()),
                "exact_sign_flip_pvalue": _exact_sign_flip_pvalue(diff),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["holm_pvalue"] = _holm_adjust(out["exact_sign_flip_pvalue"].to_numpy(float))
    return out


def build_holding_regime_summary(
    pairs: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    entry_delay_minutes: int = BASELINE_ENTRY_DELAY,
) -> pd.DataFrame:
    data = prices.copy()
    data["dt"] = pd.to_datetime(data["dt"])
    data["trade_date"] = data["dt"].dt.date
    dates = sorted(data["trade_date"].unique())
    by_date = {date: day.sort_values("dt") for date, day in data.groupby("trade_date", sort=False)}
    rows: list[dict[str, object]] = []
    for _, pair in pairs.sort_values("trade_date").iterrows():
        trade_date = pair["trade_date"]
        day = by_date.get(trade_date)
        if day is None:
            continue
        entry_bar = _next_minute_boundary(
            pd.Timestamp(pair["activation_dt"]) + pd.Timedelta(minutes=entry_delay_minutes)
        ) + pd.Timedelta(minutes=1)
        entry = day.loc[day["dt"].eq(entry_bar), "open"]
        if entry.empty:
            continue
        entry_price = float(entry.iloc[-1])
        close_price = float(day.loc[day["dt"].le(pd.Timestamp(f"{trade_date} 15:30:00")), "close"].iloc[-1])
        rows.append({"trade_date": trade_date, "rule": "entry_to_close", "ret": close_price / entry_price - 1.0})
        date_index = dates.index(trade_date)
        if date_index + 1 >= len(dates):
            continue
        next_date = dates[date_index + 1]
        next_day = by_date[next_date]
        active = next_day[next_day.get("volume", pd.Series(index=next_day.index, dtype=float)).fillna(0).gt(0)]
        first = active.iloc[0] if not active.empty else next_day.iloc[0]
        next_open = float(first["open"])
        next_close = float(next_day.loc[next_day["dt"].le(pd.Timestamp(f"{next_date} 15:30:00")), "close"].iloc[-1])
        rows.extend(
            [
                {"trade_date": trade_date, "rule": "close_to_next_open", "ret": next_open / close_price - 1.0},
                {"trade_date": trade_date, "rule": "close_to_next_close", "ret": next_close / close_price - 1.0},
                {"trade_date": trade_date, "rule": "entry_to_next_close", "ret": next_close / entry_price - 1.0},
            ]
        )
    return summarize_candidates(pd.DataFrame(rows), cost_bps=(), bootstrap_samples=10_000)


def write_report(
    output_dir: Path,
    *,
    pairs: pd.DataFrame,
    canonical_summary: pd.DataFrame,
    legacy_summary: pd.DataFrame,
    execution_summary: pd.DataFrame,
    robust_window: pd.DataFrame,
    entry_comparison: pd.DataFrame,
    exit_comparison: pd.DataFrame,
    holding_summary: pd.DataFrame,
    data_as_of: pd.Timestamp,
) -> Path:
    baseline = canonical_summary[canonical_summary["rule"].eq("a3_r3")].iloc[0]
    legacy = legacy_summary[legacy_summary["rule"].eq("a3_r3")].iloc[0]
    entry_delays = robust_window["entry_delay_m"].astype(int).tolist()
    entry_text = f"+{min(entry_delays)}~+{max(entry_delays)}분" if entry_delays else "판정 불가"
    exit_window = _conservative_exit_window(canonical_summary)
    exit_text = f"해제 +{min(exit_window)}~+{max(exit_window)}분" if exit_window else "판정 불가"

    entry_rows = canonical_summary[
        canonical_summary["exit_kind"].eq("release")
        & pd.to_numeric(canonical_summary["exit_delay_m"], errors="coerce").eq(BASELINE_EXIT_DELAY)
    ].sort_values("entry_delay_m")
    exit_rows = canonical_summary[
        canonical_summary["exit_kind"].eq("release")
        & canonical_summary["entry_delay_m"].eq(BASELINE_ENTRY_DELAY)
    ].sort_values("exit_delay_m")

    lines = [
        "# 매수 사이드카 진입 시점 연구",
        "",
        f"- 데이터 기준일: `{data_as_of:%Y-%m-%d}`",
        f"- 공식 매수 사이드카 표본: `{len(pairs)}`건",
        f"- 표본 기간: `{pairs['activation_dt'].min()}` ~ `{pairs['activation_dt'].max()}`",
        "- 거래 수단: `KODEX 레버리지` 1분 OHLC",
        "- CB 이벤트 제외, 수익률은 왕복 비용 차감 전 gross return",
        "",
        "## 결론",
        "",
        f"- 잠정 강건 진입 구간: **발동 후 {entry_text}**",
        "- 운영 중심점: **발동 +3분 목표 후 첫 분 경계 시가**",
        f"- 보수적 청산 구간: **{exit_text} 목표 후 첫 분 경계 시가**",
        f"- 중심 규칙 `A+3 / R+3`: 평균 `{_pct(baseline['mean_ret'])}`, 중앙값 `{_pct(baseline['median_ret'])}`, 승률 `{_pct(baseline['win_rate'])}`",
        f"- 10bp 비용 스트레스 후 평균 `{_pct(baseline['net_10bps_mean_ret'])}`, 승률 `{_pct(baseline['net_10bps_win_rate'])}`",
        "- 정확한 한 분짜리 최적점이 아니라 이웃 분에서도 유지되는 구간으로 해석해야 한다.",
        "",
        "## 시계 보정",
        "",
        "가격 파일은 거래일마다 09:01~15:30의 390개 봉을 가져 종료시각 표기 1분봉으로 판단했다.",
        "기존 코드는 `floor(event + delay)` 봉 종가를 사용해 선언한 지연보다 최대 59초 이른 가격을 잡는다.",
        "본 연구의 기본 체결은 목표시각 이후 첫 분 경계에서 시작하는 다음 봉 시가이며, 첫 완성봉 종가와 해당 봉 OHLC 최악 조합도 함께 확인했다.",
        f"기존 legacy `floor-close A+3/R+3` 평균은 `{_pct(legacy['mean_ret'])}`였고, 시계 보정 후 중심 규칙 평균은 `{_pct(baseline['mean_ret'])}`로 현상 자체는 유지됐다.",
        "",
        "## 진입 민감도",
        "",
        "청산을 R+3으로 고정했다.",
        "",
        "| 발동 후 | 표본 | 평균 | 중앙값 | 승률 | 최악 | 95% bootstrap 평균 CI | 10bp 후 평균 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in entry_rows.iterrows():
        lines.append(
            f"| +{int(row['entry_delay_m'])}분 | {int(row['n'])} | {_pct(row['mean_ret'])} | "
            f"{_pct(row['median_ret'])} | {_pct(row['win_rate'])} | {_pct(row['min_ret'])} | "
            f"{_pct(row['bootstrap_mean_ci_low'])} ~ {_pct(row['bootstrap_mean_ci_high'])} | "
            f"{_pct(row['net_10bps_mean_ret'])} |"
        )
    lines.extend(
        [
            "",
            "A+0은 체결 가정과 보수적 OHLC 스트레스에서 약했고, A+5는 이미 해제 직전 또는 직후가 되어 기대수익이 급감했다.",
            "A+1~A+4는 bootstrap 하단, 10bp 비용, 첫 경계 시가, OHLC 최악 평균이 모두 양수였다. A+3은 해제 전 실행 여유를 남기는 중심점이다.",
            "",
            "## 청산 민감도",
            "",
            "진입을 A+3으로 고정했다.",
            "",
            "| 해제 후 | 표본 | 평균 | 중앙값 | 승률 | 최악 | 10bp 후 평균 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in exit_rows.iterrows():
        lines.append(
            f"| +{int(row['exit_delay_m'])}분 | {int(row['n'])} | {_pct(row['mean_ret'])} | "
            f"{_pct(row['median_ret'])} | {_pct(row['win_rate'])} | {_pct(row['min_ret'])} | "
            f"{_pct(row['net_10bps_mean_ret'])} |"
        )
    lines.extend(
        [
            "",
            "R+2~R+3이 해제 직후 첫 상승 재개를 포착하는 가장 이른 안정 구간이었다.",
            "R+8 이상의 표본 내 평균은 더 높지만 손실 꼬리와 승률이 악화되어 최적화 후보로 채택하지 않았다.",
            "",
            "## 실행 및 안정성",
            "",
            f"- leave-one-event-out 평균 범위: `{_pct(baseline['loo_mean_min'])}` ~ `{_pct(baseline['loo_mean_max'])}`",
            f"- leave-one-month-out 평균 범위: `{_pct(baseline['lomo_mean_min'])}` ~ `{_pct(baseline['lomo_mean_max'])}`",
            f"- 전반부/후반부 평균: `{_pct(baseline['first_half_mean'])}` / `{_pct(baseline['second_half_mean'])}`",
            f"- 단일 양수 이벤트 최대 기여도: `{_pct(baseline['positive_pnl_concentration'])}`",
            "- exact sign-flip과 Holm 보정은 paired_comparisons.csv에 기록했다. 이미 같은 표본을 본 뒤의 비교이므로 확증 p-value로 해석하지 않는다.",
            "",
            "## 보유기간 분리",
            "",
            "| 구간 | 표본 | 평균 | 중앙값 | 승률 | 최악 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in holding_summary.iterrows():
        lines.append(
            f"| {row['rule']} | {int(row['n'])} | {_pct(row['mean_ret'])} | {_pct(row['median_ret'])} | "
            f"{_pct(row['win_rate'])} | {_pct(row['min_ret'])} |"
        )
    lines.extend(
        [
            "",
            "당일 종가 보유는 평균수익은 크지만 손실 꼬리가 약 5%로 확대된다. 다음 날 종가까지의 연장은 중앙값과 승률이 악화되어 해제 직후 전략과 별도 가설로 취급한다.",
            "",
            "## 한계와 동결 규칙",
            "",
            "- 15건 중 14건이 2026년이고 13건이 09:30 이전이라 독립 표본 수는 더 작다.",
            "- 현재 결과는 진정한 OOS가 아니다. 조건부 필터나 ML을 추가하지 않는다.",
            "- bid/ask와 체결 틱이 없어 1분 OHLC로만 체결을 근사했다.",
            "- matched non-event control을 만들지 않았으므로 사이드카의 인과 효과가 아니라 이벤트 조건부 실행 시점 연구다.",
            "- 데이터는 2026-06-15까지만 포함한다. 이후 이벤트를 반영하려면 이벤트와 ETF 원천을 함께 갱신해야 한다.",
            "- 잠정 운영 후보는 `A+3 / R+3`, 다음 10개 매수 사이드카는 규칙을 바꾸지 않는 prospective 확인 표본으로 남긴다.",
            "- 약 30개 이상의 독립 이벤트와 여러 시장 국면이 쌓이기 전에는 '최적점'으로 확정하지 않는다.",
            "",
            "## 제도적 해석",
            "",
            "KRX 규정상 선물 상승 시 프로그램 매수호가 효력이 5분간 정지되고 이후 접수 순서대로 효력이 재개된다. 관측된 A+1~A+4 진입, R+2~R+3 청산 구간은 이 일시적 매수 압력 제거와 해제 후 재유입 가설에 부합한다.",
            "",
            "- KRX: https://global.krx.co.kr/contents/GLB/06/0602/0602010204/GLB0602010204T4.jsp",
            "- KRX trading guide: https://global.krx.co.kr/contents/GLB/01/0109/0109000000/guide_to_trading_in_the_korean_stock_market.pdf",
        ]
    )
    output = output_dir / "buy_sidecar_entry_study_ko.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def plot_study(
    output_dir: Path,
    canonical_summary: pd.DataFrame,
    execution_summary: pd.DataFrame,
    holding_summary: pd.DataFrame,
) -> Path:
    del execution_summary
    entry = canonical_summary[
        canonical_summary["exit_kind"].eq("release")
        & pd.to_numeric(canonical_summary["exit_delay_m"], errors="coerce").eq(BASELINE_EXIT_DELAY)
    ].sort_values("entry_delay_m")
    baseline = canonical_summary[canonical_summary["rule"].eq("a3_r3")].iloc[0]
    sample_count = int(baseline.get("n", 15))
    gross_wins = int(baseline.get("wins", round(float(baseline["win_rate"]) * sample_count)))

    font_names = {font.name for font in font_manager.fontManager.ttflist}
    font_family = "Noto Sans KR" if "Noto Sans KR" in font_names else "Malgun Gothic"
    with plt.rc_context({"font.family": font_family, "axes.unicode_minus": False}):
        bg = "#F7F9FC"
        ink = "#17212B"
        muted = "#52606B"
        line = "#C9D3DC"
        coral = "#EE7258"
        coral_soft = "#FBE4DE"
        teal = "#0B8F87"
        teal_soft = "#DDF3EE"
        green = "#239A69"
        green_soft = "#DFF3E9"
        red = "#D65757"

        fig = plt.figure(figsize=(16, 9), dpi=160, facecolor=bg)
        fig.text(
            0.055,
            0.935,
            "매수 사이드카: A+3 진입 → R+3 청산",
            color=ink,
            fontsize=26,
            fontweight="bold",
            ha="left",
            va="top",
        )
        fig.text(
            0.055,
            0.889,
            "5분간 멈춘 프로그램 매수호가가 다시 유입되는 구간을 짧게 보유",
            color=muted,
            fontsize=11.5,
            fontweight="medium",
            ha="left",
            va="top",
        )
        fig.text(
            0.945,
            0.931,
            "PROVISIONAL  ·  n=15  ·  as of 2026-06-15",
            color=coral,
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="top",
        )

        kpi = fig.add_axes([0.055, 0.755, 0.89, 0.095], facecolor=bg)
        kpi.axis("off")
        kpis = [
            ("평균 수익률", _signed_pct(float(baseline["mean_ret"])), "gross · 첫 분 경계 시가"),
            ("10bp 비용 후", _signed_pct(float(baseline["net_10bps_mean_ret"])), "14/15 이벤트 수익"),
            ("표본 내 승률", f"{gross_wins}/{sample_count}", "규칙 동결 전 탐색 표본"),
        ]
        for index, (label, value, note) in enumerate(kpis):
            left = index / 3.0
            if index:
                kpi.plot([left, left], [0.08, 0.92], color=line, linewidth=1.0, transform=kpi.transAxes)
            kpi.text(left + 0.035, 0.78, label, color=muted, fontsize=8.5, fontweight="bold", transform=kpi.transAxes)
            kpi.text(left + 0.035, 0.34, value, color=ink, fontsize=22, fontweight="bold", transform=kpi.transAxes)
            kpi.text(left + 0.035, 0.08, note, color=muted, fontsize=8.3, fontweight="medium", transform=kpi.transAxes)

        fig.text(0.055, 0.717, "이벤트 흐름", color=ink, fontsize=12.5, fontweight="bold")
        fig.text(0.137, 0.717, "발동을 0분으로 정렬", color=muted, fontsize=8.5)
        timeline = fig.add_axes([0.065, 0.445, 0.87, 0.245], facecolor=bg)
        timeline.set_xlim(-0.55, 8.55)
        timeline.set_ylim(0.0, 1.0)
        timeline.axis("off")

        timeline.axvspan(0, 5, ymin=0.26, ymax=0.74, color=coral_soft, zorder=0)
        timeline.axvspan(1, 4, ymin=0.17, ymax=0.83, color=teal_soft, alpha=0.88, zorder=1)
        timeline.axvspan(7, 8, ymin=0.17, ymax=0.83, color=green_soft, alpha=0.95, zorder=1)
        timeline.plot([0, 5], [0.5, 0.5], color=coral, linewidth=7, solid_capstyle="round", zorder=3)
        timeline.plot([5, 8], [0.5, 0.5], color=teal, linewidth=7, solid_capstyle="round", zorder=3)
        timeline.plot([8, 8.35], [0.5, 0.5], color=line, linewidth=3, solid_capstyle="round", zorder=2)

        nodes = [
            (0, "A+0  발동", coral, 0.25, False),
            (3, "A+3  진입", teal, 0.78, True),
            (5, "R+0  해제", ink, 0.25, False),
            (8, "R+3  청산", green, 0.78, True),
        ]
        for x_value, label, color, label_y, emphasized in nodes:
            timeline.scatter(
                [x_value],
                [0.5],
                s=245 if emphasized else 165,
                color=color,
                edgecolor="white",
                linewidth=3.0 if emphasized else 2.4,
                zorder=5,
            )
            timeline.text(
                x_value,
                label_y,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=12.5 if emphasized else 10.5,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.22", "facecolor": bg, "edgecolor": "none", "alpha": 0.96},
            )

        timeline.text(2.5, 0.95, "프로그램 매수호가 효력 정지 · 5분", ha="center", color=coral, fontsize=9.5, fontweight="bold")
        timeline.text(2.5, 0.07, "강건 진입 구간  A+1 ~ A+4", ha="center", color=teal, fontsize=9.5, fontweight="bold")
        timeline.text(6.5, 0.95, "매수호가 재유입 관찰", ha="center", color=teal, fontsize=9.5, fontweight="bold")
        timeline.text(7.5, 0.07, "보수적 청산 구간  R+2 ~ R+3", ha="center", color=green, fontsize=9.5, fontweight="bold")

        entry_ax = fig.add_axes([0.06, 0.135, 0.40, 0.235], facecolor=bg)
        entry_ax.set_title("왜 A+3인가", loc="left", color=ink, fontsize=12, fontweight="bold", pad=12)
        entry_ax.text(0.0, 1.015, "R+3 청산 고정 · 평균 수익률", transform=entry_ax.transAxes, color=muted, fontsize=8.3)
        entry_delays = [0, 3, 5]
        entry_labels = ["A+0  즉시", "A+3  운영안", "A+5  해제 직전"]
        entry_values = [
            float(entry.loc[entry["entry_delay_m"].eq(delay), "mean_ret"].iloc[0]) * 100.0
            for delay in entry_delays
        ]
        entry_colors = ["#AAB5BF", teal, coral]
        y = np.arange(len(entry_labels))
        entry_ax.barh(y, entry_values, height=0.46, color=entry_colors, edgecolor="none")
        entry_ax.set_yticks(y, labels=entry_labels, color=ink, fontsize=9.8, fontweight="medium")
        entry_ax.invert_yaxis()
        entry_ax.set_xlim(0, max(entry_values) * 1.32)
        entry_ax.set_xticks([])
        entry_ax.tick_params(axis="y", length=0, pad=8)
        for index, value in enumerate(entry_values):
            entry_ax.text(value + max(entry_values) * 0.035, index, f"{value:.2f}%", va="center", color=ink, fontsize=10.8, fontweight="bold")
        for spine in entry_ax.spines.values():
            spine.set_visible(False)

        risk_ax = fig.add_axes([0.54, 0.135, 0.405, 0.235], facecolor=bg)
        risk_ax.axis("off")
        risk_ax.text(0.0, 1.04, "보유 시간을 늘리면", color=ink, fontsize=12, fontweight="bold", transform=risk_ax.transAxes)
        risk_ax.text(0.0, 0.95, "평균은 커져도 손실 꼬리가 빠르게 확대", color=muted, fontsize=8.6, fontweight="medium", transform=risk_ax.transAxes)
        risk_ax.text(0.00, 0.79, "구간", color=muted, fontsize=7.8, fontweight="bold", transform=risk_ax.transAxes)
        risk_ax.text(0.67, 0.79, "평균", color=muted, fontsize=7.8, fontweight="bold", ha="right", transform=risk_ax.transAxes)
        risk_ax.text(0.98, 0.79, "최악", color=muted, fontsize=7.8, fontweight="bold", ha="right", transform=risk_ax.transAxes)

        holding_by_rule = holding_summary.set_index("rule") if not holding_summary.empty else pd.DataFrame()
        risk_rows = [
            ("A+3 → R+3", float(baseline["mean_ret"]), float(baseline["min_ret"]), teal),
            (
                "A+3 → 당일 종가",
                _holding_value(holding_by_rule, "entry_to_close", "mean_ret"),
                _holding_value(holding_by_rule, "entry_to_close", "min_ret"),
                coral,
            ),
            (
                "당일 종가 → 다음날 종가",
                _holding_value(holding_by_rule, "close_to_next_close", "mean_ret"),
                _holding_value(holding_by_rule, "close_to_next_close", "min_ret"),
                red,
            ),
        ]
        for index, (label, mean_value, worst_value, color) in enumerate(risk_rows):
            row_y = 0.61 - index * 0.22
            risk_ax.plot([0, 1], [row_y - 0.10, row_y - 0.10], color=line, linewidth=0.8, transform=risk_ax.transAxes)
            risk_ax.scatter([0.018], [row_y], s=30, color=color, transform=risk_ax.transAxes, clip_on=False)
            risk_ax.text(0.05, row_y, label, color=ink, fontsize=9.7, fontweight="medium", va="center", transform=risk_ax.transAxes)
            risk_ax.text(0.67, row_y, _signed_pct(mean_value), color=color, fontsize=10.8, fontweight="bold", ha="right", va="center", transform=risk_ax.transAxes)
            risk_ax.text(0.98, row_y, _signed_pct(worst_value), color=ink, fontsize=9.7, fontweight="medium", ha="right", va="center", transform=risk_ax.transAxes)

        fig.text(
            0.055,
            0.047,
            "잠정 규칙 · 공식 이벤트 15건 · 진정한 OOS 아님 · bid/ask 없이 1분 OHLC로 체결 근사",
            color=muted,
            fontsize=8.8,
            fontweight="medium",
            ha="left",
        )
        fig.text(
            0.945,
            0.047,
            "KODEX 레버리지  |  first boundary open",
            color=muted,
            fontsize=8.8,
            fontweight="medium",
            ha="right",
        )

    output = output_dir / "buy_sidecar_timing_robustness.png"
    fig.savefig(output, facecolor=bg)
    plt.close(fig)
    return output


def run_study(
    parquet_dir: Path = DEFAULT_PARQUET_DIR,
    output_dir: Path = DEFAULT_OUT_DIR,
    *,
    bootstrap_samples: int = 20_000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(parquet_dir / "sidecar_events.parquet")
    prices = pd.read_parquet(parquet_dir / "kodex_leverage_1m.parquet")
    prices["dt"] = pd.to_datetime(prices["dt"])
    pairs = pair_sidecar_events(events, direction="buy")

    sensitivity = build_execution_sensitivity(pairs, prices)
    execution_summary = summarize_execution_sensitivity(sensitivity)
    close_trades = build_boundary_open_close_trades(pairs, prices)
    canonical = pd.concat(
        [
            sensitivity[sensitivity["fill_method"].eq("first_boundary_open")],
            close_trades,
        ],
        ignore_index=True,
    )
    canonical_summary = summarize_candidates(
        canonical,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    rules = build_candidate_rules()
    legacy = build_rule_trades(pairs, prices, rules)
    metadata = pd.DataFrame(
        [
            {
                "rule": rule.name,
                "entry_delay_m": rule.entry_delay_minutes,
                "exit_kind": rule.exit_anchor,
                "exit_delay_m": rule.exit_delay_minutes if rule.exit_anchor == "release" else pd.NA,
                "fill_method": "legacy_floor_close",
            }
            for rule in rules
        ]
    )
    legacy = legacy.merge(metadata, on="rule", how="left")
    legacy_summary = summarize_candidates(
        legacy,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    robust_window = select_robust_entry_window(canonical_summary, execution_summary)
    entry_rules = [f"a{delay}_r{BASELINE_EXIT_DELAY}" for delay in DEFAULT_ENTRY_DELAYS]
    exit_rules = [f"a{BASELINE_ENTRY_DELAY}_r{delay}" for delay in DEFAULT_RELEASE_DELAYS]
    entry_comparison = paired_rule_comparison(
        canonical,
        baseline_rule="a3_r3",
        candidate_rules=entry_rules,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    exit_comparison = paired_rule_comparison(
        canonical,
        baseline_rule="a3_r3",
        candidate_rules=exit_rules,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    holding_summary = build_holding_regime_summary(pairs, prices)

    pairs_path = output_dir / "buy_sidecar_pairs.csv"
    sensitivity_path = output_dir / "execution_sensitivity_trades.csv"
    execution_summary_path = output_dir / "execution_sensitivity_summary.csv"
    canonical_path = output_dir / "canonical_rule_summary.csv"
    legacy_path = output_dir / "legacy_rule_summary.csv"
    comparison_path = output_dir / "paired_comparisons.csv"
    robust_path = output_dir / "robust_entry_window.csv"
    holding_path = output_dir / "holding_regime_summary.csv"
    pairs.to_csv(pairs_path, index=False, encoding="utf-8-sig")
    sensitivity.to_csv(sensitivity_path, index=False, encoding="utf-8-sig")
    execution_summary.to_csv(execution_summary_path, index=False, encoding="utf-8-sig")
    canonical_summary.to_csv(canonical_path, index=False, encoding="utf-8-sig")
    legacy_summary.to_csv(legacy_path, index=False, encoding="utf-8-sig")
    pd.concat(
        [entry_comparison.assign(axis="entry"), exit_comparison.assign(axis="exit")],
        ignore_index=True,
    ).to_csv(comparison_path, index=False, encoding="utf-8-sig")
    robust_window.to_csv(robust_path, index=False, encoding="utf-8-sig")
    holding_summary.to_csv(holding_path, index=False, encoding="utf-8-sig")

    report_path = write_report(
        output_dir,
        pairs=pairs,
        canonical_summary=canonical_summary,
        legacy_summary=legacy_summary,
        execution_summary=execution_summary,
        robust_window=robust_window,
        entry_comparison=entry_comparison,
        exit_comparison=exit_comparison,
        holding_summary=holding_summary,
        data_as_of=prices["dt"].max(),
    )
    plot_path = plot_study(output_dir, canonical_summary, execution_summary, holding_summary)
    manifest_path = output_dir / "study_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "data_as_of": prices["dt"].max().isoformat(),
                "buy_sidecar_events": len(pairs),
                "canonical_fill": "first minute-boundary open strictly after target",
                "robust_entry_window_minutes": robust_window["entry_delay_m"].astype(int).tolist(),
                "operational_entry_minutes": BASELINE_ENTRY_DELAY,
                "conservative_exit_window_minutes": _conservative_exit_window(canonical_summary),
                "operational_exit_minutes": BASELINE_EXIT_DELAY,
                "bootstrap_seed": seed,
                "bootstrap_samples": bootstrap_samples,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "report": report_path,
        "plot": plot_path,
        "canonical_summary": canonical_path,
        "execution_summary": execution_summary_path,
        "comparisons": comparison_path,
        "manifest": manifest_path,
    }


def _conservative_exit_window(summary: pd.DataFrame) -> list[int]:
    candidates = summary[
        summary["exit_kind"].eq("release")
        & summary["entry_delay_m"].eq(BASELINE_ENTRY_DELAY)
        & pd.to_numeric(summary["exit_delay_m"], errors="coerce").le(5)
    ].copy()
    if candidates.empty:
        return []
    best_mean = float(candidates["mean_ret"].max())
    selected = candidates[
        candidates["mean_ret"].ge(best_mean * 0.95)
        & candidates["win_rate"].ge(0.90)
        & candidates["net_10bps_mean_ret"].gt(0.0)
    ]
    return selected["exit_delay_m"].dropna().astype(int).sort_values().tolist()


def _exact_sign_flip_pvalue(diff: np.ndarray) -> float:
    values = np.asarray(diff, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    observed = abs(values.mean())
    if len(values) <= 20:
        assignments = np.arange(1 << len(values), dtype=np.uint32)[:, None]
        bits = (assignments >> np.arange(len(values), dtype=np.uint32)) & 1
        signs = bits.astype(float) * 2.0 - 1.0
        distribution = np.abs((signs * values).mean(axis=1))
        return float((distribution >= observed - 1e-15).mean())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    signs = rng.choice((-1.0, 1.0), size=(100_000, len(values)))
    return float((np.abs((signs * values).mean(axis=1)) >= observed - 1e-15).mean())


def _holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values[index] * (total - rank)))
        adjusted[index] = running
    return adjusted


def _next_minute_boundary(value: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).floor("min") + pd.Timedelta(minutes=1)


def _value_at(lookup: pd.DataFrame, timestamp: pd.Timestamp, column: str) -> float:
    if timestamp not in lookup.index:
        return np.nan
    value = lookup.loc[timestamp, column]
    if hasattr(value, "iloc"):
        value = value.iloc[-1]
    return float(value)


def _pct(value: float) -> str:
    return f"{float(value) * 100:.3f}%"


def _signed_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:+.3f}%"


def _holding_value(summary: pd.DataFrame, rule: str, column: str) -> float:
    if summary.empty or rule not in summary.index or column not in summary.columns:
        return np.nan
    value = summary.loc[rule, column]
    if hasattr(value, "iloc"):
        value = value.iloc[-1]
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the buy-sidecar timing robustness study.")
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_study(
        args.parquet_dir,
        args.out_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
