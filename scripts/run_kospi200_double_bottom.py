"""Run and publish the KOSPI200 double-bottom staged-buy study."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

try:
    from scripts.kospi200_double_bottom import (
        DoubleBottomConfig,
        detect_double_bottoms,
        load_daily_ohlc,
        run_event_study,
        run_portfolio,
        summarize_equity,
    )
except ModuleNotFoundError:  # direct ``python scripts/run_...py`` execution
    from kospi200_double_bottom import (  # type: ignore[no-redef]
        DoubleBottomConfig,
        detect_double_bottoms,
        load_daily_ohlc,
        run_event_study,
        run_portfolio,
        summarize_equity,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("parquet/KOSPI200_1m.parquet")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/kospi200_double_bottom")
    )
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--holding-period", type=int, default=60)
    parser.add_argument("--min-pivot-gap", type=int, default=10)
    parser.add_argument("--max-pivot-gap", type=int, default=60)
    parser.add_argument("--low-tolerance", type=float, default=0.03)
    parser.add_argument("--min-rebound", type=float, default=0.10)
    return parser.parse_args()


def _write_chart(
    daily: pd.DataFrame, signals: pd.DataFrame, equity: pd.DataFrame, path: Path
) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib unavailable; chart omitted"
    fig, axes = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    axes[0].plot(
        daily.index,
        daily["close"],
        color="#1f4e79",
        linewidth=1.0,
        label="KOSPI200 close",
    )
    if not signals.empty:
        axes[0].scatter(
            signals["entry_date"],
            signals["entry_price"],
            color="#d55e00",
            marker="^",
            label="double-bottom entry",
        )
    axes[0].set_ylabel("Index")
    axes[0].legend(loc="upper left")
    for column in equity.columns:
        axes[1].plot(equity.index, equity[column], label=column)
    axes[1].set_ylabel("Growth of 1.0")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.2)
    fig.suptitle("KOSPI200 Double-Bottom Staged-Buy Study")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _write_report(
    path: Path,
    source: Path,
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    chart_note: str,
    config: DoubleBottomConfig,
) -> None:
    source_label = source.name
    display = summary.copy()
    percentage_columns = {
        "total_return",
        "cagr",
        "annualized_volatility",
        "max_drawdown",
        "win_rate",
    }
    for column in display.select_dtypes(include="number").columns:
        if column in percentage_columns:
            display[column] = display[column].map(lambda value: f"{value:.2%}")
        elif column == "sharpe":
            display[column] = display[column].map(lambda value: f"{value:.3f}")
        elif column == "trade_count":
            display[column] = display[column].map(lambda value: f"{int(value)}")
        else:
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    summary_table = "| " + " | ".join(display.columns) + " |\n"
    summary_table += "| " + " | ".join(["---"] * len(display.columns)) + " |\n"
    summary_table += "\n".join(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    sensitivity_table = "| " + " | ".join(sensitivity.columns) + " |\n"
    sensitivity_table += "| " + " | ".join(["---"] * len(sensitivity.columns)) + " |\n"
    sensitivity_table += "\n".join(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in sensitivity.itertuples(index=False, name=None)
    )
    signal_count = len(signals)
    text = f"""# KOSPI200 쌍바닥 이후 분할 매수 백테스트

## 결론 요약

이 결과는 `{source_label}`의 KOSPI200 연속형 분봉을 서울 현지 거래일 기준 일별 OHLC로 집계해 산출한 역사적 연구 결과입니다. 확인된 쌍바닥 신호는 **{signal_count}건**이며, 아래 표는 1회 진입 후 60거래일 보유하는 비중복 포트폴리오의 결과입니다.

{summary_table}

## 전략 scheme

- 쌍바닥: 두 거래일 저점 피벗 사이 10~60일, 두 번째 저점이 첫 저점 대비 ±{config.low_tolerance:.1%}, 중간 넥라인 반등이 첫 저점 대비 {config.min_rebound:.1%} 이상.
- 피벗은 양쪽 2거래일이 지난 뒤 확정합니다. 확정 뒤 종가가 넥라인을 처음 상향 돌파한 다음 거래일 시가에 진입합니다.
- `lump_sum`: 진입일에 100% 매수.
- `staged_50_25_25`: 진입일 50%, 5거래일 후 25%, 10거래일 후 25%를 각각 시가 매수합니다. 데이터가 끝나 추가 체결이 불가능하면 현금으로 남깁니다.
- 보유기간: 첫 체결일부터 {config.holding_period}거래일. 수수료/슬리피지는 매수·매도 각각 {config.cost_bps:.1f}bp로 차감했습니다.

## 데이터와 해석 주의

- 분석 기간: {daily.index.min().date()} ~ {daily.index.max().date()} ({len(daily):,} 거래일).
- 비교 기준은 동일한 진입일의 KOSPI200 종가 기준 단순 보유 수익률입니다.
- 반등 필터 민감도는 아래와 같습니다. 30%는 초기 권장안이지만 이 표본에서는 신호가 0건이 될 수 있어 기본값과 분리해 표시합니다.

{sensitivity_table}
- 원천 파일은 구성종목별 배당조정 주가가 아니라 KOSPI200 연속형 지수/선물형 시계열입니다. 따라서 결과를 ETF 또는 현물 구성종목 포트폴리오의 실현 수익률로 해석하면 안 됩니다.
- 신호 수가 적거나 특정 국면에 집중될 수 있으므로 통계적 유의성·미래 성과를 보장하지 않습니다. 패턴 파라미터와 고정 보유기간에 민감합니다.

## 산출물

- `signals.csv`: 피벗·넥라인·돌파·체결 원장
- `event_summary.csv`: 20/60/120거래일 이벤트 성과
- `portfolio_summary.csv`: 비중복 포트폴리오 성과
- `portfolio_equity.csv`: 일별 누적 자산곡선
- `sensitivity_summary.csv`: 반등 필터별 신호 수와 60거래일 결과
- `double_bottom_equity.png`: 가격·진입점·자산곡선
- chart: {chart_note}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    config = DoubleBottomConfig(
        cost_bps=args.cost_bps,
        holding_period=args.holding_period,
        min_pivot_gap=args.min_pivot_gap,
        max_pivot_gap=args.max_pivot_gap,
        low_tolerance=args.low_tolerance,
        min_rebound=args.min_rebound,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {"input": str(args.input), **asdict(config)}, indent=2, default=list
        ),
        encoding="utf-8",
    )
    daily = load_daily_ohlc(args.input)
    signals = detect_double_bottoms(daily, config)
    signals.to_csv(args.output_dir / "signals.csv", index=False)

    events = run_event_study(daily, signals, config)
    if events.empty:
        event_summary = pd.DataFrame(
            columns=[
                "scheme",
                "horizon_sessions",
                "count",
                "mean_return",
                "median_return",
                "win_rate",
            ]
        )
    else:
        event_summary = events.groupby(
            ["scheme", "horizon_sessions"], as_index=False
        ).agg(
            count=("strategy_return", "size"),
            mean_return=("strategy_return", "mean"),
            median_return=("strategy_return", "median"),
            win_rate=("strategy_return", lambda s: (s > 0).mean()),
        )
    events.to_csv(args.output_dir / "event_returns.csv", index=False)
    event_summary.to_csv(args.output_dir / "event_summary.csv", index=False)

    sensitivity_rows = []
    for rebound in (0.10, 0.15, 0.20, 0.25, 0.30):
        sensitivity_config = DoubleBottomConfig(
            pivot_window=config.pivot_window,
            min_pivot_gap=config.min_pivot_gap,
            max_pivot_gap=config.max_pivot_gap,
            low_tolerance=config.low_tolerance,
            min_rebound=rebound,
            holding_period=config.holding_period,
            event_horizons=(config.holding_period,),
            staged_weights=config.staged_weights,
            staged_offsets=config.staged_offsets,
            cost_bps=config.cost_bps,
        )
        sensitivity_signals = detect_double_bottoms(daily, sensitivity_config)
        row = {"min_rebound": rebound, "signal_count": len(sensitivity_signals)}
        for scheme in ("lump_sum", "staged_50_25_25"):
            series, trade_returns = run_portfolio(
                daily, sensitivity_signals, sensitivity_config, scheme
            )
            metrics = summarize_equity(series, trade_returns)
            row[f"{scheme}_total_return"] = metrics["total_return"]
            row[f"{scheme}_trade_count"] = metrics["trade_count"]
        sensitivity_rows.append(row)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(args.output_dir / "sensitivity_summary.csv", index=False)

    equity = {}
    summaries = []
    for scheme in ("lump_sum", "staged_50_25_25"):
        series, trade_returns = run_portfolio(daily, signals, config, scheme)
        equity[scheme] = series
        summaries.append({"scheme": scheme, **summarize_equity(series, trade_returns)})
    benchmark = daily["close"] / float(daily["close"].iloc[0])
    equity["buy_and_hold"] = benchmark.rename("buy_and_hold")
    summaries.append({"scheme": "buy_and_hold", **summarize_equity(benchmark)})
    equity_frame = pd.DataFrame(equity)
    equity_frame.to_csv(args.output_dir / "portfolio_equity.csv", index_label="date")
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "portfolio_summary.csv", index=False)
    chart_note = _write_chart(
        daily, signals, equity_frame, args.output_dir / "double_bottom_equity.png"
    )
    _write_report(
        args.output_dir / "report.md",
        args.input,
        daily,
        signals,
        summary,
        sensitivity,
        chart_note,
        config,
    )
    print(summary.to_string(index=False))
    print(f"signals={len(signals)} daily_rows={len(daily)} output={args.output_dir}")


if __name__ == "__main__":
    main()
