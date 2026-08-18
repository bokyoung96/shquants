from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from .comparison import performance_metrics
from ..factor_registry import get_factor_set_definition


@dataclass(frozen=True, slots=True)
class ComparisonRun:
    label: str
    factor_set: str
    run_root: Path


@dataclass(frozen=True, slots=True)
class _RunData:
    config: ComparisonRun
    summary: dict[str, object]
    gross_returns: pd.Series
    net_returns: pd.Series
    latest_weights: pd.Series


_COLORS = ("#475569", "#0F766E", "#D97706", "#2563EB")


def build_adjust_comparison_report(
    *,
    runs: tuple[ComparisonRun, ...],
    benchmark_path: Path,
    output_dir: Path,
    benchmark_code: str = "IKS200",
) -> dict[str, object]:
    if len(runs) < 2:
        raise ValueError("at least two EMP008 runs are required")
    data = tuple(_read_run(run) for run in runs)
    _validate_same_risk_conditions(data)
    common_dates = _common_dates(data)
    gross = pd.concat(
        [item.gross_returns.reindex(common_dates).rename(item.config.label) for item in data],
        axis=1,
    )
    net = pd.concat(
        [item.net_returns.reindex(common_dates).rename(item.config.label) for item in data],
        axis=1,
    )
    benchmark = _benchmark_returns(benchmark_path, benchmark_code, common_dates).rename("KOSPI200 BM")
    gross = pd.concat([gross, benchmark], axis=1).fillna(0.0)
    net = pd.concat([net, benchmark], axis=1).fillna(0.0)

    net_metrics = pd.DataFrame(
        {column: performance_metrics(net[column], periods_per_year=252) for column in net.columns}
    ).T
    gross_metrics = pd.DataFrame(
        {column: performance_metrics(gross[column], periods_per_year=252) for column in gross.columns}
    ).T
    model_labels = tuple(item.config.label for item in data)
    yearly_excess = _yearly_excess_frame(net, model_labels)
    cumulative_difference = _cumulative_difference_frame(
        net,
        model_labels,
    )
    latest_positions = pd.concat(
        [item.latest_weights.rename(item.config.label) for item in data], axis=1
    ).fillna(0.0)
    latest_positions.index.name = "symbol"

    by_factor_set = {item.config.factor_set: item for item in data}
    adjust = by_factor_set.get("adjust")
    mfbt = by_factor_set.get("mfbt")
    gap_payload: dict[str, object] = {}
    if adjust is not None and mfbt is not None:
        gap = latest_positions[adjust.config.label].sub(latest_positions[mfbt.config.label])
        latest_positions["Adjust-MFBT"] = gap
        latest_positions["abs_Adjust-MFBT"] = gap.abs()
        gap_payload = {
            "adjust_vs_mfbt_active_share_pct": float(gap.abs().sum() * 50.0),
            "adjust_overweight_symbols": gap.nlargest(5).index.astype(str).tolist(),
            "adjust_underweight_symbols": gap.nsmallest(5).index.astype(str).tolist(),
        }

    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report_html": output_dir / "report.html",
        "report_json": output_dir / "report.json",
        "report_xlsx": output_dir / "comparison.xlsx",
        "daily_net_returns_csv": output_dir / "daily_net_returns.csv",
        "latest_positions_csv": output_dir / "latest_positions.csv",
        "cumulative_difference_csv": output_dir / "cumulative_difference_bp.csv",
        "cumulative_mdd_png": figures_dir / "cumulative_mdd.png",
        "cumulative_difference_png": figures_dir / "cumulative_difference.png",
        "yearly_cumulative_png": figures_dir / "yearly_cumulative.png",
        "return_distribution_png": figures_dir / "return_distribution.png",
        "latest_positions_png": figures_dir / "latest_positions.png",
    }
    net.to_csv(paths["daily_net_returns_csv"])
    latest_positions.to_csv(paths["latest_positions_csv"])
    cumulative_difference.to_csv(paths["cumulative_difference_csv"])
    with pd.ExcelWriter(paths["report_xlsx"], engine="openpyxl") as writer:
        net_metrics.to_excel(writer, sheet_name="net_metrics")
        gross_metrics.to_excel(writer, sheet_name="gross_metrics")
        net.to_excel(writer, sheet_name="daily_net_returns")
        gross.to_excel(writer, sheet_name="daily_gross_returns")
        yearly_excess.to_excel(writer, sheet_name="yearly_excess_bp")
        cumulative_difference.to_excel(writer, sheet_name="cumulative_difference_bp")
        latest_positions.to_excel(writer, sheet_name="latest_positions")

    _configure_font()
    _plot_cumulative_mdd(net, paths["cumulative_mdd_png"])
    _plot_cumulative_difference(
        cumulative_difference,
        model_labels,
        path=paths["cumulative_difference_png"],
    )
    _plot_yearly_excess(net, model_labels, paths["yearly_cumulative_png"])
    _plot_distribution(net, paths["return_distribution_png"])
    _plot_latest_positions(latest_positions, tuple(item.config.label for item in data), paths["latest_positions_png"])

    factor_sets = {
        item.config.label: [factor.value for factor in get_factor_set_definition(item.config.factor_set).factors]
        for item in data
    }
    sector_taxonomies = {item.config.label: _sector_taxonomy(item) for item in data}
    payload: dict[str, object] = {
        **{key: str(path) for key, path in paths.items()},
        "period": {
            "start": common_dates[0].date().isoformat(),
            "end": common_dates[-1].date().isoformat(),
            "days": len(common_dates),
        },
        "risk_conditions": _risk_conditions(data[0]),
        "sector_neutral_datasets": sector_taxonomies,
        "run_sources": {item.config.label: str(item.config.run_root) for item in data},
        "factor_sets": factor_sets,
        "net_metrics": json.loads(net_metrics.to_json(orient="index")),
        "gross_metrics": json.loads(gross_metrics.to_json(orient="index")),
        "yearly_excess_bp": json.loads(yearly_excess.to_json(orient="index")),
        "cumulative_difference_final_bp": {
            column: float(cumulative_difference[column].iloc[-1]) for column in cumulative_difference
        },
        "latest_position_gap": gap_payload,
    }
    paths["report_json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["report_html"].write_text(
        _render_html(
            data,
            net_metrics,
            yearly_excess,
            latest_positions,
            factor_sets,
            sector_taxonomies,
            gap_payload,
            payload["period"],
        ),
        encoding="utf-8",
    )
    return payload


def _read_run(config: ComparisonRun) -> _RunData:
    summary_path = Path(config.run_root) / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    actual_factor_set = str(summary.get("factor_set"))
    if actual_factor_set != config.factor_set:
        raise ValueError(
            f"expected factor_set={config.factor_set!r}, got {actual_factor_set!r} in {summary_path}"
        )
    gross_dir = _summary_output_dir(summary, "backtest")
    net_dir = _summary_output_dir(summary, "costed_backtest")
    gross = _read_returns(gross_dir / "series" / "returns.csv")
    net = _read_returns(net_dir / "series" / "returns.csv")
    weights = pd.read_csv(Path(config.run_root) / "weights" / "target_weights.csv", index_col=0)
    weights.index = pd.to_datetime(weights.index)
    return _RunData(
        config=config,
        summary=summary,
        gross_returns=gross,
        net_returns=net,
        latest_weights=weights.sort_index().iloc[-1].astype(float),
    )


def _summary_output_dir(summary: dict[str, object], key: str) -> Path:
    section = summary.get(key)
    if not isinstance(section, dict) or "output_dir" not in section:
        raise ValueError(f"run summary is missing {key}.output_dir")
    return Path(str(section["output_dir"]))


def _read_returns(path: Path) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return frame["returns"].astype(float).sort_index()


def _common_dates(data: tuple[_RunData, ...]) -> pd.DatetimeIndex:
    common = data[0].gross_returns.index.intersection(data[0].net_returns.index)
    for item in data[1:]:
        common = common.intersection(item.gross_returns.index).intersection(item.net_returns.index)
    common = pd.DatetimeIndex(common).sort_values()
    if len(common) < 2:
        raise ValueError("EMP008 runs need at least two common return dates")
    return common


def _risk_conditions(item: _RunData) -> dict[str, object]:
    return {
        "risk_model": item.summary.get("risk_model"),
        "tracking_error_annual": item.summary.get("tracking_error_annual"),
    }


def _sector_taxonomy(item: _RunData) -> str:
    dataset = item.summary.get("sector_neutral_dataset")
    if dataset in (None, "", "qw_wi_sec_26_big"):
        return "WI26"
    if dataset == "qw_wics_sec_big":
        return "WICS"
    return str(dataset)


def _validate_same_risk_conditions(data: tuple[_RunData, ...]) -> None:
    expected = _risk_conditions(data[0])
    mismatched = [item.config.label for item in data[1:] if _risk_conditions(item) != expected]
    if mismatched:
        raise ValueError(f"all runs must use the same risk conditions; mismatched: {', '.join(mismatched)}")


def _benchmark_returns(path: Path, code: str, index: pd.DatetimeIndex) -> pd.Series:
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index)
    if isinstance(frame.columns, pd.MultiIndex):
        close = frame[(code, "close")]
    elif code in frame.columns:
        close = frame[code]
    elif "close" in frame.columns:
        close = frame["close"]
    else:
        raise ValueError(f"benchmark close column not found in {path}")
    return close.astype(float).replace(0.0, np.nan).reindex(index).pct_change(fill_method=None).fillna(0.0)


def _cumulative_difference_frame(
    returns: pd.DataFrame,
    model_labels: tuple[str, ...],
) -> pd.DataFrame:
    wealth = (1.0 + returns).cumprod()
    benchmark = wealth["KOSPI200 BM"]
    difference = pd.DataFrame(index=returns.index)
    for label in model_labels:
        difference[f"{label} vs BM"] = wealth[label].div(benchmark).sub(1.0) * 10_000.0

    difference.index.name = "date"
    return difference


def _yearly_excess_frame(
    returns: pd.DataFrame,
    model_labels: tuple[str, ...],
) -> pd.DataFrame:
    grouped = (1.0 + returns).groupby(returns.index.year).prod()
    excess = grouped.loc[:, list(model_labels)].div(grouped["KOSPI200 BM"], axis=0).sub(1.0) * 10_000.0
    excess.index.name = "year"
    return excess


def _yearly_cumulative_excess_frame(
    returns: pd.DataFrame,
    model_labels: tuple[str, ...],
    year: int,
) -> pd.DataFrame:
    subset = returns[returns.index.year == year]
    wealth = (1.0 + subset).cumprod()
    excess = wealth.loc[:, list(model_labels)].div(wealth["KOSPI200 BM"], axis=0).sub(1.0) * 10_000.0
    prior_dates = returns.index[returns.index < subset.index[0]]
    baseline_date = prior_dates[-1] if len(prior_dates) else subset.index[0] - pd.Timedelta(days=1)
    baseline = pd.DataFrame(0.0, index=[baseline_date], columns=excess.columns)
    return pd.concat([baseline, excess])


def _configure_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"):
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False


def _plot_cumulative_mdd(returns: pd.DataFrame, path: Path) -> None:
    wealth = (1.0 + returns).cumprod() * 100.0
    drawdown = wealth.div(wealth.cummax()).sub(1.0) * 100.0
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    for column, color in zip(returns.columns, _COLORS, strict=False):
        axes[0].plot(wealth.index, wealth[column], label=column, color=color, linewidth=1.9)
        axes[1].plot(drawdown.index, drawdown[column], label=f"{column} ({drawdown[column].min():.1f}%)", color=color, linewidth=1.35)
    axes[0].set_title("EMP008 누적성과 (비용 반영)")
    axes[0].set_ylabel("NAV (시작=100)")
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("하락률 (%)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_cumulative_difference(
    difference: pd.DataFrame,
    model_labels: tuple[str, ...],
    *,
    path: Path,
) -> None:
    benchmark_columns = [f"{label} vs BM" for label in model_labels]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for column, color in zip(benchmark_columns, _COLORS, strict=False):
        values = difference[column].astype(float)
        ax.plot(values.index, values, color=color, linewidth=1.5, label=column)
        ax.fill_between(values.index, 0.0, values.to_numpy(), color=color, alpha=0.14)
    ax.set_title("KOSPI200 대비 누적 초과성과")
    ax.set_xlabel("Date")
    ax.axhline(0.0, color="#94A3B8", linewidth=0.8)
    ax.set_ylabel("누적 초과성과 (bp)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_yearly_excess(
    returns: pd.DataFrame,
    model_labels: tuple[str, ...],
    path: Path,
) -> None:
    years = sorted(set(returns.index.year))
    rows = int(np.ceil(len(years) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, max(3.2 * rows, 4.2)), squeeze=False)
    for ax, year in zip(axes.flat, years, strict=False):
        yearly_excess = _yearly_cumulative_excess_frame(returns, model_labels, year)
        for column, color in zip(model_labels, _COLORS, strict=False):
            ax.plot(yearly_excess.index, yearly_excess[column], label=column, color=color, linewidth=1.5)
        ax.set_title(str(year))
        ax.axhline(0.0, color="#94A3B8", linewidth=0.8)
        ax.set_ylabel("초과성과 (bp)")
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(years):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("연도별 KOSPI200 대비 누적 초과성과", y=1.01, fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_distribution(returns: pd.DataFrame, path: Path) -> None:
    values = returns * 100.0
    lower = float(values.min().min())
    upper = float(values.max().max())
    bins = np.linspace(lower, upper, 45) if not np.isclose(lower, upper) else 10
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for column, color in zip(values.columns, _COLORS, strict=False):
        ax.hist(values[column], bins=bins, density=True, alpha=0.3, label=column, color=color)
    ax.axvline(0.0, color="#94A3B8", linewidth=0.9)
    ax.set_title("일간 수익률 분포")
    ax.set_xlabel("일간 수익률 (%)")
    ax.set_ylabel("밀도")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_latest_positions(frame: pd.DataFrame, labels: tuple[str, ...], path: Path) -> None:
    top_symbols = frame.loc[:, list(labels)].max(axis=1).nlargest(15).index
    top = frame.loc[top_symbols, list(labels)].sort_values(labels[-1])
    y = np.arange(len(top))
    width = 0.75 / len(labels)
    fig, ax = plt.subplots(figsize=(11, max(5.5, len(top) * 0.38)))
    for idx, (label, color) in enumerate(zip(labels, _COLORS, strict=False)):
        offset = (idx - (len(labels) - 1) / 2) * width
        ax.barh(y + offset, top[label] * 100.0, height=width * 0.9, color=color, label=label)
    ax.set_yticks(y, top.index.astype(str))
    ax.set_xlabel("포트폴리오 비중 (%)")
    ax.set_title("최근 포지션 상위 종목")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _render_html(
    data: tuple[_RunData, ...],
    metrics: pd.DataFrame,
    yearly_excess: pd.DataFrame,
    positions: pd.DataFrame,
    factor_sets: dict[str, list[str]],
    sector_taxonomies: dict[str, str],
    gap: dict[str, object],
    period: object,
) -> str:
    metric_view = metrics[["cagr_pct", "annual_vol_pct", "sharpe", "max_drawdown_pct", "total_return_pct"]].copy()
    metric_view.columns = ["CAGR (%)", "연 변동성 (%)", "Sharpe", "MDD (%)", "누적수익률 (%)"]
    metric_view.index.name = "모델"
    yearly_excess_view = yearly_excess.round(1)
    factor_rows = pd.DataFrame(
        [
            {
                "모델": label,
                "업종 중립화": sector_taxonomies[label],
                "사용 팩터": ", ".join(factors),
            }
            for label, factors in factor_sets.items()
        ]
    )
    position_view = positions.sort_values("abs_Adjust-MFBT", ascending=False).head(20) if "abs_Adjust-MFBT" in positions else positions.head(20)
    position_labels = {
        item.config.label: {"origin": "기존", "mfbt": "1차", "adjust": "2차"}.get(
            item.config.factor_set,
            item.config.label,
        )
        for item in data
    }
    position_labels.update({"Adjust-MFBT": "2차-1차", "abs_Adjust-MFBT": "절대차이"})
    position_view = (position_view * 100.0).round(3).rename(columns=position_labels)
    position_view.index.name = "종목코드"
    best_cagr = metric_view["CAGR (%)"].idxmax()
    best_mdd = metric_view["MDD (%)"].idxmax()
    gap_text = ""
    if gap:
        gap_text = (
            f"최근 2차와 1차의 active share는 {gap['adjust_vs_mfbt_active_share_pct']:.2f}%입니다. "
            f"2차의 1차 대비 고비중 종목은 {', '.join(gap['adjust_overweight_symbols'])}, "
            f"저비중 종목은 {', '.join(gap['adjust_underweight_symbols'])}입니다."
        )
    period_map = period if isinstance(period, dict) else {}
    conditions = _risk_conditions(data[0])
    report_title = " · ".join([item.config.label for item in data] + ["KOSPI200 BM"])
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>EMP008 Adjust 비교 보고서</title><style>
:root{{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--panel:#f8fafc;--accent:#0f766e}}
*{{box-sizing:border-box}} body{{margin:0;background:#eef2f6;color:var(--ink);font-family:"Malgun Gothic",sans-serif;line-height:1.55}}
main{{max-width:1180px;margin:32px auto;background:#fff;padding:44px 52px;box-shadow:0 12px 32px rgba(15,23,42,.08)}}
h1{{font-size:31px;margin:0 0 8px}} h2{{font-size:21px;margin:40px 0 14px;border-bottom:1px solid var(--line);padding-bottom:8px}}
.sub{{color:var(--muted)}} .summary{{background:#f0fdfa;border-left:4px solid var(--accent);padding:15px 19px;border-radius:4px}}
img{{width:100%;border:1px solid var(--line);border-radius:8px}} table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:var(--panel)}} td,th{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right}} th:first-child,td:first-child{{text-align:left}}
@media(max-width:760px){{main{{margin:0;padding:24px 18px}} table{{display:block;overflow-x:auto}}}}
</style></head><body><main>
<p class="sub">EMP008 MODEL COMPARISON</p><h1>{report_title}</h1>
<p class="sub">기간 {period_map.get('start')} ~ {period_map.get('end')} · {conditions['risk_model']} · 연 TE {float(conditions['tracking_error_annual']):.2%} · 업종 중립화는 모델별 적용</p>
<div class="summary"><b>핵심 요약</b><br>비용 반영 CAGR 최고 모델: {best_cagr}. MDD가 가장 작은 모델: {best_mdd}. {gap_text}</div>
<h2>팩터 구성</h2>{factor_rows.to_html(index=False, border=0, escape=True)}
<h2>성과지표</h2>{metric_view.round(3).to_html(border=0, escape=True)}
<h2>누적성과와 MDD</h2><img src="figures/cumulative_mdd.png" alt="누적성과와 MDD">
<h2>KOSPI200 대비 누적 초과성과</h2><img src="figures/cumulative_difference.png" alt="KOSPI200 대비 누적 초과성과"><p class="sub">각 EMP008의 KOSPI200 대비 누적 초과성과를 bp 단위로 표시합니다.</p>
<h2>연도별 KOSPI200 대비 누적 초과성과</h2><img src="figures/yearly_cumulative.png" alt="연도별 KOSPI200 대비 누적 초과성과">{yearly_excess_view.to_html(border=0, escape=True)}
<h2>Return distribution</h2><img src="figures/return_distribution.png" alt="수익률 분포">
<h2>최근 포지션</h2><img src="figures/latest_positions.png" alt="최근 포지션">{position_view.to_html(border=0, escape=True)}
<h2>성과 차이 해석</h2><p>기존 EMP008은 WI26, 1차수정 EMP008과 2차수정 EMP008은 WICS 기준으로 업종 중립화했습니다. 2차수정 EMP008은 1차수정 EMP008의 price_to_252d_high를 momentum_12_1m으로 교체하고 retail_flow를 제외했습니다. 위험모형, TE와 거래비용은 동일하므로 성과 차이는 팩터 구성과 업종 중립화 기준에서 발생한 종목별 비중 차이로 해석합니다.</p>
</main></body></html>"""


__all__ = ["ComparisonRun", "build_adjust_comparison_report"]
