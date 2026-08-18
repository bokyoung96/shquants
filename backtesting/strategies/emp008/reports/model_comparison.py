from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


_MODIFIED_COLOR = "#0F766E"
_ORIGINAL_COLOR = "#475569"
_ACCENT_COLOR = "#D97706"


@dataclass(frozen=True)
class ModelReportInput:
    label: str
    factor_set: str
    risk_model: str
    tracking_error_annual: float
    gross_run_dir: Path
    net_run_dir: Path


@dataclass(frozen=True)
class _ModelData:
    config: ModelReportInput
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    latest_weights: pd.Series
    daily_weights: pd.DataFrame


def performance_metrics(returns: pd.Series) -> dict[str, float]:
    clean = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if clean.empty:
        return {
            "total_return": float("nan"),
            "cagr": float("nan"),
            "annual_volatility": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "positive_day_rate": float("nan"),
            "positive_month_rate": float("nan"),
        }
    wealth = (1.0 + clean).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0)
    elapsed_years = len(clean) / 252.0
    cagr = float(wealth.iloc[-1] ** (1.0 / elapsed_years) - 1.0) if elapsed_years > 0 else float("nan")
    annual_volatility = float(clean.std(ddof=0) * np.sqrt(252.0))
    sharpe = float(clean.mean() * 252.0 / annual_volatility) if annual_volatility > 0 else float("nan")
    drawdown = wealth.div(wealth.cummax()).sub(1.0)
    monthly = (1.0 + clean).resample("ME").prod().sub(1.0)
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "positive_day_rate": float(clean.gt(0.0).mean()),
        "positive_month_rate": float(monthly.gt(0.0).mean()),
    }


def build_emp008_model_comparison_report(
    *,
    modified: ModelReportInput,
    original: ModelReportInput,
    adjusted_close_path: Path,
    sector_path: Path,
    output_dir: Path,
    cost_assumptions: Mapping[str, float],
) -> dict[str, object]:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    modified_data = _read_model_data(modified)
    original_data = _read_model_data(original)
    modified_data, original_data = _align_model_data(modified_data, original_data)

    metrics = _metrics_frame(modified_data, original_data)
    yearly_returns = _yearly_returns_frame(modified_data.net_returns, original_data.net_returns, modified.label, original.label)
    latest_positions = _latest_positions_frame(modified_data.latest_weights, original_data.latest_weights, modified.label, original.label)
    sector_exposure = _latest_sector_exposure(
        latest_positions,
        pd.read_parquet(sector_path),
        modified_label=modified.label,
        original_label=original.label,
        as_of=modified_data.net_returns.index[-1],
    )
    contributors = _performance_gap_contributors(
        modified_data,
        original_data,
        pd.read_parquet(adjusted_close_path),
    )

    paths = {
        "report_html": output_dir / "report.html",
        "report_json": output_dir / "report.json",
        "metrics_csv": output_dir / "metrics.csv",
        "yearly_returns_csv": output_dir / "yearly_returns.csv",
        "latest_positions_csv": output_dir / "latest_positions.csv",
        "latest_sector_exposure_csv": output_dir / "latest_sector_exposure.csv",
        "performance_gap_contributors_csv": output_dir / "performance_gap_contributors.csv",
        "cumulative_mdd_png": figures_dir / "cumulative_mdd.png",
        "yearly_cumulative_subplots_png": figures_dir / "yearly_cumulative_subplots.png",
        "return_distribution_png": figures_dir / "return_distribution.png",
        "latest_positions_png": figures_dir / "latest_positions.png",
    }
    metrics.to_csv(paths["metrics_csv"], index=False)
    yearly_returns.to_csv(paths["yearly_returns_csv"], index=False)
    latest_positions.to_csv(paths["latest_positions_csv"], index=False)
    sector_exposure.to_csv(paths["latest_sector_exposure_csv"], index=False)
    contributors.to_csv(paths["performance_gap_contributors_csv"], index=False)

    _configure_korean_font()
    _plot_cumulative_mdd(modified_data, original_data, paths["cumulative_mdd_png"])
    _plot_yearly_cumulative(modified_data, original_data, paths["yearly_cumulative_subplots_png"])
    _plot_return_distribution(modified_data, original_data, paths["return_distribution_png"])
    _plot_latest_positions(latest_positions, modified.label, original.label, paths["latest_positions_png"])

    reasons = _reason_lines(modified_data, original_data, metrics, latest_positions, sector_exposure, contributors)
    payload: dict[str, object] = {
        **{name: str(path) for name, path in paths.items()},
        "period": {
            "start": modified_data.net_returns.index[0].date().isoformat(),
            "end": modified_data.net_returns.index[-1].date().isoformat(),
            "observations": int(len(modified_data.net_returns)),
        },
        "cost_assumptions": {key: float(value) for key, value in cost_assumptions.items()},
        "metrics": metrics.to_dict(orient="records"),
        "position_active_share": float(latest_positions["weight_diff"].abs().sum() * 0.5),
        "reasons": reasons,
    }
    paths["report_json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["report_html"].write_text(
        _render_html(
            modified_data,
            original_data,
            metrics,
            yearly_returns,
            latest_positions,
            sector_exposure,
            contributors,
            reasons,
            cost_assumptions,
        ),
        encoding="utf-8",
    )
    return payload


def _read_model_data(config: ModelReportInput) -> _ModelData:
    return _ModelData(
        config=config,
        gross_returns=_read_series(config.gross_run_dir / "series" / "returns.csv", "returns"),
        net_returns=_read_series(config.net_run_dir / "series" / "returns.csv", "returns"),
        turnover=_read_series(config.net_run_dir / "series" / "turnover.csv", "turnover"),
        latest_weights=_read_latest_weights(config.net_run_dir / "positions" / "latest_weights.csv"),
        daily_weights=_read_daily_weights(config.net_run_dir / "positions" / "weights.parquet"),
    )


def _read_series(path: Path, column: str) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["date"])
    series = frame.set_index("date")[column].astype(float).sort_index()
    if not series.index.is_unique:
        raise ValueError(f"duplicate dates in {path}")
    return series


def _read_latest_weights(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    weights = frame.set_index("symbol")["target_weight"].astype(float)
    return weights[weights.abs().gt(1e-12)].sort_index()


def _read_daily_weights(path: Path) -> pd.DataFrame:
    weights = pd.read_parquet(path).astype(float)
    weights.index = pd.to_datetime(weights.index)
    return weights.sort_index()


def _align_model_data(modified: _ModelData, original: _ModelData) -> tuple[_ModelData, _ModelData]:
    common = modified.net_returns.index.intersection(original.net_returns.index)
    common = common.intersection(modified.gross_returns.index).intersection(original.gross_returns.index).sort_values()
    if len(common) < 2:
        raise ValueError("the two models need at least two common return dates")

    def aligned(data: _ModelData) -> _ModelData:
        return _ModelData(
            config=data.config,
            gross_returns=data.gross_returns.reindex(common),
            net_returns=data.net_returns.reindex(common),
            turnover=data.turnover.reindex(common).fillna(0.0),
            latest_weights=data.latest_weights,
            daily_weights=data.daily_weights.reindex(common).fillna(0.0),
        )

    return aligned(modified), aligned(original)


def _metrics_frame(modified: _ModelData, original: _ModelData) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for data in (modified, original):
        nonzero_turnover = data.turnover[data.turnover.gt(1e-12)]
        for variant, returns in (("Gross", data.gross_returns), ("Net", data.net_returns)):
            rows.append(
                {
                    "model": data.config.label,
                    "variant": variant,
                    **performance_metrics(returns),
                    "average_rebalance_turnover": float(nonzero_turnover.mean()) if not nonzero_turnover.empty else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _yearly_returns_frame(
    modified: pd.Series,
    original: pd.Series,
    modified_label: str,
    original_label: str,
) -> pd.DataFrame:
    frame = pd.DataFrame({modified_label: modified, original_label: original})
    yearly = (1.0 + frame).groupby(frame.index.year).prod().sub(1.0)
    yearly = yearly.loc[yearly.index.isin(_report_years(frame.index))]
    yearly.index.name = "year"
    yearly["성과차이"] = yearly[modified_label] - yearly[original_label]
    return yearly.reset_index()


def _latest_positions_frame(
    modified: pd.Series,
    original: pd.Series,
    modified_label: str,
    original_label: str,
) -> pd.DataFrame:
    frame = pd.concat(
        [modified.rename("modified_weight"), original.rename("original_weight")],
        axis=1,
    ).fillna(0.0)
    modified_held = frame["modified_weight"].abs().gt(1e-12)
    original_held = frame["original_weight"].abs().gt(1e-12)
    frame["status"] = np.select(
        [modified_held & original_held, modified_held, original_held],
        ["공통편입", f"{modified_label} 전용", f"{original_label} 전용"],
        default="미편입",
    )
    frame["weight_diff"] = frame["modified_weight"] - frame["original_weight"]
    frame["abs_weight_diff"] = frame["weight_diff"].abs()
    frame.index.name = "symbol"
    return frame.reset_index().sort_values(["abs_weight_diff", "symbol"], ascending=[False, True]).reset_index(drop=True)


def _latest_sector_exposure(
    latest_positions: pd.DataFrame,
    sector: pd.DataFrame,
    *,
    modified_label: str,
    original_label: str,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    sector = sector.copy()
    sector.index = pd.to_datetime(sector.index)
    eligible_dates = sector.index[sector.index <= as_of]
    if eligible_dates.empty:
        return pd.DataFrame(columns=["sector", "modified_weight", "original_weight", "weight_diff"])
    labels = sector.loc[eligible_dates[-1]].reindex(latest_positions["symbol"]).fillna("미분류")
    positions = latest_positions.set_index("symbol").copy()
    positions["sector"] = labels
    grouped = positions.groupby("sector")[["modified_weight", "original_weight"]].sum()
    grouped["weight_diff"] = grouped["modified_weight"] - grouped["original_weight"]
    grouped.index.name = "sector"
    return grouped.reset_index().sort_values("weight_diff", key=lambda values: values.abs(), ascending=False)


def _performance_gap_contributors(
    modified: _ModelData,
    original: _ModelData,
    adjusted_close: pd.DataFrame,
) -> pd.DataFrame:
    close = adjusted_close.astype(float).copy()
    close.index = pd.to_datetime(close.index)
    common_dates = modified.net_returns.index.intersection(close.index)
    symbols = modified.daily_weights.columns.union(original.daily_weights.columns).intersection(close.columns)
    asset_returns = close.reindex(index=common_dates, columns=symbols).pct_change(fill_method=None)

    def contribution(data: _ModelData) -> pd.Series:
        weights = data.daily_weights.reindex(index=common_dates, columns=symbols).fillna(0.0).shift(1).fillna(0.0)
        return weights.mul(asset_returns).replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=0)

    modified_contribution = contribution(modified)
    original_contribution = contribution(original)
    frame = pd.concat(
        [
            modified_contribution.rename("modified_contribution"),
            original_contribution.rename("original_contribution"),
        ],
        axis=1,
    ).fillna(0.0)
    frame["gap_contribution"] = frame["modified_contribution"] - frame["original_contribution"]
    frame.index.name = "symbol"
    return frame.reset_index().sort_values("gap_contribution", key=lambda values: values.abs(), ascending=False)


def _configure_korean_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"):
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False


def _plot_cumulative_mdd(modified: _ModelData, original: _ModelData, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    for data, color in ((modified, _MODIFIED_COLOR), (original, _ORIGINAL_COLOR)):
        wealth = (1.0 + data.net_returns).cumprod() * 100.0
        drawdown = wealth.div(wealth.cummax()).sub(1.0) * 100.0
        axes[0].plot(wealth.index, wealth, color=color, linewidth=2.0, label=data.config.label)
        axes[1].plot(drawdown.index, drawdown, color=color, linewidth=1.5, label=f"{data.config.label} MDD {drawdown.min():.1f}%")
        axes[1].fill_between(drawdown.index, drawdown.to_numpy(), 0.0, color=color, alpha=0.08)
    axes[0].set_title("동일 비용 기준 누적성과")
    axes[0].set_ylabel("NAV (시작=100)")
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("하락률 (%)")
    axes[1].axhline(0.0, color="#CBD5E1", linewidth=0.8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, loc="best")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_yearly_cumulative(modified: _ModelData, original: _ModelData, path: Path) -> None:
    years = _report_years(modified.net_returns.index)
    cols = 2
    rows = int(np.ceil(len(years) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(3.0 * rows, 4.0)), squeeze=False)
    for ax, year in zip(axes.flat, years, strict=False):
        for data, color in ((modified, _MODIFIED_COLOR), (original, _ORIGINAL_COLOR)):
            returns = data.net_returns[data.net_returns.index.year == year]
            wealth = (1.0 + returns).cumprod() * 100.0
            ax.plot(wealth.index, wealth, color=color, linewidth=1.7, label=data.config.label)
        ax.set_title(str(year))
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(years) :]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, loc="best")
    fig.suptitle("연도별 누적성과 (각 연도 시작=100)", y=1.01, fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _report_years(index: pd.DatetimeIndex) -> list[int]:
    counts = pd.Series(1, index=pd.DatetimeIndex(index)).groupby(lambda date: date.year).sum()
    return [int(year) for year, count in counts.items() if count >= 2]


def _plot_return_distribution(modified: _ModelData, original: _ModelData, path: Path) -> None:
    combined = pd.concat([modified.net_returns, original.net_returns]) * 100.0
    lower, upper = float(combined.min()), float(combined.max())
    if np.isclose(lower, upper):
        lower, upper = lower - 0.1, upper + 0.1
    bins = np.linspace(lower, upper, 42)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.2, 1.0]})
    for data, color in ((modified, _MODIFIED_COLOR), (original, _ORIGINAL_COLOR)):
        axes[0].hist(data.net_returns * 100.0, bins=bins, density=True, alpha=0.42, color=color, label=data.config.label)
    axes[0].axvline(0.0, color="#94A3B8", linewidth=0.9)
    axes[0].set_title("일간수익률 분포")
    axes[0].set_xlabel("일간수익률 (%)")
    axes[0].set_ylabel("밀도")
    axes[0].legend(frameon=False)
    axes[1].boxplot(
        [modified.net_returns * 100.0, original.net_returns * 100.0],
        tick_labels=[modified.config.label, original.config.label],
        orientation="vertical",
        patch_artist=True,
        boxprops={"facecolor": "#E2E8F0", "edgecolor": "#64748B"},
        medianprops={"color": _ACCENT_COLOR, "linewidth": 1.6},
    )
    axes[1].set_title("분포 범위")
    axes[1].set_ylabel("일간수익률 (%)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_latest_positions(frame: pd.DataFrame, modified_label: str, original_label: str, path: Path) -> None:
    top = frame.nlargest(15, "abs_weight_diff").sort_values("abs_weight_diff")
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(11, max(5.5, len(top) * 0.38)))
    ax.barh(y - 0.18, top["modified_weight"] * 100.0, height=0.34, color=_MODIFIED_COLOR, label=modified_label)
    ax.barh(y + 0.18, top["original_weight"] * 100.0, height=0.34, color=_ORIGINAL_COLOR, label=original_label)
    ax.set_yticks(y, top["symbol"])
    ax.set_xlabel("포트폴리오 비중 (%)")
    ax.set_title("최근 포지션 비중 차이 상위 종목")
    ax.grid(axis="x", alpha=0.18)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _reason_lines(
    modified: _ModelData,
    original: _ModelData,
    metrics: pd.DataFrame,
    positions: pd.DataFrame,
    sectors: pd.DataFrame,
    contributors: pd.DataFrame,
) -> list[str]:
    modified_net = metrics[(metrics["model"] == modified.config.label) & (metrics["variant"] == "Net")].iloc[0]
    original_net = metrics[(metrics["model"] == original.config.label) & (metrics["variant"] == "Net")].iloc[0]
    active_share = float(positions["weight_diff"].abs().sum() * 0.5)
    top_modified = positions.nlargest(3, "weight_diff")["symbol"].tolist()
    top_original = positions.nsmallest(3, "weight_diff")["symbol"].tolist()
    top_positive = contributors.nlargest(3, "gap_contribution")["symbol"].tolist()
    top_negative = contributors.nsmallest(3, "gap_contribution")["symbol"].tolist()
    sector_text = ""
    if not sectors.empty and float(sectors["weight_diff"].abs().max()) <= 1e-8:
        sector_text = " 두 모델의 최신 섹터 비중은 섹터 중립 제약으로 사실상 동일합니다."
    elif not sectors.empty:
        top_sector = sectors.iloc[0]
        sector_text = f" 최신 섹터 차이 중 가장 큰 항목은 {top_sector['sector']}({top_sector['weight_diff']:+.1%})입니다."
    return [
        (
            f"구조적으로 {modified.config.label}는 {modified.config.risk_model} 위험모형과 연 {modified.config.tracking_error_annual:.1%} "
            f"추적오차 제약을, {original.config.label}는 {original.config.risk_model} 위험모형과 연 "
            f"{original.config.tracking_error_annual:.1%} 제약을 사용합니다. 팩터셋도 각각 {modified.config.factor_set}, "
            f"{original.config.factor_set}으로 다릅니다."
        ),
        (
            f"최근 두 포트폴리오 사이 active share는 {active_share:.1%}입니다. {modified.config.label}의 상대적 고비중은 "
            f"{', '.join(top_modified)}, {original.config.label}의 상대적 고비중은 {', '.join(top_original)}입니다.{sector_text}"
        ),
        (
            f"비용 반영 CAGR 차이는 {(modified_net['cagr'] - original_net['cagr']):+.2%}p, MDD 차이는 "
            f"{(modified_net['max_drawdown'] - original_net['max_drawdown']):+.2%}p입니다."
        ),
        (
            "일별 보유비중×종목수익률 근사에서 수정EMP008에 유리했던 주요 종목은 "
            f"{', '.join(top_positive)}, 불리했던 주요 종목은 {', '.join(top_negative)}입니다. 이 값은 설명용 근사이며 "
            "복리 및 거래비용 때문에 최종 성과 차이와 정확히 일치하지 않습니다."
        ),
    ]


def _render_html(
    modified: _ModelData,
    original: _ModelData,
    metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    positions: pd.DataFrame,
    sectors: pd.DataFrame,
    contributors: pd.DataFrame,
    reasons: list[str],
    costs: Mapping[str, float],
) -> str:
    metric_view = metrics.copy()
    pct_columns = [
        "total_return",
        "cagr",
        "annual_volatility",
        "max_drawdown",
        "positive_day_rate",
        "positive_month_rate",
        "average_rebalance_turnover",
    ]
    for column in pct_columns:
        metric_view[column] = metric_view[column].map(lambda value: f"{value:.2%}")
    metric_view["sharpe"] = metric_view["sharpe"].map(lambda value: f"{value:.2f}")
    metric_view = metric_view.rename(
        columns={
            "model": "모델",
            "variant": "기준",
            "total_return": "누적수익률",
            "cagr": "CAGR",
            "annual_volatility": "연변동성",
            "sharpe": "Sharpe",
            "max_drawdown": "MDD",
            "positive_day_rate": "양의 일 비율",
            "positive_month_rate": "양의 월 비율",
            "average_rebalance_turnover": "평균 리밸런싱 회전율",
        }
    )
    yearly_view = yearly.copy()
    for column in yearly_view.columns[1:]:
        yearly_view[column] = yearly_view[column].map(lambda value: f"{value:.2%}")
    positions_view = positions.head(20).copy()
    for column in ("modified_weight", "original_weight", "weight_diff"):
        positions_view[column] = positions_view[column].map(lambda value: f"{value:.2%}")
    positions_view = positions_view[["symbol", "status", "modified_weight", "original_weight", "weight_diff"]]
    sectors_view = sectors.head(12).copy()
    for column in ("modified_weight", "original_weight", "weight_diff"):
        if column in sectors_view:
            sectors_view[column] = sectors_view[column].map(lambda value: f"{value:.2%}")
    contributor_view = pd.concat([contributors.head(6), contributors.tail(6)]).drop_duplicates("symbol").copy()
    for column in ("modified_contribution", "original_contribution", "gap_contribution"):
        contributor_view[column] = contributor_view[column].map(lambda value: f"{value:.2%}")
    reason_html = "".join(f"<li>{line}</li>" for line in reasons)
    period_start = modified.net_returns.index[0].date().isoformat()
    period_end = modified.net_returns.index[-1].date().isoformat()
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>EMP008 모델 비교 리포트</title>
<style>
:root{{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--panel:#f8fafc;--accent:#0f766e}}
*{{box-sizing:border-box}} body{{margin:0;background:#eef2f6;color:var(--ink);font-family:"Malgun Gothic","Apple SD Gothic Neo",sans-serif;line-height:1.55}}
main{{max-width:1180px;margin:32px auto;background:white;padding:44px 52px;box-shadow:0 12px 32px rgba(15,23,42,.08)}}
h1{{font-size:32px;margin:0 0 8px}} h2{{font-size:21px;margin:42px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--line)}}
.sub{{color:var(--muted)}} .chips{{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}} .chip{{background:var(--panel);border:1px solid var(--line);padding:8px 12px;border-radius:999px;font-size:13px}}
img{{width:100%;border:1px solid var(--line);border-radius:8px;background:white}} table{{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0 22px}}
th{{background:var(--panel);font-weight:600;text-align:right}} th:first-child,td:first-child{{text-align:left}} td,th{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
.reason{{background:#f0fdfa;border-left:4px solid var(--accent);padding:16px 20px;border-radius:4px}} .reason li{{margin:7px 0}} .note{{font-size:12px;color:var(--muted)}}
@media(max-width:760px){{main{{margin:0;padding:24px 18px}} table{{display:block;overflow-x:auto}}}}
</style></head><body><main>
<p class="sub">EMP008 MODEL REVIEW</p><h1>{modified.config.label} vs {original.config.label}</h1>
<p class="sub">동일 기간·동일 거래비용으로 다시 계산한 비교 리포트</p>
<div class="chips"><span class="chip">기간 {period_start} ~ {period_end}</span><span class="chip">수수료 {costs['fee']:.2%}</span><span class="chip">매도세 {costs['sell_tax']:.2%}</span><span class="chip">슬리피지 {costs['slippage']:.2%}</span></div>
<h2>핵심 성과지표</h2>{metric_view.to_html(index=False, border=0, escape=True)}
<h2>누적성과와 MDD</h2><img src="figures/cumulative_mdd.png" alt="누적성과와 MDD">
<h2>연도별 누적성과</h2><img src="figures/yearly_cumulative_subplots.png" alt="연도별 누적성과">{yearly_view.to_html(index=False, border=0, escape=True)}
<h2>Return distribution</h2><img src="figures/return_distribution.png" alt="수익률 분포">
<h2>최근 포지션</h2><img src="figures/latest_positions.png" alt="최근 포지션">{positions_view.to_html(index=False, border=0, escape=True)}
<h2>최근 섹터 노출</h2>{sectors_view.to_html(index=False, border=0, escape=True)}
<h2>최근 포지션과 성과 차이의 이유</h2><div class="reason"><ul>{reason_html}</ul></div>
<h2>성과 차이 기여 근사</h2>{contributor_view.to_html(index=False, border=0, escape=True)}
<p class="note">기여도는 전일 보유비중×종목 일간수익률의 합계입니다. 복리, 체결시점 및 거래비용 때문에 최종 성과 차이와 정확히 일치하지 않으며 원인 탐색용으로만 사용합니다.</p>
</main></body></html>"""


__all__ = ["ModelReportInput", "build_emp008_model_comparison_report", "performance_metrics"]
