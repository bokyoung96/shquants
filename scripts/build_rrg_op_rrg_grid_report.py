from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from root import ROOT

from backtesting.analytics import summarize_perf
from backtesting.catalog import DataCatalog, DatasetId
from backtesting.data import DataLoader, LoadRequest, ParquetStore
from backtesting.data.benchmarks import benchmark_price_series


START = "2020-01-01"
END = "2026-05-11"
GRID_SUMMARY = ROOT.results_path / "rrg_research" / "op_rrg_grid" / "grid_summary_20260618_175714.csv"
OUT_DIR = ROOT.results_path / "rrg_research" / "op_rrg_grid_report"
DOC_PATH = ROOT.root / "docs" / "research" / "rrg-op-rrg-grid-results.md"

SELECTED_IDS = [
    "rrg_op_rrg_grid_base_qavg_k2_none",
    "rrg_op_rrg_grid_base_qavg_k1_none",
    "rrg_op_rrg_grid_self_qavg_k2_none",
    "rrg_op_rrg_grid_ex10_qavg_k2_flow",
]

DISPLAY_NAMES = {
    "rrg_op_rrg_grid_base_qavg_k2_none": "base qavg k2",
    "rrg_op_rrg_grid_base_qavg_k1_none": "base qavg k1 compact",
    "rrg_op_rrg_grid_self_qavg_k2_none": "self-denom qavg k2",
    "rrg_op_rrg_grid_ex10_qavg_k2_flow": "ex10pct qavg k2 flow",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    grid = pd.read_csv(GRID_SUMMARY)
    selected = grid.loc[grid["strategy_id"].isin(SELECTED_IDS)].copy()
    selected["selection_reason"] = selected["strategy_id"].map(
        {
            "rrg_op_rrg_grid_base_qavg_k2_none": "Best qavg core: highest CAGR with shallowest MDD among high-Sharpe core variants.",
            "rrg_op_rrg_grid_base_qavg_k1_none": "Compact core: cuts average holdings to about 12 while preserving MDD near -20%.",
            "rrg_op_rrg_grid_self_qavg_k2_none": "Alternative OP denominator: checks if sector OP should be compared to its own history rather than market OP share.",
            "rrg_op_rrg_grid_ex10_qavg_k2_flow": "Ex10pct diagnostic: best risk-controlled high-index-weight exclusion case.",
        }
    )

    runs = [_load_run(row) for _, row in selected.iterrows()]
    bm_returns = _benchmark_returns(runs[0]["returns"].index)
    bm = _benchmark_pack(bm_returns)
    sector = _load_sector(runs[0]["weights"].index, runs[0]["weights"].columns)

    stats = pd.DataFrame([_stats_row(run, bm_returns) for run in runs] + [_stats_row(bm, bm_returns)])
    annual = _annual_returns([*runs, bm], bm_returns)
    selected_metrics = selected[
        [
            "strategy_id",
            "op_rrg_mode",
            "stock_score",
            "compression",
            "confirm",
            "cagr",
            "mdd",
            "sharpe",
            "monthly_win_rate",
            "monthly_bm_win_rate",
            "avg_turnover",
            "avg_total_count",
            "median_total_count",
            "p90_total_count",
            "max_total_count",
            "selection_reason",
            "output_dir",
        ]
    ].copy()

    grid.to_csv(OUT_DIR / "all_48_variant_summary.csv", index=False)
    selected_metrics.to_csv(OUT_DIR / "selected_variant_summary.csv", index=False)
    stats.to_csv(OUT_DIR / "selected_stats_vs_bm.csv", index=False)
    annual.to_csv(OUT_DIR / "selected_annual_returns_vs_bm.csv", index=False)

    plot_paths = [_plot_grid_scatter(grid), _plot_selected_comparison(runs, bm, annual)]
    for run in runs:
        plot_paths.append(_plot_strategy_dashboard(run, bm, sector))

    _write_doc(grid=grid, selected=selected_metrics, stats=stats, annual=annual, plot_paths=plot_paths)
    print(
        json.dumps(
            {
                "doc": str(DOC_PATH),
                "out_dir": str(OUT_DIR),
                "plots": [str(path) for path in plot_paths],
                "selected": selected["strategy_id"].tolist(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _load_run(row: pd.Series) -> dict[str, object]:
    path = Path(row["output_dir"])
    returns = pd.read_csv(path / "series" / "returns.csv", index_col="date", parse_dates=True)["returns"].astype(float)
    equity = pd.read_csv(path / "series" / "equity.csv", index_col="date", parse_dates=True)["equity"].astype(float)
    turnover = pd.read_csv(path / "series" / "turnover.csv", index_col="date", parse_dates=True)["turnover"].astype(float)
    weights = pd.read_parquet(path / "positions" / "weights.parquet").fillna(0.0).astype(float)
    weights.index = pd.to_datetime(weights.index)
    returns = returns.loc[START:END]
    return {
        "id": row["strategy_id"],
        "name": DISPLAY_NAMES.get(row["strategy_id"], row["strategy_id"]),
        "path": path,
        "returns": returns,
        "equity": equity.loc[returns.index.min() : returns.index.max()].reindex(returns.index).ffill(),
        "turnover": turnover.loc[returns.index.min() : returns.index.max()].reindex(returns.index).fillna(0.0),
        "weights": _effective_weights(weights.loc[returns.index.min() : returns.index.max()], returns.index),
        "meta": row.to_dict(),
    }


def _effective_weights(weights: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    aligned = weights.reindex(index).fillna(0.0)
    gross = aligned.abs().sum(axis=1)
    return aligned.mask(gross.le(1e-12)).ffill().fillna(0.0).astype(float)


def _benchmark_returns(index: pd.Index) -> pd.Series:
    loader = DataLoader(DataCatalog.default(), ParquetStore(ROOT.parquet_path))
    market = loader.load(LoadRequest(datasets=[DatasetId.QW_BM], start=str(index.min().date()), end=str(index.max().date())))
    benchmark = benchmark_price_series(market.frames["benchmark"], "IKS200").reindex(index).ffill().astype(float)
    return benchmark.pct_change(fill_method=None).fillna(0.0).rename("KOSPI200")


def _benchmark_pack(returns: pd.Series) -> dict[str, object]:
    return {
        "id": "KOSPI200",
        "name": "KOSPI200",
        "returns": returns,
        "equity": (1.0 + returns).cumprod().mul(100_000_000.0),
        "turnover": pd.Series(0.0, index=returns.index),
        "weights": pd.DataFrame(index=returns.index),
        "meta": {},
    }


def _load_sector(index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    loader = DataLoader(DataCatalog.default(), ParquetStore(ROOT.parquet_path))
    market = loader.load(LoadRequest(datasets=[DatasetId.QW_WI_SEC_26_BIG], start=str(index.min().date()), end=str(index.max().date())))
    return market.frames["sector_big"].reindex(index=index, columns=columns).ffill()


def _stats_row(pack: dict[str, object], bm_returns: pd.Series) -> dict[str, object]:
    returns = pack["returns"].reindex(bm_returns.index).fillna(0.0)
    summary = summarize_perf(returns)
    monthly = (1.0 + returns).resample("ME").prod().sub(1.0)
    bm_monthly = (1.0 + bm_returns).resample("ME").prod().sub(1.0)
    downside = returns[returns.lt(0.0)]
    var_95 = float(returns.quantile(0.05))
    counts = _position_counts(pack["weights"])
    return {
        "strategy": pack["name"],
        "cagr": summary["cagr"],
        "mdd": summary["mdd"],
        "sharpe": summary["sharpe"],
        "sortino": float(returns.mean() / downside.std(ddof=0) * np.sqrt(252.0)) if not downside.empty and downside.std(ddof=0) > 0 else np.nan,
        "calmar": float(summary["cagr"] / abs(summary["mdd"])) if summary["mdd"] < 0.0 else np.nan,
        "vol_annual": float(returns.std(ddof=0) * np.sqrt(252.0)),
        "total_return": float((1.0 + returns).prod() - 1.0),
        "daily_win_rate": float(returns.gt(0.0).mean()),
        "monthly_win_rate": float(monthly.gt(0.0).mean()),
        "monthly_bm_win_rate": float(monthly.sub(bm_monthly, fill_value=0.0).gt(0.0).mean()) if pack["name"] != "KOSPI200" else np.nan,
        "best_month": float(monthly.max()),
        "worst_month": float(monthly.min()),
        "var_95_daily": var_95,
        "cvar_95_daily": float(returns[returns.le(var_95)].mean()),
        "avg_turnover": float(pack["turnover"].reindex(bm_returns.index).fillna(0.0).mean()),
        "avg_total_count": counts["avg_total_count"],
        "median_total_count": counts["median_total_count"],
        "p90_total_count": counts["p90_total_count"],
        "max_total_count": counts["max_total_count"],
    }


def _position_counts(weights: pd.DataFrame) -> dict[str, float]:
    if weights.empty:
        return {"avg_total_count": np.nan, "median_total_count": np.nan, "p90_total_count": np.nan, "max_total_count": np.nan}
    active = weights.abs().sum(axis=1).gt(1e-12)
    counts = weights.ne(0.0).sum(axis=1).loc[active]
    return {
        "avg_total_count": float(counts.mean()) if not counts.empty else 0.0,
        "median_total_count": float(counts.median()) if not counts.empty else 0.0,
        "p90_total_count": float(counts.quantile(0.90)) if not counts.empty else 0.0,
        "max_total_count": int(counts.max()) if not counts.empty else 0,
    }


def _annual_returns(packs: list[dict[str, object]], bm_returns: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = {pack["name"]: (1.0 + pack["returns"].reindex(bm_returns.index).fillna(0.0)).resample("YE").prod().sub(1.0) for pack in packs}
    for date in values["KOSPI200"].index:
        row = {"year": int(date.year)}
        for name, series in values.items():
            row[name] = float(series.reindex([date]).iloc[0])
            if name != "KOSPI200":
                row[f"{name}_excess_vs_bm"] = row[name] - float(values["KOSPI200"].reindex([date]).iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _drawdown(equity: pd.Series) -> pd.Series:
    normalized = equity / equity.iloc[0]
    return normalized / normalized.cummax() - 1.0


def _sector_exposure(weights: pd.DataFrame, sector: pd.DataFrame) -> pd.Series:
    rows: list[pd.Series] = []
    for ts in weights.index:
        row = weights.loc[ts]
        sec = sector.loc[ts].reindex(row.index)
        gross = row.abs()
        rows.append(gross.groupby(sec).sum())
    frame = pd.DataFrame(rows, index=weights.index).fillna(0.0)
    return frame.mean().sort_values(ascending=False)


def _top_holdings(weights: pd.DataFrame, n: int = 10) -> pd.Series:
    return weights.abs().mean().sort_values(ascending=False).head(n)


def _plot_grid_scatter(grid: pd.DataFrame) -> Path:
    path = OUT_DIR / "grid_48_variant_scatter.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"base": "#2f6fbb", "ex10": "#c75146", "self": "#4c9a5b"}
    for mode, part in grid.groupby("op_rrg_mode"):
        axes[0].scatter(part["avg_total_count"], part["sharpe"], s=70, alpha=0.78, label=mode, color=colors.get(mode))
        axes[1].scatter(part["mdd"], part["cagr"], s=70, alpha=0.78, label=mode, color=colors.get(mode))
    axes[0].set_title("48 variants: holdings vs Sharpe")
    axes[0].set_xlabel("Average holdings")
    axes[0].set_ylabel("Sharpe")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_title("48 variants: MDD vs CAGR")
    axes[1].set_xlabel("MDD")
    axes[1].set_ylabel("CAGR")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_selected_comparison(runs: list[dict[str, object]], bm: dict[str, object], annual: pd.DataFrame) -> Path:
    path = OUT_DIR / "selected_comparison_dashboard.png"
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for pack in [*runs, bm]:
        equity = pack["equity"] / pack["equity"].iloc[0]
        axes[0, 0].plot(equity.index, equity, label=pack["name"], linewidth=1.8)
        axes[0, 1].plot(equity.index, _drawdown(pack["equity"]), label=pack["name"], linewidth=1.2)
    axes[0, 0].set_title("Normalized equity")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("Drawdown")
    axes[0, 1].grid(alpha=0.25)
    for run in runs:
        monthly = (1.0 + run["returns"]).resample("ME").prod().sub(1.0)
        axes[1, 0].hist(monthly, bins=24, alpha=0.45, label=run["name"])
    axes[1, 0].set_title("Monthly return distribution")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)
    plot_annual = annual.set_index("year")[[run["name"] for run in runs] + ["KOSPI200"]]
    plot_annual.plot(kind="bar", ax=axes[1, 1], width=0.78)
    axes[1, 1].set_title("Annual return vs BM")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_strategy_dashboard(run: dict[str, object], bm: dict[str, object], sector: pd.DataFrame) -> Path:
    safe_name = run["name"].replace(" ", "_").replace("/", "_")
    path = OUT_DIR / f"dashboard_{safe_name}.png"
    equity = run["equity"] / run["equity"].iloc[0]
    bm_equity = bm["equity"].reindex(equity.index).ffill() / bm["equity"].reindex(equity.index).ffill().iloc[0]
    drawdown = _drawdown(run["equity"])
    monthly = (1.0 + run["returns"]).resample("ME").prod().sub(1.0)
    weights = run["weights"]
    counts = weights.ne(0.0).sum(axis=1)
    sector_exp = _sector_exposure(weights, sector).head(10)
    holdings = _top_holdings(weights, n=10).sort_values()

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes[0, 0].plot(equity.index, equity, label=run["name"], color="#2f6fbb", linewidth=1.8)
    axes[0, 0].plot(bm_equity.index, bm_equity, label="KOSPI200", color="#666666", linewidth=1.3)
    axes[0, 0].set_title("Equity vs BM")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(drawdown.index, drawdown, color="#c75146", linewidth=1.3)
    axes[0, 1].set_title("MDD path")
    axes[0, 1].grid(alpha=0.25)

    axes[0, 2].hist(monthly, bins=22, color="#4c9a5b", alpha=0.75)
    axes[0, 2].axvline(0.0, color="#222222", linewidth=1.0)
    axes[0, 2].set_title("Monthly return distribution")
    axes[0, 2].grid(alpha=0.25)

    axes[1, 0].plot(counts.index, counts, color="#7b5ea7", linewidth=1.2)
    axes[1, 0].set_title("Position count")
    axes[1, 0].grid(alpha=0.25)

    sector_exp.sort_values().plot(kind="barh", ax=axes[1, 1], color="#d28b45")
    axes[1, 1].set_title("Sector classification: avg gross exposure")
    axes[1, 1].grid(axis="x", alpha=0.25)

    holdings.plot(kind="barh", ax=axes[1, 2], color="#5c8ca8")
    axes[1, 2].set_title("Top holdings: avg abs weight")
    axes[1, 2].grid(axis="x", alpha=0.25)
    fig.suptitle(run["name"], fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _format_pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def _write_doc(*, grid: pd.DataFrame, selected: pd.DataFrame, stats: pd.DataFrame, annual: pd.DataFrame, plot_paths: list[Path]) -> None:
    top = grid.sort_values("sharpe", ascending=False).head(12)
    ex10 = grid.loc[grid["op_rrg_mode"].eq("ex10")].sort_values("sharpe", ascending=False).head(6)
    lines = [
        "# RRG OP RRG 48-variant backtest review",
        "",
        "## Scope",
        "",
        "- Test window: 2020-01-01 to 2026-05-11.",
        "- Universe/sector: KOSPI200 with WI26 big sector classification.",
        "- Execution: weekly target generation, next-open fills, 2bp fee, 15bp sell tax, 5bp slippage, long gross 1.0 and short gross 0.5.",
        "- Variant count: 48 conceptual alternatives, not a parameter search.",
        "- OP RRG modes: `base` uses sector OP share of total market OP, `ex10` excludes every stock with BM weight greater than 10% from both sector and market OP, `self` compares sector OP to its own history.",
        "",
        "## Selection",
        "",
        _markdown_table(
            selected[
                [
                    "strategy_id",
                    "op_rrg_mode",
                    "stock_score",
                    "compression",
                    "confirm",
                    "cagr",
                    "mdd",
                    "sharpe",
                    "monthly_win_rate",
                    "monthly_bm_win_rate",
                    "avg_total_count",
                    "selection_reason",
                ]
            ],
            pct_cols={"cagr", "mdd", "monthly_win_rate", "monthly_bm_win_rate"},
        ),
        "",
        "## Top 12 By Sharpe",
        "",
        _markdown_table(
            top[["strategy_id", "cagr", "mdd", "sharpe", "monthly_bm_win_rate", "avg_total_count", "avg_turnover"]],
            pct_cols={"cagr", "mdd", "monthly_bm_win_rate", "avg_turnover"},
        ),
        "",
        "## Ex10pct Diagnostic",
        "",
        "Excluding all BM-weight-above-10% names did not improve the main qavg path. The best risk-controlled ex10pct case was `ex10_qavg_k2_flow`, but it gave up a large amount of CAGR and Sharpe versus `base_qavg_k2_none`. This points to high-index-weight OP being informative rather than merely contaminating the market OP denominator.",
        "",
        _markdown_table(
            ex10[["strategy_id", "cagr", "mdd", "sharpe", "monthly_bm_win_rate", "avg_total_count", "avg_turnover"]],
            pct_cols={"cagr", "mdd", "monthly_bm_win_rate", "avg_turnover"},
        ),
        "",
        "## Selected Stats Vs BM",
        "",
        _markdown_table(
            stats[["strategy", "cagr", "mdd", "sharpe", "calmar", "monthly_win_rate", "monthly_bm_win_rate", "avg_total_count"]],
            pct_cols={"cagr", "mdd", "monthly_win_rate", "monthly_bm_win_rate"},
        ),
        "",
        "## Annual Return",
        "",
        "2026 is YTD through 2026-05-11, not a full-year return.",
        "",
        _markdown_table(annual, pct_cols=set(annual.columns) - {"year"}),
        "",
        "## Plots",
        "",
    ]
    for path in plot_paths:
        rel = path.relative_to(ROOT.root).as_posix()
        lines.append(f"![{path.stem}](../../{rel})")
        lines.append("")
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame, *, pct_cols: set[str] | None = None) -> str:
    pct_cols = pct_cols or set()
    rows = []
    for _, row in frame.iterrows():
        rendered = []
        for col in frame.columns:
            value = row[col]
            if col == "year":
                rendered.append(str(int(value)))
            elif col in pct_cols:
                rendered.append(_format_pct(float(value)))
            elif isinstance(value, float):
                rendered.append(f"{value:.3f}")
            else:
                rendered.append(str(value))
        rows.append(rendered)
    header = "| " + " | ".join(frame.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


if __name__ == "__main__":
    main()
