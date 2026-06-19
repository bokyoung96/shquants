from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from root import ROOT

from backtesting.analytics import summarize_perf
from backtesting.catalog import DataCatalog, DatasetId
from backtesting.data import DataLoader, LoadRequest, ParquetStore
from backtesting.data.benchmarks import benchmark_price_series
from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.execution import CostModel, WeeklySchedule
from backtesting.reporting import RunWriter
from backtesting.run import RunConfig, RunReport
from backtesting.strategies import build_strategy


START = "2020-01-01"
END = "2026-05-29"
CAPITAL = 100_000_000.0
FEE = 0.0002
SELL_TAX = 0.0015
SLIPPAGE = 0.0005
OUT_DIR = ROOT.results_path / "rrg_research" / "op_rrg_strat"
BACKTEST_DIR = ROOT.results_path / "backtests" / "op_rrg_strat_20260619_140615"

SECTOR_NAMES = {
    "WI100": "Energy",
    "WI110": "Chemicals",
    "WI200": "Nonferrous Metals",
    "WI210": "Steel",
    "WI220": "Construction",
    "WI230": "Machinery",
    "WI240": "Shipbuilding",
    "WI250": "Trading/Capital Goods",
    "WI260": "Transportation",
    "WI300": "Autos",
    "WI310": "Cosmetics/Apparel",
    "WI320": "Hotels/Leisure",
    "WI330": "Media/Education",
    "WI340": "Retail",
    "WI400": "Consumer Staples",
    "WI410": "Healthcare",
    "WI500": "Banks",
    "WI510": "Securities",
    "WI520": "Insurance",
    "WI600": "Software",
    "WI610": "IT Hardware",
    "WI620": "Semiconductors",
    "WI630": "IT Appliances",
    "WI640": "Display",
    "WI700": "Telecom Services",
    "WI800": "Utilities",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = _run_strategy()
    run_dir = report.output_dir
    if run_dir is None:
        raise RuntimeError("backtest did not produce an output directory")

    returns = report.result.returns.loc[START:END].astype(float)
    equity = report.result.equity.loc[START:END].astype(float)
    turnover = report.result.turnover.loc[START:END].astype(float)
    weights = report.result.weights.loc[START:END].fillna(0.0).astype(float)
    qty = report.result.qty.loc[START:END].fillna(0.0).astype(float)
    long100_short100 = _run_long100_short100_variant(report)
    long100_short100_returns = long100_short100.returns.loc[START:END].astype(float)
    long100_short100_equity = long100_short100.equity.loc[START:END].astype(float)

    market = _load_market(index=returns.index, columns=weights.columns)
    benchmark_returns = market["benchmark_returns"].reindex(returns.index).fillna(0.0)
    actual_weights = _actual_weights(qty=qty, close=market["close"], equity=equity)

    summary = pd.DataFrame(
        [
            _stats_row("op_rrg_strat", returns, equity, turnover, weights, benchmark_returns),
            _stats_row(
                "KOSPI200",
                benchmark_returns,
                (1.0 + benchmark_returns).cumprod().mul(float(equity.iloc[0])),
                pd.Series(0.0, index=benchmark_returns.index),
                pd.DataFrame(index=benchmark_returns.index),
                benchmark_returns,
            ),
        ]
    )
    annual = _annual_returns(returns=returns, benchmark_returns=benchmark_returns)
    latest_holdings = _latest_holdings(
        actual_weights=actual_weights,
        target_weights=weights,
        qty=qty,
        sector=market["sector"],
        name_map=market["name_map"],
    )
    latest_sector = _latest_sector_exposure(actual_weights=actual_weights, sector=market["sector"])
    monthly_sector = _monthly_sector_allocation(actual_weights=actual_weights, sector=market["sector"])
    semi_diag = _semiconductor_exposure(actual_weights=actual_weights, target_weights=weights, sector=market["sector"])

    paths = {
        "run_dir": str(run_dir),
        "summary_csv": str(OUT_DIR / "op_rrg_strat_summary.csv"),
        "annual_csv": str(OUT_DIR / "op_rrg_strat_annual_returns.csv"),
        "latest_holdings_csv": str(OUT_DIR / "op_rrg_strat_latest_holdings.csv"),
        "latest_sector_csv": str(OUT_DIR / "op_rrg_strat_latest_sector_exposure.csv"),
        "monthly_sector_csv": str(OUT_DIR / "op_rrg_strat_monthly_sector_allocation_100pct.csv"),
        "semiconductor_csv": str(OUT_DIR / "op_rrg_strat_semiconductor_exposure.csv"),
        "performance_png": str(OUT_DIR / "op_rrg_strat_performance_subplots.png"),
        "histogram_png": str(OUT_DIR / "op_rrg_strat_return_histogram.png"),
        "sector_png": str(OUT_DIR / "op_rrg_strat_sector_allocation_100pct.png"),
        "excel": str(OUT_DIR / "op_rrg_strat_report_tables.xlsx"),
        "markdown": str(OUT_DIR / "op_rrg_strat_summary.md"),
        "manifest": str(OUT_DIR / "manifest.json"),
    }

    summary.to_csv(paths["summary_csv"], index=False, encoding="utf-8-sig")
    annual.to_csv(paths["annual_csv"], index=False, encoding="utf-8-sig")
    latest_holdings.to_csv(paths["latest_holdings_csv"], index=False, encoding="utf-8-sig")
    latest_sector.to_csv(paths["latest_sector_csv"], index=False, encoding="utf-8-sig")
    monthly_sector.to_csv(paths["monthly_sector_csv"], index_label="date", encoding="utf-8-sig")
    semi_diag.to_csv(paths["semiconductor_csv"], index=False, encoding="utf-8-sig")

    _plot_performance_subplots(
        returns=returns,
        equity=equity,
        benchmark_returns=benchmark_returns,
        long100_short100_returns=long100_short100_returns,
        long100_short100_equity=long100_short100_equity,
        path=Path(paths["performance_png"]),
    )
    _plot_return_histogram(returns=returns, benchmark_returns=benchmark_returns, path=Path(paths["histogram_png"]))
    _plot_sector_allocation(monthly_sector=monthly_sector, path=Path(paths["sector_png"]))
    _write_excel(
        summary=summary,
        annual=annual,
        latest_holdings=latest_holdings,
        latest_sector=latest_sector,
        monthly_sector=monthly_sector,
        semi_diag=semi_diag,
        paths=paths,
    )
    _write_markdown(summary=summary, annual=annual, latest_sector=latest_sector, latest_holdings=latest_holdings, paths=paths)
    Path(paths["manifest"]).write_text(json.dumps(paths, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(paths, ensure_ascii=False, indent=2))


def _run_strategy():
    loader = DataLoader(DataCatalog.default(), ParquetStore(ROOT.parquet_path))
    strategy = build_strategy("op_rrg_strat")
    datasets = list(dict.fromkeys((*strategy.datasets, DatasetId.QW_ADJ_O)))
    market = loader.load(LoadRequest(datasets=datasets, start="2019-01-01", end=END))
    plan = strategy.build_plan(market)
    close = market.frames["close"].astype(float)
    open_ = market.frames["open"].reindex(index=close.index, columns=close.columns).astype(float)
    k200 = market.frames["k200_yn"].reindex_like(close).fillna(False).astype(bool)
    result = BacktestEngine(cost=CostModel(fee=FEE, sell_tax=SELL_TAX, slippage=SLIPPAGE)).run(
        close=close,
        open=open_,
        weights=plan.target_weights,
        capital=CAPITAL,
        tradable=close.notna() & k200,
        exit_tradable=close.notna(),
        schedule=WeeklySchedule(),
        fill_mode="next_open",
        allow_fractional=True,
    )
    result.equity = result.equity.loc[START:END].copy()
    result.returns = result.returns.loc[START:END].copy()
    result.weights = result.weights.loc[START:END].copy()
    result.qty = result.qty.loc[START:END].copy()
    result.turnover = result.turnover.loc[START:END].copy()
    summary = summarize_perf(result.returns)
    summary["final_equity"] = float(result.equity.iloc[-1])
    summary["avg_turnover"] = float(result.turnover.mean())
    config = RunConfig(
        start=START,
        end=END,
        capital=CAPITAL,
        strategy="op_rrg_strat",
        strategy_params={},
        name="op_rrg_strat",
        schedule="weekly",
        fill_mode="next_open",
        fee=FEE,
        sell_tax=SELL_TAX,
        slippage=SLIPPAGE,
        use_k200=True,
        warmup_days=365,
    )
    output_dir = _FixedRunWriter(
        root_dir=ROOT.results_path / "backtests",
        run_dir=BACKTEST_DIR,
        write_report_assets=False,
    ).write(
        RunReport(config=config, summary=summary, result=result, position_plan=plan)
    )
    report = RunReport(config=config, summary=summary, result=result, position_plan=plan, output_dir=output_dir)
    return report


def _run_long100_short100_variant(report: RunReport) -> BacktestResult:
    if report.position_plan is None:
        raise ValueError("position plan required for long100/short100 variant")
    loader = DataLoader(DataCatalog.default(), ParquetStore(ROOT.parquet_path))
    market = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_ADJ_C, DatasetId.QW_ADJ_O, DatasetId.QW_K200_YN],
            start="2019-01-01",
            end=END,
        )
    )
    close = market.frames["close"].astype(float)
    open_ = market.frames["open"].reindex(index=close.index, columns=close.columns).astype(float)
    k200 = market.frames["k200_yn"].reindex_like(close).fillna(False).astype(bool)
    base = report.position_plan.target_weights.reindex(index=close.index, columns=close.columns).fillna(0.0).astype(float)
    weights = base.where(base.ge(0.0), base * 2.0)
    result = BacktestEngine(cost=CostModel(fee=FEE, sell_tax=SELL_TAX, slippage=SLIPPAGE)).run(
        close=close,
        open=open_,
        weights=weights,
        capital=CAPITAL,
        tradable=close.notna() & k200,
        exit_tradable=close.notna(),
        schedule=WeeklySchedule(),
        fill_mode="next_open",
        allow_fractional=True,
    )
    result.equity = result.equity.loc[START:END].copy()
    result.returns = result.returns.loc[START:END].copy()
    result.weights = result.weights.loc[START:END].copy()
    result.qty = result.qty.loc[START:END].copy()
    result.turnover = result.turnover.loc[START:END].copy()
    return result


class _FixedRunWriter(RunWriter):
    def __init__(self, *, root_dir: Path, run_dir: Path, write_report_assets: bool) -> None:
        super().__init__(root_dir=root_dir, write_report_assets=write_report_assets)
        self._fixed_run_dir = run_dir

    def _run_dir(self, report: RunReport) -> Path:
        return self._fixed_run_dir


def _latest_op_rrg_run() -> Path | None:
    candidates = [
        path
        for path in (ROOT.results_path / "backtests").glob("op_rrg_strat_*")
        if (path / "series" / "returns.csv").exists()
        and (path / "series" / "equity.csv").exists()
        and (path / "series" / "turnover.csv").exists()
        and (path / "positions" / "weights.parquet").exists()
        and (path / "positions" / "qty.parquet").exists()
    ]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0] if candidates else None


def _load_run_report(run_dir: Path) -> RunReport:
    returns = pd.read_csv(run_dir / "series" / "returns.csv", index_col="date", parse_dates=True)["returns"].astype(float)
    equity = pd.read_csv(run_dir / "series" / "equity.csv", index_col="date", parse_dates=True)["equity"].astype(float)
    turnover = pd.read_csv(run_dir / "series" / "turnover.csv", index_col="date", parse_dates=True)["turnover"].astype(float)
    weights = pd.read_parquet(run_dir / "positions" / "weights.parquet").fillna(0.0).astype(float)
    qty = pd.read_parquet(run_dir / "positions" / "qty.parquet").fillna(0.0).astype(float)
    result = BacktestResult(
        equity=equity.loc[START:END].copy(),
        returns=returns.loc[START:END].copy(),
        weights=weights.loc[START:END].copy(),
        qty=qty.loc[START:END].copy(),
        turnover=turnover.loc[START:END].copy(),
    )
    config = RunConfig(
        start=START,
        end=END,
        capital=CAPITAL,
        strategy="op_rrg_strat",
        strategy_params={},
        name="op_rrg_strat",
        schedule="weekly",
        fill_mode="next_open",
        fee=FEE,
        sell_tax=SELL_TAX,
        slippage=SLIPPAGE,
        use_k200=True,
        warmup_days=365,
    )
    return RunReport(config=config, summary=summarize_perf(result.returns), result=result, position_plan=None, output_dir=run_dir)


def _load_market(*, index: pd.Index, columns: pd.Index) -> dict[str, object]:
    loader = DataLoader(DataCatalog.default(), ParquetStore(ROOT.parquet_path))
    market = loader.load(
        LoadRequest(
            datasets=[DatasetId.QW_ADJ_C, DatasetId.QW_WI_SEC_26_BIG, DatasetId.QW_BM],
            start=str(index.min().date()),
            end=str(index.max().date()),
        )
    )
    close = market.frames["close"].reindex(index=index, columns=columns).ffill().astype(float)
    sector = market.frames["sector_big"].reindex(index=index, columns=columns).ffill()
    benchmark = benchmark_price_series(market.frames["benchmark"], "IKS200").reindex(index).ffill().astype(float)
    name_map = pd.read_parquet(ROOT.parquet_path / "map__ticker_name_gics_sector_map.parquet").set_index("TICKER")["NAME"].to_dict()
    return {
        "close": close,
        "sector": sector,
        "benchmark_returns": benchmark.pct_change(fill_method=None).fillna(0.0).rename("KOSPI200"),
        "name_map": name_map,
    }


def _actual_weights(*, qty: pd.DataFrame, close: pd.DataFrame, equity: pd.Series) -> pd.DataFrame:
    return qty.mul(close).div(equity.reindex(qty.index), axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _stats_row(
    label: str,
    returns: pd.Series,
    equity: pd.Series,
    turnover: pd.Series,
    weights: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> dict[str, object]:
    returns = returns.reindex(benchmark_returns.index).fillna(0.0)
    summary = summarize_perf(returns)
    monthly = (1.0 + returns).resample("ME").prod().sub(1.0)
    bm_monthly = (1.0 + benchmark_returns).resample("ME").prod().sub(1.0)
    daily_downside = returns.loc[returns.lt(0.0)]
    counts = weights.reindex(index=returns.index).fillna(0.0).ne(0.0).sum(axis=1) if not weights.empty else pd.Series(dtype=float)
    active = counts.gt(0.0) if not counts.empty else pd.Series(dtype=bool)
    return {
        "strategy": label,
        "cagr": summary["cagr"],
        "mdd": summary["mdd"],
        "sharpe": summary["sharpe"],
        "sortino": float(returns.mean() / daily_downside.std(ddof=0) * np.sqrt(252.0)) if not daily_downside.empty and daily_downside.std(ddof=0) > 0 else np.nan,
        "calmar": float(summary["cagr"] / abs(summary["mdd"])) if summary["mdd"] < 0.0 else np.nan,
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "monthly_win_rate": float(monthly.gt(0.0).mean()),
        "monthly_bm_win_rate": float(monthly.sub(bm_monthly, fill_value=0.0).gt(0.0).mean()) if label != "KOSPI200" else np.nan,
        "avg_turnover": float(turnover.reindex(returns.index).fillna(0.0).mean()),
        "avg_total_count": float(counts.loc[active].mean()) if bool(active.any()) else np.nan,
        "median_total_count": float(counts.loc[active].median()) if bool(active.any()) else np.nan,
        "p90_total_count": float(counts.loc[active].quantile(0.90)) if bool(active.any()) else np.nan,
        "max_total_count": int(counts.loc[active].max()) if bool(active.any()) else 0,
    }


def _annual_returns(*, returns: pd.Series, benchmark_returns: pd.Series) -> pd.DataFrame:
    strategy = (1.0 + returns).resample("YE").prod().sub(1.0)
    benchmark = (1.0 + benchmark_returns).resample("YE").prod().sub(1.0)
    rows = []
    for date, value in strategy.items():
        bm_value = float(benchmark.loc[date])
        rows.append(
            {
                "year": int(date.year),
                "op_rrg_strat": float(value),
                "KOSPI200": bm_value,
                "excess": float(value - bm_value),
            }
        )
    return pd.DataFrame(rows)


def _latest_holdings(
    *,
    actual_weights: pd.DataFrame,
    target_weights: pd.DataFrame,
    qty: pd.DataFrame,
    sector: pd.DataFrame,
    name_map: dict[str, str],
) -> pd.DataFrame:
    date = actual_weights.index[-1]
    actual = actual_weights.loc[date]
    target = target_weights.reindex(index=actual_weights.index).fillna(0.0).loc[date]
    names = actual.loc[actual.abs().gt(1e-10)].abs().sort_values(ascending=False).index
    rows = []
    for ticker in names:
        weight = float(actual.loc[ticker])
        sector_code = sector.loc[date, ticker]
        rows.append(
            {
                "date": date.date().isoformat(),
                "ticker": ticker,
                "name": name_map.get(str(ticker), str(ticker)),
                "side": "Long" if weight > 0.0 else "Short",
                "actual_weight": weight,
                "target_weight": float(target.get(ticker, 0.0)),
                "qty": float(qty.loc[date, ticker]),
                "sector_code": sector_code,
                "sector_name": SECTOR_NAMES.get(str(sector_code), str(sector_code)),
            }
        )
    return pd.DataFrame(rows)


def _latest_sector_exposure(*, actual_weights: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    date = actual_weights.index[-1]
    row = actual_weights.loc[date]
    sector_row = sector.loc[date]
    rows = []
    for sector_code in sorted(sector_row.dropna().unique()):
        names = sector_row[sector_row.eq(sector_code)].index
        values = row.reindex(names).fillna(0.0)
        if float(values.abs().sum()) <= 1e-10:
            continue
        rows.append(
            {
                "date": date.date().isoformat(),
                "sector_code": sector_code,
                "sector_name": SECTOR_NAMES.get(str(sector_code), str(sector_code)),
                "net_exposure": float(values.sum()),
                "long_exposure": float(values.clip(lower=0.0).sum()),
                "short_exposure": float(values.clip(upper=0.0).sum()),
                "gross_exposure": float(values.abs().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("gross_exposure", ascending=False).reset_index(drop=True)


def _monthly_sector_allocation(*, actual_weights: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date in actual_weights.index:
        row = actual_weights.loc[date]
        sector_row = sector.loc[date]
        gross_by_sector = {}
        for sector_code in sector_row.dropna().unique():
            names = sector_row[sector_row.eq(sector_code)].index
            gross = float(row.reindex(names).fillna(0.0).abs().sum())
            if gross > 1e-10:
                gross_by_sector[SECTOR_NAMES.get(str(sector_code), str(sector_code))] = gross
        rows.append(pd.Series(gross_by_sector, name=date))
    daily = pd.DataFrame(rows).fillna(0.0)
    monthly = daily.resample("ME").mean()
    total = monthly.sum(axis=1).replace(0.0, np.nan)
    return monthly.divide(total, axis=0).fillna(0.0)


def _semiconductor_exposure(*, actual_weights: pd.DataFrame, target_weights: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date in actual_weights.loc["2026-01-01":].index:
        if date.weekday() != 0:
            continue
        sector_row = sector.loc[date]
        semi_names = sector_row[sector_row.eq("WI620")].index
        actual = actual_weights.loc[date]
        target = target_weights.reindex(index=actual_weights.index).fillna(0.0).loc[date]
        rows.append(
            {
                "date": date.date().isoformat(),
                "actual_semiconductor_net": float(actual.reindex(semi_names).fillna(0.0).sum()),
                "actual_semiconductor_gross": float(actual.reindex(semi_names).fillna(0.0).abs().sum()),
                "actual_samsung": float(actual.get("A005930", 0.0)),
                "actual_hynix": float(actual.get("A000660", 0.0)),
                "target_semiconductor_net": float(target.reindex(semi_names).fillna(0.0).sum()),
                "target_samsung": float(target.get("A005930", 0.0)),
                "target_hynix": float(target.get("A000660", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _plot_performance_subplots(
    *,
    returns: pd.Series,
    equity: pd.Series,
    benchmark_returns: pd.Series,
    long100_short100_returns: pd.Series,
    long100_short100_equity: pd.Series,
    path: Path,
) -> None:
    years = list(range(2020, 2027))
    bm_equity = (1.0 + benchmark_returns.reindex(returns.index).fillna(0.0)).cumprod()
    strat_growth = equity / float(equity.iloc[0])
    ls_growth = long100_short100_equity.reindex(returns.index).ffill() / float(long100_short100_equity.iloc[0])
    drawdown = strat_growth / strat_growth.cummax() - 1.0
    bm_growth = bm_equity / float(bm_equity.iloc[0])

    colors = {
        "strategy": "#2563EB",
        "long100_short100": "#059669",
        "benchmark": "#64748B",
        "drawdown": "#DC2626",
    }
    fig = plt.figure(figsize=(16, 12.8), dpi=170)
    grid = GridSpec(5, 3, figure=fig, height_ratios=[1.35, 0.58, 1.0, 1.0, 1.0], hspace=0.55, wspace=0.22)
    ax_main = fig.add_subplot(grid[0, :])
    ax_dd = fig.add_subplot(grid[1, :], sharex=ax_main)

    ax_main.plot(strat_growth.index, strat_growth, label="op_rrg_strat", linewidth=2.2, color=colors["strategy"])
    ax_main.plot(ls_growth.index, ls_growth, label="Long 100 / Short 100", linewidth=1.8, color=colors["long100_short100"])
    ax_main.plot(bm_growth.index, bm_growth, label="KOSPI200", linewidth=1.7, color=colors["benchmark"])
    ax_main.set_title("Full Period Growth", loc="left", fontsize=13, fontweight="bold")
    ax_main.set_ylabel("Growth of 1")
    ax_main.grid(True, axis="y", alpha=0.18)
    ax_main.legend(loc="upper left", frameon=False, ncol=3)
    _clean_axis(ax_main)

    ax_dd.fill_between(drawdown.index, drawdown.values, 0.0, color=colors["drawdown"], alpha=0.18, linewidth=0.0)
    ax_dd.plot(drawdown.index, drawdown, color=colors["drawdown"], linewidth=1.4, label="op_rrg_strat")
    ax_dd.set_title("Drawdown", loc="left", fontsize=11, fontweight="bold")
    ax_dd.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_dd.grid(True, axis="y", alpha=0.18)
    ax_dd.legend(loc="lower left", frameon=False, fontsize=8)
    _clean_axis(ax_dd)

    for idx, year in enumerate(years):
        ax = fig.add_subplot(grid[2 + idx // 3, idx % 3])
        year_returns = returns.loc[str(year)]
        year_bm = benchmark_returns.loc[str(year)].reindex(year_returns.index).fillna(0.0)
        year_ls = long100_short100_returns.loc[str(year)].reindex(year_returns.index).fillna(0.0)
        if year_returns.empty:
            ax.axis("off")
            continue
        strategy_year = (1.0 + year_returns).cumprod()
        bm_year = (1.0 + year_bm).cumprod()
        ls_year = (1.0 + year_ls).cumprod()
        ax.plot(strategy_year.index, strategy_year, label="strategy", linewidth=1.7, color=colors["strategy"])
        ax.plot(ls_year.index, ls_year, label="L100/S100", linewidth=1.35, color=colors["long100_short100"])
        ax.plot(bm_year.index, bm_year, label="BM", linewidth=1.25, color=colors["benchmark"])
        ax.set_title(str(year), loc="left", fontsize=10, fontweight="bold")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.grid(True, axis="y", alpha=0.16)
        _clean_axis(ax)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("op_rrg_strat Performance", x=0.06, y=0.975, ha="left", fontsize=16, fontweight="bold")
    fig.subplots_adjust(top=0.925, bottom=0.065, left=0.06, right=0.985, hspace=0.70, wspace=0.28)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_return_histogram(*, returns: pd.Series, benchmark_returns: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    ax.hist(returns, bins=80, alpha=0.65, label="op_rrg_strat")
    ax.hist(benchmark_returns.reindex(returns.index).fillna(0.0), bins=80, alpha=0.45, label="KOSPI200")
    ax.set_title("Daily Return Histogram")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_sector_allocation(*, monthly_sector: pd.DataFrame, path: Path) -> None:
    order = monthly_sector.mean().sort_values(ascending=False)
    plot_frame = monthly_sector.loc[:, order[order.gt(0.0)].index].copy()
    palette = _sector_palette(len(plot_frame.columns))
    fig, ax = plt.subplots(figsize=(16, 8), dpi=170)
    ax.stackplot(
        plot_frame.index,
        [plot_frame[column].values for column in plot_frame.columns],
        labels=plot_frame.columns,
        colors=palette,
        linewidth=0.0,
        alpha=0.92,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Sector Allocation", loc="left", fontsize=15, fontweight="bold")
    ax.set_ylabel("Share of Gross Exposure")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="y", alpha=0.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4, fontsize=8, frameon=False)
    _clean_axis(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=8)
    ax.title.set_color("#0F172A")


def _sector_palette(count: int) -> list[tuple[float, float, float, float]]:
    maps = [plt.get_cmap(name) for name in ("tab20", "tab20b", "tab20c")]
    colors: list[tuple[float, float, float, float]] = []
    for cmap in maps:
        colors.extend(cmap(i) for i in range(cmap.N))
    return colors[:count]


def _write_excel(
    *,
    summary: pd.DataFrame,
    annual: pd.DataFrame,
    latest_holdings: pd.DataFrame,
    latest_sector: pd.DataFrame,
    monthly_sector: pd.DataFrame,
    semi_diag: pd.DataFrame,
    paths: dict[str, str],
) -> None:
    with pd.ExcelWriter(paths["excel"], engine="openpyxl") as writer:
        pd.DataFrame([paths]).to_excel(writer, sheet_name="Readme", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        annual.to_excel(writer, sheet_name="Annual Returns", index=False)
        latest_sector.to_excel(writer, sheet_name="Latest Sector", index=False)
        latest_holdings.to_excel(writer, sheet_name="Latest Holdings", index=False)
        monthly_sector.to_excel(writer, sheet_name="Monthly Sector")
        semi_diag.to_excel(writer, sheet_name="Semiconductor", index=False)
        worksheet = writer.book.create_sheet("Figures")
        for row, image_path in ((1, paths["performance_png"]), (33, paths["sector_png"]), (65, paths["histogram_png"])):
            try:
                from openpyxl.drawing.image import Image

                image = Image(image_path)
                image.width = int(image.width * 0.70)
                image.height = int(image.height * 0.70)
                worksheet.add_image(image, f"A{row}")
            except Exception:
                worksheet.cell(row=row, column=1, value=image_path)


def _write_markdown(
    *,
    summary: pd.DataFrame,
    annual: pd.DataFrame,
    latest_sector: pd.DataFrame,
    latest_holdings: pd.DataFrame,
    paths: dict[str, str],
) -> None:
    strat = summary.loc[summary["strategy"].eq("op_rrg_strat")].iloc[0]
    bm = summary.loc[summary["strategy"].eq("KOSPI200")].iloc[0]
    lines = [
        "# op_rrg_strat 최종 보고서",
        "",
        "WI26 섹터를 먼저 고르고, 그 안에서 OP revision이 좋은 종목을 리더로 뽑는 롱숏 전략입니다.",
        "이번 버전은 개별 종목 OP revision을 월초가 아니라 월말 기준으로 계산합니다.",
        "",
        "## 한 줄 요약",
        "",
        "- 가격으로 먼저 뜨는 섹터를 고릅니다.",
        "- 그 섹터의 이익 컨센서스 흐름이 같이 좋은지 한 번 더 확인합니다.",
        "- 마지막으로 섹터 안에서 OP revision이 가장 좋은 종목만 압축해서 담습니다.",
        "- 기본 포지션은 롱 100%, 숏 최대 50%입니다. 숏 후보가 없으면 롱 100%만 운용합니다.",
        "",
        "## 전략 스키마",
        "",
        "### 1. 가격 RRG로 1차 섹터 선별",
        "",
        "- LONG 후보 섹터: `Leading`, `Improving`, `Weakening`",
        "- SHORT 후보 섹터: `Lagging`",
        "- 가격 RRG는 WI26 섹터 가격을 KOSPI200 대비 상대강도로 봅니다.",
        "- 중기 상대강도는 126거래일 평균 대비 현재 상대강도입니다.",
        "- 단기 모멘텀은 42거래일 평균 대비 상대강도의 21거래일 변화입니다.",
        "",
        "### 2. OP RRG로 2차 섹터 확인",
        "",
        "- LONG 후보 섹터: `Leading`, `Improving`",
        "- SHORT 후보 섹터: `Lagging`, `Weakening`",
        "- OP RRG는 섹터의 `fwd_12m` 영업이익 컨센서스 비중을 봅니다.",
        "- 계산 방식은 `섹터 fwd_12m OP / KOSPI200 전체 fwd_12m OP`입니다.",
        "- 이 OP share를 가격 RRG와 같은 방식으로 4분면 분류합니다.",
        "",
        "### 3. 섹터 안에서 종목 선별",
        "",
        "- LONG 종목 조건:",
        "- 가격 RRG long 조건을 통과한 섹터에 속해야 합니다.",
        "- OP RRG long 조건을 통과한 섹터에 속해야 합니다.",
        "- 개별 종목 OP revision이 0보다 커야 합니다.",
        "- SHORT 종목 조건:",
        "- 가격 RRG short 조건을 통과한 섹터에 속해야 합니다.",
        "- OP RRG short 조건을 통과한 섹터에 속해야 합니다.",
        "- 개별 종목 OP revision이 0보다 작아야 합니다.",
        "",
        "### 4. 개별 종목 OP revision",
        "",
        "- 사용 데이터: `qw_op_nfq1`, `qw_op_nfq2`, `qw_op_fwd_12m`",
        "- 각 데이터는 월말 값을 기준으로 전월 월말 값과 비교합니다.",
        "- 계산식: `(이번 월말 OP - 전월 월말 OP) / abs(전월 월말 OP)`",
        "- 각 revision은 -100%에서 +100%로 clip합니다.",
        "- Q1, Q2, fwd_12m revision을 평균해서 종목 OP revision 점수로 씁니다.",
        "- OP revision and OP RRG state are shifted by one trading day before `next_open` execution.",
        "",
        "### 5. 비중 산출",
        "",
        "- 선별된 전체 롱 후보를 OP revision rank로 비중화합니다.",
        "- 선별된 전체 숏 후보도 OP revision rank로 비중화합니다.",
        "- 이후 섹터별로 롱은 최대 2종목, 숏은 최대 1종목만 남깁니다.",
        "- 롱 target 합계는 100%입니다.",
        "- 숏 target 합계는 후보가 있을 때 최대 -50%입니다.",
        "- 최신처럼 숏 후보가 없으면 숏은 0%이고 롱 100%만 운용합니다.",
        "",
        "### 6. 리밸런싱과 체결",
        "",
        "- 리밸런싱: 주간",
        "- 체결: 다음 거래일 시가",
        "- 비용: 매수/매도 fee 2bp, 매도세 15bp, slippage 5bp",
        "- 백테스트 AUM은 `cash + 보유수량 * 종가`로 계산한 NAV입니다.",
        "- 리밸런싱 직후 target은 맞지만, 다음 리밸런싱 전까지 actual exposure는 가격 변동으로 흔들릴 수 있습니다.",
        "",
        "## Performance",
        "",
        "| Strategy | CAGR | MDD | Sharpe | Total Return | Monthly Win | BM 대비 월간 승률 | Avg Names |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| op_rrg_strat | {strat.cagr:.2%} | {strat.mdd:.2%} | {strat.sharpe:.3f} | {strat.total_return:.2%} | {strat.monthly_win_rate:.2%} | {strat.monthly_bm_win_rate:.2%} | {strat.avg_total_count:.2f} |",
        f"| KOSPI200 | {bm.cagr:.2%} | {bm.mdd:.2%} | {bm.sharpe:.3f} | {bm.total_return:.2%} | {bm.monthly_win_rate:.2%} |  |  |",
        "",
        "월말 OP revision 방식으로 바꾸면서 기존 월초 방식 대비 성과가 크게 낮아졌습니다.",
        "이 결과는 전략이 월중 컨센서스 변화에 얼마나 민감한지 확인해야 한다는 신호입니다.",
        "",
        "## Annual Returns",
        "",
        _markdown_table(annual, float_columns=("op_rrg_strat", "KOSPI200", "excess")),
        "",
        "## Latest Sector Exposure",
        "",
        _markdown_table(
            latest_sector.head(15),
            float_columns=("net_exposure", "long_exposure", "short_exposure", "gross_exposure"),
        ),
        "",
        "## Latest Holdings",
        "",
        _markdown_table(
            latest_holdings.head(30),
            float_columns=("actual_weight", "target_weight", "qty"),
        ),
        "",
        "## Artifacts",
        "",
        f"- 성과 subplot: `{paths['performance_png']}`",
        "- 성과 subplot에는 기본 전략, KOSPI200, 진단용 Long 100 / Short 100 버전이 함께 들어갑니다.",
        "- MDD 그래프는 기본 전략만 표시합니다.",
        f"- 섹터 배분 100% 그래프: `{paths['sector_png']}`",
        f"- 수익률 히스토그램: `{paths['histogram_png']}`",
        f"- 엑셀 표: `{paths['excel']}`",
    ]
    Path(paths["markdown"]).write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame, *, float_columns: tuple[str, ...]) -> str:
    if frame.empty:
        return "_empty_"
    columns = list(frame.columns)
    rows = []
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in float_columns and pd.notna(value):
                values.append(f"{float(value):.4f}")
            else:
                values.append("" if pd.isna(value) else str(value))
        rows.append(values)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(values) + " |" for values in rows]
    return "\n".join([header, separator, *body])


if __name__ == "__main__":
    main()
