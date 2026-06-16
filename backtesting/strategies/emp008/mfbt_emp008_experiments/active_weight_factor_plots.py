from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from backtesting.strategies.emp008.mfbt_emp008 import (
    _apply_expected_alpha_policy,
    _common_month_end_dates,
    _neutralize_large_benchmark_weight_factor_exposures,
    _positive_benchmark_weights,
    run_mfbt_emp008,
)
from backtesting.strategies.emp008.mfbt_emp008_data import MfbtEmp008Config, load_mfbt_emp008_market
from backtesting.strategies.emp008.mfbt_emp008_factors import build_raw_mfbt_factors
from backtesting.strategies.emp008.mfbt_emp008_preprocess import (
    build_sector_active_exposures,
    combine_exposures,
    preprocess_factor_frame,
)
from backtesting.strategies.emp008.mfbt_emp008_risk import compute_expected_alpha, fit_cross_sectional_factor_returns
from backtesting.strategies.emp008.run_weights import build_emp008_config


DEFAULT_TICKERS = ("A005930", "A000660")
TICKER_LABELS = {
    "A005930": "Samsung Electronics",
    "A000660": "SK Hynix",
}
STRATEGY_LABELS = {
    "mfbt": "MFBT",
    "mfbt_wics": "MFBT WICS neutral",
    "mfbt_zcap5": "MFBT value zcap 5",
    "origin": "Origin",
}


@dataclass(frozen=True, slots=True)
class StrategyWeights:
    target_weights: pd.DataFrame
    active_weights: pd.DataFrame


def active_weight_panel(
    active_weights_by_strategy: dict[str, pd.DataFrame],
    *,
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
) -> pd.DataFrame:
    frames: list[pd.Series] = []
    for strategy, active_weights in active_weights_by_strategy.items():
        active = active_weights.astype(float).copy()
        active.index = pd.to_datetime(active.index)
        for ticker in tickers:
            frames.append(active[ticker].rename((strategy, ticker)))

    wide = pd.concat(frames, axis=1).sort_index()
    panel_rows: list[pd.Series] = []
    for ticker in tickers:
        row = pd.DataFrame(index=wide.index)
        for strategy in active_weights_by_strategy:
            row[f"{strategy}_active"] = wide[(strategy, ticker)]
        if "origin_active" in row.columns:
            for strategy in active_weights_by_strategy:
                if strategy != "origin":
                    row[f"origin_minus_{strategy}_active"] = row["origin_active"].sub(row[f"{strategy}_active"])
        row["ticker"] = ticker
        panel_rows.append(row.reset_index(names="date").set_index(["date", "ticker"]))
    return pd.concat(panel_rows).sort_index()


def factor_driver_summary(
    contribution_panel: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "mean_contribution_bp": contribution_panel.loc[:, factor_columns].mean().mul(10_000.0),
            "mean_abs_contribution_bp": contribution_panel.loc[:, factor_columns].abs().mean().mul(10_000.0),
        }
    )
    return summary.sort_values("mean_abs_contribution_bp", ascending=False)


def build_strategy_config(
    *,
    strategy: str,
    tracking_error_annual: float,
    risk_model: str,
) -> MfbtEmp008Config:
    if strategy == "mfbt_zcap5":
        base = build_emp008_config(
            tracking_error_annual=tracking_error_annual,
            risk_model=risk_model,
            factor_set="mfbt",
        )
        return replace(base, value_zscore_cap=5.0)
    if strategy == "mfbt_wics":
        return build_emp008_config(
            tracking_error_annual=tracking_error_annual,
            risk_model=risk_model,
            factor_set="mfbt",
            sector_neutral_dataset="wics",
        )
    return build_emp008_config(
        tracking_error_annual=tracking_error_annual,
        risk_model=risk_model,
        factor_set=strategy,
    )


def load_or_run_strategy_weights(
    *,
    strategy: str,
    parquet_dir: Path,
    cache_dir: Path,
    start: str,
    end: str,
    tracking_error_annual: float,
    risk_model: str,
) -> StrategyWeights:
    strategy_cache = cache_dir / strategy
    target_path = strategy_cache / "target_weights.parquet"
    active_path = strategy_cache / "active_weights.parquet"
    if target_path.exists() and active_path.exists():
        return StrategyWeights(
            target_weights=pd.read_parquet(target_path).astype(float),
            active_weights=pd.read_parquet(active_path).astype(float),
        )

    config = build_strategy_config(
        strategy=strategy,
        tracking_error_annual=tracking_error_annual,
        risk_model=risk_model,
    )
    result = run_mfbt_emp008(
        parquet_dir=parquet_dir,
        start=start,
        end=end,
        config=config,
        output_dir=None,
    )
    strategy_cache.mkdir(parents=True, exist_ok=True)
    result.target_weights.to_parquet(target_path)
    result.active_weights.to_parquet(active_path)
    result.diagnostics.to_parquet(strategy_cache / "diagnostics.parquet")
    return StrategyWeights(target_weights=result.target_weights, active_weights=result.active_weights)


def factor_contribution_panel(
    *,
    strategy: str,
    parquet_dir: Path,
    start: str,
    end: str,
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    tracking_error_annual: float = 0.007,
    risk_model: str = "factor_idio",
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    config = build_strategy_config(
        strategy=strategy,
        tracking_error_annual=tracking_error_annual,
        risk_model=risk_model,
    )
    market = load_mfbt_emp008_market(parquet_dir=parquet_dir, start=start, end=end, config=config)
    raw_factors = build_raw_mfbt_factors(market, config)
    alpha_factor_names = tuple(raw_factors)

    close = market.frames["close"].astype(float)
    float_mktcap = market.frames["float_market_cap"].reindex(index=close.index, columns=close.columns).astype(float)
    universe = market.frames["k200_yn"].reindex(index=close.index, columns=close.columns).fillna(0).astype(bool)
    sector = market.frames["sector_neutral_big"].reindex(index=close.index, columns=close.columns).ffill()
    bm_weights = market.frames["bm_weights"].reindex(index=close.index, columns=close.columns).astype(float)

    alpha_factors = {
        name: preprocess_factor_frame(
            frame,
            float_mktcap,
            universe,
            rank_transform=name in config.rank_transform_factors,
            winsor_quantile=config.value_raw_winsor_quantile if name == "value" else None,
            zscore_cap=config.value_zscore_cap if name == "value" else None,
        )
        for name, frame in raw_factors.items()
    }
    alpha_factors = _neutralize_large_benchmark_weight_factor_exposures(alpha_factors, bm_weights, config)
    sector_factors = build_sector_active_exposures(sector, float_mktcap, universe)
    sector_factor_names = list(sector_factors)

    factor_return_rows: list[pd.Series] = []
    factor_return_dates: list[pd.Timestamp] = []
    contribution_rows: list[pd.Series] = []
    monthly_dates = _common_month_end_dates(raw_factors)
    requested_start = pd.Timestamp(start)
    requested_end = pd.Timestamp(end)

    for factor_date, return_date in zip(monthly_dates[:-1], monthly_dates[1:], strict=True):
        if return_date > requested_end:
            break
        try:
            regression_exposures = combine_exposures(alpha_factors, sector_factors, factor_date)
            stock_returns = close.loc[return_date].divide(close.loc[factor_date]).sub(1.0)
            bm = _positive_benchmark_weights(
                bm_weights.reindex(index=[return_date], columns=stock_returns.index).iloc[0]
            )
        except (KeyError, ValueError):
            continue

        excess_returns = stock_returns.sub(stock_returns.reindex(bm.index).mul(bm).sum())
        regression = fit_cross_sectional_factor_returns(regression_exposures, excess_returns)
        factor_return_rows.append(regression.factor_returns)
        factor_return_dates.append(return_date)

        factor_returns = pd.DataFrame(factor_return_rows, index=factor_return_dates).fillna(0.0)
        if len(factor_returns) < config.risk_window or return_date < requested_start:
            continue

        expected_alpha = compute_expected_alpha(
            factor_returns,
            alpha_factor_names=list(alpha_factor_names),
            sector_factor_names=sector_factor_names,
            window=config.risk_window,
        )
        expected_alpha = _apply_expected_alpha_policy(expected_alpha, config)
        target_exposures = combine_exposures(alpha_factors, sector_factors, return_date).loc[:, list(alpha_factor_names)]
        stock_contributions = target_exposures.mul(expected_alpha.reindex(alpha_factor_names), axis=1)
        for ticker in tickers:
            row = stock_contributions.loc[ticker].rename((return_date, ticker))
            row["strategy"] = strategy
            row["stock_alpha"] = float(stock_contributions.loc[ticker].sum())
            contribution_rows.append(row)

    if not contribution_rows:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"])
        return pd.DataFrame(index=empty_index), alpha_factor_names

    panel = pd.DataFrame(contribution_rows)
    panel.index = pd.MultiIndex.from_tuples(panel.index, names=["date", "ticker"])
    return panel.sort_index(), alpha_factor_names


def plot_active_weight_subplots(panel: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(len(DEFAULT_TICKERS), 1, figsize=(13.5, 7.5), sharex=True)
    if len(DEFAULT_TICKERS) == 1:
        axes = [axes]
    colors = ("#1f77b4", "#d55e00", "#2ca02c", "#4d4d4d", "#9467bd", "#8c564b")
    for ax, ticker in zip(axes, DEFAULT_TICKERS, strict=True):
        data = panel.xs(ticker, level="ticker")
        plot_columns = [column for column in data.columns if column.endswith("_active")]
        for idx, column in enumerate(plot_columns):
            ax.plot(
                data.index,
                data[column] * 100.0,
                label=_active_column_label(column),
                color=colors[idx % len(colors)],
                linewidth=1.9,
            )
        ax.axhline(0.0, color="#777777", linewidth=0.8)
        ax.set_title(f"{TICKER_LABELS.get(ticker, ticker)} active weight")
        ax.set_ylabel("Active weight (%p)")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", ncol=3, frameon=False)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_factor_contribution_subplots(
    *,
    contribution_by_strategy: dict[str, pd.DataFrame],
    active_panel: pd.DataFrame,
    factor_names_by_strategy: dict[str, tuple[str, ...]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(len(DEFAULT_TICKERS), len(contribution_by_strategy), figsize=(15.5, 8.8), sharex=True)
    strategy_order = tuple(contribution_by_strategy)
    for row_idx, ticker in enumerate(DEFAULT_TICKERS):
        active = active_panel.xs(ticker, level="ticker")
        for col_idx, strategy in enumerate(strategy_order):
            ax = axes[row_idx][col_idx]
            contributions = contribution_by_strategy[strategy].xs(ticker, level="ticker")
            factors = list(factor_names_by_strategy[strategy])
            bottom_pos = pd.Series(0.0, index=contributions.index)
            bottom_neg = pd.Series(0.0, index=contributions.index)
            for factor in factors:
                values = contributions[factor].astype(float) * 10_000.0
                bottom = values.where(values.ge(0.0), bottom_neg).where(values.lt(0.0), bottom_pos)
                ax.bar(values.index, values, bottom=bottom, width=22, label=factor, alpha=0.86)
                bottom_pos = bottom_pos.add(values.where(values.gt(0.0), 0.0), fill_value=0.0)
                bottom_neg = bottom_neg.add(values.where(values.lt(0.0), 0.0), fill_value=0.0)

            line_ax = ax.twinx()
            active_column = f"{strategy}_active"
            line_ax.plot(active.index, active[active_column] * 100.0, color="#111111", linewidth=1.5, label="active")
            line_ax.set_ylabel("Active weight (%p)")
            line_ax.grid(False)
            ax.axhline(0.0, color="#777777", linewidth=0.8)
            ax.set_title(f"{TICKER_LABELS.get(ticker, ticker)} - {STRATEGY_LABELS.get(strategy, strategy)}")
            ax.set_ylabel("Factor contribution (bp)")
            ax.grid(axis="y", alpha=0.22)
            ax.spines[["top"]].set_visible(False)
            line_ax.spines[["top"]].set_visible(False)
            ax.legend(loc="upper left", fontsize=7, ncol=2, frameon=False)
            line_ax.legend(loc="upper right", fontsize=7, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("Date")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_experiment(
    *,
    parquet_dir: Path,
    output_dir: Path,
    start: str,
    end: str,
    tracking_error_annual: float,
    risk_model: str,
    strategies: tuple[str, ...] = ("mfbt", "origin"),
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
) -> dict[str, object]:
    cache_dir = output_dir / "cache"
    weights_by_strategy = {
        strategy: load_or_run_strategy_weights(
            strategy=strategy,
            parquet_dir=parquet_dir,
            cache_dir=cache_dir,
            start=start,
            end=end,
            tracking_error_annual=tracking_error_annual,
            risk_model=risk_model,
        )
        for strategy in strategies
    }
    active_panel = active_weight_panel(
        {strategy: weights.active_weights for strategy, weights in weights_by_strategy.items()},
        tickers=tickers,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    active_csv = output_dir / "samsung_hynix_active_weights.csv"
    active_panel.to_csv(active_csv)

    contribution_by_strategy: dict[str, pd.DataFrame] = {}
    factor_names_by_strategy: dict[str, tuple[str, ...]] = {}
    summary_paths: dict[str, str] = {}
    for strategy in strategies:
        contributions, factor_names = factor_contribution_panel(
            strategy=strategy,
            parquet_dir=parquet_dir,
            start=start,
            end=end,
            tickers=tickers,
            tracking_error_annual=tracking_error_annual,
            risk_model=risk_model,
        )
        contribution_by_strategy[strategy] = contributions
        factor_names_by_strategy[strategy] = factor_names
        contribution_path = output_dir / f"{strategy}_samsung_hynix_factor_contributions.csv"
        contributions.to_csv(contribution_path)
        summary = factor_driver_summary(contributions, factor_columns=factor_names)
        summary_path = output_dir / f"{strategy}_factor_driver_summary.csv"
        summary.to_csv(summary_path)
        summary_paths[f"{strategy}_contributions_csv"] = str(contribution_path)
        summary_paths[f"{strategy}_factor_driver_summary_csv"] = str(summary_path)

    active_png = output_dir / "samsung_hynix_active_weight_subplots.png"
    factor_png = output_dir / "samsung_hynix_factor_contribution_subplots.png"
    plot_active_weight_subplots(active_panel, active_png)
    plot_factor_contribution_subplots(
        contribution_by_strategy=contribution_by_strategy,
        active_panel=active_panel,
        factor_names_by_strategy=factor_names_by_strategy,
        path=factor_png,
    )
    payload = {
        "output_dir": str(output_dir),
        "active_weights_csv": str(active_csv),
        "active_weight_subplots_png": str(active_png),
        "factor_contribution_subplots_png": str(factor_png),
        **summary_paths,
        "start": start,
        "end": end,
        "tracking_error_annual": tracking_error_annual,
        "risk_model": risk_model,
        "strategies": list(strategies),
        "tickers": list(tickers),
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    payload = run_experiment(
        parquet_dir=args.parquet_dir,
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
        tracking_error_annual=args.tracking_error_annual,
        risk_model=args.risk_model,
        strategies=tuple(args.strategies),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Samsung/SK Hynix EMP008 active weights and factor drivers.")
    parser.add_argument("--parquet-dir", type=Path, default=Path("parquet"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backtesting")
        / "strategies"
        / "emp008"
        / "mfbt_emp008_experiments"
        / "results"
        / "samsung_hynix_active_factor",
    )
    parser.add_argument("--start", default="2022-12-29")
    parser.add_argument("--end", default="2026-05-28")
    parser.add_argument("--tracking-error-annual", type=float, default=0.007)
    parser.add_argument("--risk-model", choices=("factor_idio", "direct_covariance"), default="factor_idio")
    parser.add_argument("--strategies", nargs="+", default=["mfbt", "origin"])
    return parser


def _active_column_label(column: str) -> str:
    if column.startswith("origin_minus_"):
        strategy = column.removeprefix("origin_minus_").removesuffix("_active")
        return f"Origin - {STRATEGY_LABELS.get(strategy, strategy)} active"
    strategy = column.removesuffix("_active")
    return f"{STRATEGY_LABELS.get(strategy, strategy)} active vs BM"


if __name__ == "__main__":
    main()
