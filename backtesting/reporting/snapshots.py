from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analytics import (
    ROLLING_WINDOW,
    SECTOR_CONTRIBUTION_METHOD_WEIGHTED_ASSET_RETURNS,
    DrawdownStats,
    ExposureSnapshot,
    PerformanceMetrics,
    ResearchSnapshot,
    RollingMetrics,
    SectorSnapshot,
    annualized_sharpe,
    build_monthly_heatmap,
    build_return_distribution,
    build_yearly_excess_returns,
    monthly_return_series,
)
from .benchmarks import BenchmarkRepository, SectorRepository
from .models import BenchmarkConfig, SavedRun


def _default_benchmark() -> BenchmarkConfig:
    return BenchmarkConfig.default_kospi200()


@dataclass(frozen=True, slots=True)
class PerformanceSnapshot:
    run_id: str
    display_name: str
    metrics: PerformanceMetrics
    rolling: RollingMetrics
    drawdowns: DrawdownStats
    exposure: ExposureSnapshot
    sectors: SectorSnapshot
    strategy_equity: pd.Series
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    benchmark_equity: pd.Series
    strategy_name: str = "unknown"
    benchmark: BenchmarkConfig = field(default_factory=_default_benchmark)
    research: ResearchSnapshot = field(default_factory=ResearchSnapshot)


class PerformanceSnapshotFactory:
    def __init__(self, benchmark_repo: BenchmarkRepository, sector_repo: SectorRepository) -> None:
        self.benchmark_repo = benchmark_repo
        self.sector_repo = sector_repo

    def build(self, run: SavedRun, benchmark: BenchmarkConfig) -> PerformanceSnapshot:
        strategy_returns = run.returns.astype(float).sort_index()
        strategy_equity = run.equity.astype(float).sort_index()
        benchmark_series = self.benchmark_repo.load_series(
            benchmark,
            start=str(strategy_returns.index.min().date()),
            end=str(strategy_returns.index.max().date()),
        )
        benchmark_returns = benchmark_series.returns.reindex(strategy_returns.index).fillna(0.0).astype(float)
        benchmark_equity = self._equity_from_returns(benchmark_returns, starting_value=float(strategy_equity.iloc[0]))

        rolling = self._build_rolling_metrics(strategy_returns, benchmark_returns)
        drawdowns = self._build_drawdowns(strategy_equity)
        exposure = self._build_exposure(run)
        sectors = self._build_sectors(run.weights)
        research = self._build_research(run, strategy_returns, benchmark_returns, drawdowns)
        metrics = self._build_metrics(run, strategy_returns, strategy_equity, benchmark_returns, drawdowns.underwater)

        return PerformanceSnapshot(
            run_id=run.run_id,
            display_name=str(run.config.get("name") or run.run_id),
            strategy_name=str(run.config.get("strategy") or "unknown"),
            benchmark=benchmark,
            metrics=metrics,
            rolling=rolling,
            drawdowns=drawdowns,
            exposure=exposure,
            sectors=sectors,
            research=research,
            strategy_equity=strategy_equity,
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            benchmark_equity=benchmark_equity,
        )

    def _build_metrics(
        self,
        run: SavedRun,
        strategy_returns: pd.Series,
        strategy_equity: pd.Series,
        benchmark_returns: pd.Series,
        underwater: pd.Series,
    ) -> PerformanceMetrics:
        active_returns = strategy_returns.sub(benchmark_returns, fill_value=0.0)
        benchmark_variance = float(benchmark_returns.var(ddof=0))
        covariance = float(strategy_returns.cov(benchmark_returns, ddof=0))
        beta = 0.0 if abs(benchmark_variance) < 1e-12 else covariance / benchmark_variance
        alpha = float((strategy_returns.mean() - beta * benchmark_returns.mean()) * 252.0)
        annual_volatility = float(strategy_returns.std(ddof=0) * (252.0**0.5)) if len(strategy_returns) > 1 else 0.0
        downside_returns = strategy_returns.clip(upper=0.0)
        downside_volatility = float((downside_returns.pow(2).mean() ** 0.5) * (252.0**0.5))
        tracking_error = float(active_returns.std(ddof=0) * (252.0**0.5)) if len(active_returns) > 1 else 0.0
        information_ratio = 0.0 if abs(tracking_error) < 1e-12 else float(active_returns.mean() * 252.0 / tracking_error)
        cumulative_return = float(strategy_equity.iloc[-1] / strategy_equity.iloc[0] - 1.0)
        cagr = self._cagr(strategy_equity)
        max_drawdown = float(underwater.min()) if not underwater.empty else 0.0
        sortino = 0.0 if abs(downside_volatility) < 1e-12 else float(strategy_returns.mean() * 252.0 / downside_volatility)
        calmar = 0.0 if max_drawdown >= 0.0 else float(cagr / abs(max_drawdown))

        return PerformanceMetrics(
            cumulative_return=cumulative_return,
            cagr=cagr,
            annual_volatility=annual_volatility,
            sharpe=annualized_sharpe(strategy_returns),
            sortino=sortino,
            calmar=calmar,
            max_drawdown=max_drawdown,
            final_equity=float(strategy_equity.iloc[-1]),
            avg_turnover=float(run.turnover.fillna(0.0).mean()),
            alpha=alpha,
            beta=float(beta),
            tracking_error=tracking_error,
            information_ratio=information_ratio,
        )

    def _build_rolling_metrics(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> RollingMetrics:
        window = ROLLING_WINDOW
        rolling_sharpe = strategy_returns.rolling(window=window, min_periods=252).apply(
            lambda values: annualized_sharpe(pd.Series(values)),
            raw=False,
        )
        benchmark_variance = benchmark_returns.rolling(window=window, min_periods=252).var(ddof=0)
        rolling_beta = strategy_returns.rolling(window=window, min_periods=252).cov(benchmark_returns, ddof=0).div(
            benchmark_variance
        )
        rolling_beta = rolling_beta.replace([float("inf"), float("-inf")], pd.NA)
        rolling_correlation = strategy_returns.rolling(window=window, min_periods=252).corr(benchmark_returns)
        return RollingMetrics(
            window=window,
            series={
                "rolling_sharpe": rolling_sharpe.rename("rolling_sharpe"),
                "rolling_beta": rolling_beta.rename("rolling_beta"),
                "rolling_correlation": rolling_correlation.rename("rolling_correlation"),
            }
        )

    def _build_drawdowns(self, equity: pd.Series) -> DrawdownStats:
        peak = equity.cummax()
        underwater = equity.div(peak).sub(1.0).rename("underwater")
        records: list[dict[str, object]] = []
        in_drawdown = False
        start = trough = peak_date = last_recovered = equity.index[0]
        trough_value = 0.0

        for date, value in underwater.items():
            value = float(value)
            if value >= 0.0:
                last_recovered = date
            if value < 0.0 and not in_drawdown:
                peak_date = last_recovered
                start = trough = date
                trough_value = value
                in_drawdown = True
            elif value < 0.0 and in_drawdown and value <= trough_value:
                trough = date
                trough_value = value
            elif value >= 0.0 and in_drawdown:
                records.append(self._build_drawdown_record(peak_date, start, trough, date, trough_value, recovered=True))
                in_drawdown = False

        if in_drawdown:
            records.append(
                self._build_drawdown_record(
                    peak_date,
                    start,
                    trough,
                    equity.index[-1],
                    trough_value,
                    recovered=False,
                )
            )

        episodes = pd.DataFrame.from_records(
            records,
            columns=[
                "peak",
                "start",
                "trough",
                "end",
                "drawdown",
                "duration_days",
                "time_to_trough_days",
                "recovery_days",
                "recovered",
            ],
        )
        return DrawdownStats(underwater=underwater, episodes=episodes)

    def _build_exposure(self, run: SavedRun) -> ExposureSnapshot:
        holdings_count = run.weights.fillna(0.0).ne(0.0).sum(axis=1).rename("holdings_count")
        latest_holdings = run.latest_weights.copy() if run.latest_weights is not None else self._latest_holdings(run.weights)
        relative_performance = self._latest_holdings_relative_performance(run, latest_holdings)
        display_latest_holdings = self._decorate_holding_symbols(latest_holdings)
        display_winners = self._decorate_holding_symbols(self._rank_latest_holdings(relative_performance, ascending=False))
        display_losers = self._decorate_holding_symbols(self._rank_latest_holdings(relative_performance, ascending=True))
        return ExposureSnapshot(
            holdings_count=holdings_count.astype(float),
            latest_holdings=display_latest_holdings,
            latest_holdings_winners=display_winners,
            latest_holdings_losers=display_losers,
        )

    def _build_sectors(self, weights: pd.DataFrame) -> SectorSnapshot:
        latest_weighted = self.sector_repo.latest_sector_weights(weights)
        latest_weighted = latest_weighted.loc[latest_weighted.ne(0.0)]
        counts = self.sector_repo.latest_sector_counts(weights)
        concentration = latest_weighted.abs().sort_values(ascending=False).rename("concentration")
        return SectorSnapshot(latest_weighted=latest_weighted, latest_count=counts, concentration=concentration)

    def _build_research(
        self,
        run: SavedRun,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        drawdowns: DrawdownStats,
    ) -> ResearchSnapshot:
        sector_weights = self.sector_repo.sector_weight_timeseries(run.weights)
        sector_contribution = self.sector_repo.sector_contribution_timeseries(run.qty, run.equity)
        drawdown_episodes = drawdowns.episodes.sort_values(["drawdown", "start"], ascending=[True, True])
        return ResearchSnapshot(
            monthly_heatmap=build_monthly_heatmap(strategy_returns, run.monthly_returns),
            return_distribution=build_return_distribution(strategy_returns),
            monthly_return_distribution=build_return_distribution(monthly_return_series(strategy_returns, run.monthly_returns)),
            yearly_excess_returns=build_yearly_excess_returns(strategy_returns, benchmark_returns),
            sector_contribution_method=SECTOR_CONTRIBUTION_METHOD_WEIGHTED_ASSET_RETURNS,
            sector_contribution=sector_contribution,
            sector_weights=sector_weights,
            drawdown_episodes=drawdown_episodes,
        )

    def _latest_holdings_relative_performance(self, run: SavedRun, latest_holdings: pd.DataFrame) -> pd.DataFrame:
        if latest_holdings.empty:
            return pd.DataFrame(columns=["symbol", "target_weight", "abs_weight", "return_since_latest_rebalance"])

        if run.path is not None:
            returns_path = run.path / "positions" / "latest_holdings_returns.csv"
            if returns_path.exists():
                returns_frame = pd.read_csv(returns_path)
                return self._merge_latest_holdings_returns(latest_holdings, returns_frame)

        computed_returns = self._compute_latest_holdings_returns(run, latest_holdings)
        if computed_returns.empty:
            return pd.DataFrame(columns=["symbol", "target_weight", "abs_weight", "return_since_latest_rebalance"])
        return self._merge_latest_holdings_returns(latest_holdings, computed_returns)

    def _compute_latest_holdings_returns(self, run: SavedRun, latest_holdings: pd.DataFrame) -> pd.DataFrame:
        prices = self.sector_repo.prices
        if prices is None or latest_holdings.empty:
            return pd.DataFrame(columns=["symbol", "return_since_latest_rebalance"])

        symbols = [str(symbol) for symbol in latest_holdings["symbol"]]
        latest_date = pd.Timestamp(run.equity.index.max())
        rebalance_date = self._latest_rebalance_date(run)
        if rebalance_date is None:
            return pd.DataFrame(columns=["symbol", "return_since_latest_rebalance"])

        price_window = prices.reindex(columns=symbols).sort_index().loc[rebalance_date:latest_date]
        rows: list[dict[str, object]] = []
        for symbol in symbols:
            series = price_window[symbol].dropna()
            if series.empty:
                continue
            starting_price = float(series.iloc[0])
            ending_price = float(series.iloc[-1])
            if starting_price == 0.0:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "return_since_latest_rebalance": ending_price / starting_price - 1.0,
                }
            )
        return pd.DataFrame(rows, columns=["symbol", "return_since_latest_rebalance"])

    @staticmethod
    def _merge_latest_holdings_returns(latest_holdings: pd.DataFrame, returns_frame: pd.DataFrame) -> pd.DataFrame:
        if returns_frame.empty or "symbol" not in returns_frame.columns:
            return pd.DataFrame(columns=["symbol", "target_weight", "abs_weight", "return_since_latest_rebalance"])

        ranked = latest_holdings.copy()
        merged = ranked.merge(
            returns_frame.loc[:, [column for column in ["symbol", "return_since_latest_rebalance"] if column in returns_frame.columns]],
            on="symbol",
            how="inner",
        )
        if merged.empty or "return_since_latest_rebalance" not in merged.columns:
            return pd.DataFrame(columns=["symbol", "target_weight", "abs_weight", "return_since_latest_rebalance"])

        for column in ["target_weight", "abs_weight", "return_since_latest_rebalance"]:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
        merged = merged.replace([float("inf"), float("-inf")], pd.NA).dropna(
            subset=["target_weight", "abs_weight", "return_since_latest_rebalance"]
        )
        return merged.loc[:, ["symbol", "target_weight", "abs_weight", "return_since_latest_rebalance"]].reset_index(
            drop=True
        )

    @staticmethod
    def _rank_latest_holdings(frame: pd.DataFrame, *, ascending: bool) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "target_weight", "abs_weight", "return_since_latest_rebalance"])
        ranked = frame.sort_values(
            ["return_since_latest_rebalance", "abs_weight", "symbol"],
            ascending=[ascending, False, True],
        )
        return ranked.head(5).reset_index(drop=True)

    def _decorate_holding_symbols(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "symbol" not in frame.columns:
            return frame
        decorated = frame.copy()
        decorated["symbol"] = decorated["symbol"].map(lambda symbol: self.sector_repo.display_symbol(str(symbol)))
        return decorated

    @staticmethod
    def _latest_rebalance_date(run: SavedRun) -> pd.Timestamp | None:
        if run.weights.empty:
            return None

        weights = run.weights.fillna(0.0).astype(float).sort_index()
        final_weights = weights.iloc[-1]
        if final_weights.eq(0.0).all():
            return None

        trailing_start = pd.Timestamp(weights.index[-1])
        for index in range(len(weights.index) - 2, -1, -1):
            row = weights.iloc[index]
            if not np.allclose(row.to_numpy(dtype=float), final_weights.to_numpy(dtype=float), rtol=1e-7, atol=1e-9):
                break
            trailing_start = pd.Timestamp(weights.index[index])
        return trailing_start

    @staticmethod
    def _latest_holdings(weights: pd.DataFrame) -> pd.DataFrame:
        latest_weight = weights.iloc[-1].astype(float)
        frame = pd.DataFrame(
            {
                "symbol": latest_weight.index,
                "target_weight": latest_weight.values,
            }
        )
        frame = frame.loc[frame["target_weight"].ne(0.0)].copy()
        frame["abs_weight"] = frame["target_weight"].abs()
        return frame.sort_values(["abs_weight", "symbol"], ascending=[False, True]).reset_index(drop=True)

    @staticmethod
    def _equity_from_returns(returns: pd.Series, starting_value: float) -> pd.Series:
        return (1.0 + returns.fillna(0.0)).cumprod().mul(starting_value).rename("benchmark_equity")

    @staticmethod
    def _cagr(equity: pd.Series, periods: int = 252) -> float:
        if len(equity) < 2:
            return 0.0

        starting = float(equity.iloc[0])
        ending = float(equity.iloc[-1])
        if starting <= 0.0 or ending <= 0.0:
            return 0.0

        years = len(equity) / float(periods)
        if years <= 0.0:
            return 0.0
        return float((ending / starting) ** (1.0 / years) - 1.0)

    @staticmethod
    def _build_drawdown_record(
        peak: pd.Timestamp,
        start: pd.Timestamp,
        trough: pd.Timestamp,
        end: pd.Timestamp,
        drawdown: float,
        *,
        recovered: bool,
    ) -> dict[str, object]:
        peak_ts = pd.Timestamp(peak)
        start_ts = pd.Timestamp(start)
        trough_ts = pd.Timestamp(trough)
        end_ts = pd.Timestamp(end)
        return {
            "peak": peak_ts,
            "start": start_ts,
            "trough": trough_ts,
            "end": end_ts,
            "drawdown": float(drawdown),
            "duration_days": int((end_ts - start_ts).days),
            "time_to_trough_days": int((trough_ts - start_ts).days),
            "recovery_days": int((end_ts - trough_ts).days) if recovered else None,
            "recovered": recovered,
        }
