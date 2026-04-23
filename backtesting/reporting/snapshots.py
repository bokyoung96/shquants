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
    annualized_downside_deviation,
    annualized_sharpe,
    annualized_volatility,
    build_monthly_heatmap,
    build_return_distribution,
    build_yearly_excess_returns,
    capture_ratio,
    conditional_value_at_risk,
    hit_ratio,
    kurtosis,
    max_duration,
    monthly_return_series,
    payoff_ratio,
    profit_factor,
    rolling_downside_deviation,
    rolling_return,
    rolling_volatility,
    skewness,
    value_at_risk,
    win_rate,
    yearly_return_series,
)
from .benchmarks import BenchmarkRepository, SectorRepository
from .models import BenchmarkConfig, ReportProfile, SavedRun


def _default_benchmark() -> BenchmarkConfig:
    return BenchmarkConfig.default_kospi200()


def _empty_series(name: str) -> pd.Series:
    return pd.Series(dtype=float, name=name)


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
    benchmark: BenchmarkConfig | None = field(default_factory=_default_benchmark)
    profile: ReportProfile = ReportProfile.ALPHA
    has_benchmark: bool = True
    research: ResearchSnapshot = field(default_factory=ResearchSnapshot)


class PerformanceSnapshotFactory:
    def __init__(self, benchmark_repo: BenchmarkRepository, sector_repo: SectorRepository) -> None:
        self.benchmark_repo = benchmark_repo
        self.sector_repo = sector_repo

    def build(
        self,
        run: SavedRun,
        benchmark: BenchmarkConfig | None,
        profile: ReportProfile | str | None = None,
    ) -> PerformanceSnapshot:
        strategy_returns = run.returns.astype(float).sort_index()
        strategy_equity = run.equity.astype(float).sort_index()
        benchmark_config, benchmark_returns, benchmark_equity = self._load_benchmark(strategy_returns, strategy_equity, benchmark)
        has_benchmark = not benchmark_returns.empty
        resolved_profile = self._resolve_profile(run, benchmark_config, profile)
        if not has_benchmark:
            resolved_profile = ReportProfile.ABSOLUTE

        rolling = self._build_rolling_metrics(strategy_returns, benchmark_returns, has_benchmark)
        drawdowns = self._build_drawdowns(strategy_equity)
        exposure = self._build_exposure(run)
        sectors = self._build_sectors(run.weights)
        research = self._build_research(run, strategy_returns, benchmark_returns, drawdowns, has_benchmark)
        metrics = self._build_metrics(
            run,
            strategy_returns,
            strategy_equity,
            benchmark_returns,
            drawdowns.underwater,
            drawdowns.episodes,
            has_benchmark,
        )

        return PerformanceSnapshot(
            run_id=run.run_id,
            display_name=str(run.config.get("name") or run.run_id),
            strategy_name=str(run.config.get("strategy") or "unknown"),
            benchmark=benchmark_config,
            profile=resolved_profile,
            has_benchmark=has_benchmark,
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

    def _load_benchmark(
        self,
        strategy_returns: pd.Series,
        strategy_equity: pd.Series,
        benchmark: BenchmarkConfig | None,
    ) -> tuple[BenchmarkConfig | None, pd.Series, pd.Series]:
        if benchmark is None or strategy_returns.empty:
            return None, _empty_series("benchmark_returns"), _empty_series("benchmark_equity")

        try:
            benchmark_series = self.benchmark_repo.load_series(
                benchmark,
                start=str(strategy_returns.index.min().date()),
                end=str(strategy_returns.index.max().date()),
            )
        except Exception:
            return None, _empty_series("benchmark_returns"), _empty_series("benchmark_equity")

        benchmark_returns = benchmark_series.returns.reindex(strategy_returns.index).fillna(0.0).astype(float)
        benchmark_equity = self._equity_from_returns(
            benchmark_returns,
            starting_value=float(strategy_equity.iloc[0]),
        )
        return benchmark, benchmark_returns.rename("benchmark_returns"), benchmark_equity.rename("benchmark_equity")

    def _resolve_profile(
        self,
        run: SavedRun,
        benchmark: BenchmarkConfig | None,
        profile: ReportProfile | str | None,
    ) -> ReportProfile:
        explicit = ReportProfile.normalize(profile)
        if explicit is None:
            explicit = ReportProfile.normalize(
                run.config.get("report_profile") or run.config.get("profile")  # type: ignore[arg-type]
            )
        if explicit is not None:
            return explicit
        if benchmark is None:
            return ReportProfile.ABSOLUTE
        strategy_tokens = " ".join(
            str(run.config.get(key, "")) for key in ("strategy", "name", "benchmark_code", "benchmark_name")
        ).lower()
        if any(token in strategy_tokens for token in ("index", "tracker", "tracking", "benchmark", " bm", "etf")):
            return ReportProfile.INDEX
        return ReportProfile.ALPHA

    def _build_metrics(
        self,
        run: SavedRun,
        strategy_returns: pd.Series,
        strategy_equity: pd.Series,
        benchmark_returns: pd.Series,
        underwater: pd.Series,
        drawdown_episodes: pd.DataFrame,
        has_benchmark: bool,
    ) -> PerformanceMetrics:
        monthly_returns = monthly_return_series(strategy_returns, run.monthly_returns)
        yearly_returns = yearly_return_series(strategy_returns)
        annual_volatility = annualized_volatility(strategy_returns)
        downside_deviation = annualized_downside_deviation(strategy_returns)
        cumulative_return = float(strategy_equity.iloc[-1] / strategy_equity.iloc[0] - 1.0)
        cagr = self._cagr(strategy_equity)
        max_drawdown = float(underwater.min()) if not underwater.empty else 0.0
        sortino = 0.0 if abs(downside_deviation) < 1e-12 else float(strategy_returns.mean() * 252.0 / downside_deviation)
        calmar = 0.0 if max_drawdown >= 0.0 else float(cagr / abs(max_drawdown))
        clean_returns = strategy_returns.dropna().astype(float)

        alpha: float | None = None
        beta: float | None = None
        tracking_error: float | None = None
        information_ratio: float | None = None
        correlation: float | None = None
        upside_capture: float | None = None
        downside_capture: float | None = None
        active_return: float | None = None
        active_risk: float | None = None

        if has_benchmark:
            active_returns = strategy_returns.sub(benchmark_returns, fill_value=0.0)
            benchmark_variance = float(benchmark_returns.var(ddof=0))
            covariance = float(strategy_returns.cov(benchmark_returns, ddof=0))
            beta = 0.0 if abs(benchmark_variance) < 1e-12 else covariance / benchmark_variance
            alpha = float((strategy_returns.mean() - float(beta) * benchmark_returns.mean()) * 252.0)
            tracking_error = annualized_volatility(active_returns)
            information_ratio = (
                None
                if abs(tracking_error) < 1e-12
                else float(active_returns.mean() * 252.0 / tracking_error)
            )
            correlation_value = float(strategy_returns.corr(benchmark_returns)) if len(strategy_returns) > 1 else np.nan
            correlation = None if pd.isna(correlation_value) else correlation_value
            upside_capture = capture_ratio(strategy_returns, benchmark_returns, upside=True)
            downside_capture = capture_ratio(strategy_returns, benchmark_returns, upside=False)
            active_return = float(active_returns.mean() * 252.0)
            active_risk = tracking_error

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
            beta=beta,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            downside_deviation=downside_deviation,
            value_at_risk_95=value_at_risk(strategy_returns),
            conditional_value_at_risk_95=conditional_value_at_risk(strategy_returns),
            win_rate=win_rate(strategy_returns),
            payoff_ratio=payoff_ratio(strategy_returns),
            profit_factor=profit_factor(strategy_returns),
            skew=skewness(strategy_returns),
            kurtosis=kurtosis(strategy_returns),
            best_day=0.0 if clean_returns.empty else float(clean_returns.max()),
            worst_day=0.0 if clean_returns.empty else float(clean_returns.min()),
            best_month=0.0 if monthly_returns.empty else float(monthly_returns.max()),
            worst_month=0.0 if monthly_returns.empty else float(monthly_returns.min()),
            best_year=0.0 if yearly_returns.empty else float(yearly_returns.max()),
            worst_year=0.0 if yearly_returns.empty else float(yearly_returns.min()),
            longest_drawdown_days=max_duration(drawdown_episodes, "duration_days"),
            recovery_days=max_duration(drawdown_episodes, "recovery_days"),
            current_drawdown=0.0 if underwater.empty else float(underwater.iloc[-1]),
            month_hit_ratio=hit_ratio(monthly_returns),
            year_hit_ratio=hit_ratio(yearly_returns),
            correlation=correlation,
            upside_capture=upside_capture,
            downside_capture=downside_capture,
            active_return=active_return,
            active_risk=active_risk,
        )

    def _build_rolling_metrics(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        has_benchmark: bool,
    ) -> RollingMetrics:
        window = ROLLING_WINDOW
        series: dict[str, pd.Series] = {
            "rolling_sharpe": strategy_returns.rolling(window=window, min_periods=window).apply(
                lambda values: annualized_sharpe(pd.Series(values)),
                raw=False,
            ).rename("rolling_sharpe"),
            "rolling_volatility": rolling_volatility(strategy_returns, window=window),
            "rolling_return": rolling_return(strategy_returns, window=window),
            "rolling_downside_deviation": rolling_downside_deviation(strategy_returns, window=window),
        }
        if has_benchmark:
            benchmark_variance = benchmark_returns.rolling(window=window, min_periods=window).var(ddof=0)
            rolling_beta = strategy_returns.rolling(window=window, min_periods=window).cov(
                benchmark_returns, ddof=0
            ).div(benchmark_variance)
            rolling_beta = rolling_beta.replace([float("inf"), float("-inf")], pd.NA)
            rolling_correlation = strategy_returns.rolling(window=window, min_periods=window).corr(benchmark_returns)
            active_returns = strategy_returns.sub(benchmark_returns, fill_value=0.0)
            rolling_tracking_error = active_returns.rolling(window=window, min_periods=window).std(ddof=0).mul(
                252.0**0.5
            )
            rolling_information_ratio = active_returns.rolling(window=window, min_periods=window).mean().mul(252.0).div(
                rolling_tracking_error.replace(0.0, pd.NA)
            )
            rolling_alpha = (
                strategy_returns.rolling(window=window, min_periods=window).mean()
                - rolling_beta * benchmark_returns.rolling(window=window, min_periods=window).mean()
            ).mul(252.0)
            series.update(
                {
                    "rolling_beta": rolling_beta.rename("rolling_beta"),
                    "rolling_correlation": rolling_correlation.rename("rolling_correlation"),
                    "rolling_tracking_error": rolling_tracking_error.rename("rolling_tracking_error"),
                    "rolling_information_ratio": rolling_information_ratio.rename("rolling_information_ratio"),
                    "rolling_alpha": rolling_alpha.rename("rolling_alpha"),
                }
            )
        return RollingMetrics(window=window, series=series)

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
            turnover=run.turnover.fillna(0.0).astype(float).rename("turnover"),
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
        has_benchmark: bool,
    ) -> ResearchSnapshot:
        sector_weights = self.sector_repo.sector_weight_timeseries(run.weights)
        sector_contribution = self.sector_repo.sector_contribution_timeseries(run.qty, run.equity)
        drawdown_episodes = drawdowns.episodes.sort_values(["drawdown", "start"], ascending=[True, True])
        return ResearchSnapshot(
            monthly_heatmap=build_monthly_heatmap(strategy_returns, run.monthly_returns),
            return_distribution=build_return_distribution(strategy_returns),
            monthly_return_distribution=build_return_distribution(monthly_return_series(strategy_returns, run.monthly_returns)),
            yearly_returns=yearly_return_series(strategy_returns),
            yearly_excess_returns=(
                build_yearly_excess_returns(strategy_returns, benchmark_returns)
                if has_benchmark
                else pd.Series(dtype=float, name="yearly_excess_return")
            ),
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
