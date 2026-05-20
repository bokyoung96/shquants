from dataclasses import dataclass

import numpy as np
import pandas as pd


INVESTABLE_ASSETS: tuple[str, str] = ("SPY US Equity", "IEF US Equity")


@dataclass(frozen=True, slots=True)
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    drawdown: pd.Series
    gross_returns: pd.Series
    gross_equity: pd.Series
    gross_drawdown: pd.Series
    benchmark_returns: pd.Series
    benchmark_equity: pd.Series
    benchmark_drawdown: pd.Series
    benchmark_weights: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    benchmark_costs: pd.Series
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class TwoAssetBacktester:
    periods_per_year: int = 252
    benchmark_equity_weight: float = 0.75
    transaction_cost_bps: float = 5.0

    def run(self, weights: pd.DataFrame, daily_returns: pd.DataFrame) -> BacktestResult:
        missing = [
            asset
            for asset in INVESTABLE_ASSETS
            if asset not in weights.columns or asset not in daily_returns.columns
        ]
        if missing:
            raise ValueError(f"missing required backtest columns: {', '.join(missing)}")

        aligned_weights = weights.loc[:, INVESTABLE_ASSETS].sort_index().astype(float)
        aligned_returns = daily_returns.loc[:, INVESTABLE_ASSETS].sort_index().astype(float)
        common = aligned_weights.index.intersection(aligned_returns.index)
        aligned_weights = aligned_weights.loc[common]
        aligned_returns = aligned_returns.loc[common]

        next_day_returns = aligned_returns.shift(-1)
        turnover = self._turnover(aligned_weights)
        costs = (turnover * (float(self.transaction_cost_bps) / 10_000.0)).rename("transaction_cost")
        gross_returns = (aligned_weights * next_day_returns).sum(axis=1, min_count=1).iloc[:-1]
        gross_returns.name = "gross_portfolio_return"
        gross_equity = (1.0 + gross_returns.fillna(0.0)).cumprod().rename("gross_equity")
        gross_drawdown = (gross_equity / gross_equity.cummax() - 1.0).rename("gross_drawdown")
        portfolio_returns = gross_returns.sub(costs).iloc[:-1]
        portfolio_returns.name = "portfolio_return"
        equity = (1.0 + portfolio_returns.fillna(0.0)).cumprod().rename("equity")
        drawdown = (equity / equity.cummax() - 1.0).rename("drawdown")
        benchmark_weights = pd.Series(
            {
                "SPY US Equity": float(self.benchmark_equity_weight),
                "IEF US Equity": 1.0 - float(self.benchmark_equity_weight),
            },
            name="benchmark_weight",
        )
        benchmark_turnover = pd.Series(0.0, index=aligned_weights.index, name="benchmark_turnover")
        if len(benchmark_turnover):
            benchmark_turnover.iloc[0] = float(benchmark_weights.abs().sum())
        benchmark_costs = (benchmark_turnover * (float(self.transaction_cost_bps) / 10_000.0)).rename(
            "benchmark_transaction_cost"
        )
        benchmark_returns = (next_day_returns * benchmark_weights).sum(axis=1, min_count=1).sub(benchmark_costs).iloc[:-1]
        benchmark_returns.name = "benchmark_return"
        benchmark_equity = (1.0 + benchmark_returns.fillna(0.0)).cumprod().rename("benchmark_equity")
        benchmark_drawdown = (benchmark_equity / benchmark_equity.cummax() - 1.0).rename("benchmark_drawdown")

        metrics = self._metrics(portfolio_returns, equity, turnover)
        gross_metrics = self._metrics(gross_returns, gross_equity, turnover)
        metrics.update({f"gross_{key}": value for key, value in gross_metrics.items()})
        metrics["transaction_cost_bps"] = float(self.transaction_cost_bps)
        metrics["total_transaction_cost"] = float(costs.loc[portfolio_returns.index].sum())
        benchmark_metrics = self._metrics(benchmark_returns, benchmark_equity, benchmark_turnover)
        benchmark_metrics["transaction_cost_bps"] = float(self.transaction_cost_bps)
        benchmark_metrics["total_transaction_cost"] = float(benchmark_costs.loc[benchmark_returns.index].sum())
        metrics.update({f"benchmark_{key}": value for key, value in benchmark_metrics.items()})
        metrics["active_total_return"] = metrics["total_return"] - metrics["benchmark_total_return"]
        metrics["active_annualized_return"] = metrics["annualized_return"] - metrics["benchmark_annualized_return"]
        metrics["active_sharpe"] = metrics["sharpe"] - metrics["benchmark_sharpe"]

        return BacktestResult(
            portfolio_returns,
            equity,
            drawdown,
            gross_returns,
            gross_equity,
            gross_drawdown,
            benchmark_returns,
            benchmark_equity,
            benchmark_drawdown,
            benchmark_weights,
            aligned_weights,
            turnover,
            costs,
            benchmark_costs,
            metrics,
        )

    @staticmethod
    def _turnover(weights: pd.DataFrame) -> pd.Series:
        previous = weights.shift(1).fillna(0.0)
        return weights.sub(previous).abs().sum(axis=1).rename("turnover")

    def _metrics(self, returns: pd.Series, equity: pd.Series, turnover: pd.Series) -> dict[str, float]:
        observations = len(returns)
        annualized_volatility = float(returns.std(ddof=0) * np.sqrt(self.periods_per_year)) if observations else 0.0
        return {
            "observations": float(observations),
            "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
            "annualized_return": self._annualized_return(equity),
            "annualized_volatility": annualized_volatility,
            "sharpe": self._sharpe(returns),
            "max_drawdown": float(equity.div(equity.cummax()).sub(1.0).min()) if len(equity) else 0.0,
            "turnover": float(turnover.mean()) if len(turnover) else 0.0,
        }

    def _annualized_return(self, equity: pd.Series) -> float:
        if len(equity) == 0:
            return 0.0
        years = len(equity) / float(self.periods_per_year)
        if years <= 0.0 or equity.iloc[-1] <= 0.0:
            return 0.0
        return float(equity.iloc[-1] ** (1.0 / years) - 1.0)

    def _sharpe(self, returns: pd.Series) -> float:
        if len(returns) == 0:
            return 0.0
        std = float(returns.std(ddof=0))
        if abs(std) < 1e-12:
            return 0.0
        return float(returns.mean() / std * np.sqrt(self.periods_per_year))
