from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle

from .composable import ComposableStrategy
from .benchmark_overlay import _BenchmarkOverlayConstruction


@dataclass(slots=True)
class BenchmarkTilt(ComposableStrategy):
    lookback: int = 20
    flow_lookback: int = 20
    momentum_lookback: int = 60
    active_share_target: float = 0.20
    max_stock_active: float = 0.015
    max_sector_active: float = 0.05
    min_names: int = 35

    def __post_init__(self) -> None:
        self.signal_producer = _BenchmarkTiltSignal(
            lookback=self.lookback,
            flow_lookback=self.flow_lookback,
            beta_lookback=max(4, self.momentum_lookback * 3),
            momentum_lookback=self.momentum_lookback,
        )
        self.construction_rule = _BenchmarkOverlayConstruction(
            active_share_target=self.active_share_target,
            max_stock_active=self.max_stock_active,
            max_sector_active=self.max_sector_active,
            min_names=self.min_names,
        )


@dataclass(slots=True)
class _BenchmarkTiltSignal:
    lookback: int = 20
    flow_lookback: int = 20
    beta_lookback: int = 180
    momentum_lookback: int = 60

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_BM,
            DatasetId.QW_EPS_NFQ1,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_FOREIGN,
            DatasetId.QW_INSTITUTION,
            DatasetId.QW_RETAIL,
            DatasetId.QW_WICS_SEC_BIG,
            DatasetId.QW_MKTCAP,
            DatasetId.QW_K200_YN,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        k200 = market.frames["k200_yn"].reindex_like(close).fillna(0).astype(bool)
        active_columns = k200.columns[k200.any(axis=0)]
        if len(active_columns) > 0:
            close = close.loc[:, active_columns]
            k200 = k200.loc[:, active_columns]
        benchmark_frame = market.frames["benchmark"]
        eps = market.frames["eps_fwd_q1"].reindex_like(close).ffill()
        op = market.frames["op_fwd_q1"].reindex_like(close).ffill()
        foreign = market.frames["foreign_flow"].reindex_like(close).fillna(0.0)
        inst = market.frames["inst_flow"].reindex_like(close).fillna(0.0)
        retail = market.frames["retail_flow"].reindex_like(close).fillna(0.0)
        sector = market.frames["sector_big"].reindex_like(close).ffill()
        market_cap = market.frames["market_cap"].reindex_like(close).ffill()

        benchmark = benchmark_frame["IKS200"] if "IKS200" in benchmark_frame.columns else benchmark_frame.iloc[:, 0]
        benchmark = benchmark.reindex(close.index).ffill()
        bench_ret = benchmark.pct_change(fill_method=None)
        stock_ret = close.pct_change(fill_method=None)
        min_periods = max(3, min(self.beta_lookback, self.beta_lookback // 3))
        beta = self._rolling_beta(
            stock_ret=stock_ret,
            bench_ret=bench_ret,
            window=self.beta_lookback,
            min_periods=min_periods,
        )
        beta_var = bench_ret.rolling(self.beta_lookback, min_periods=min_periods).var(ddof=0)
        beta = beta.divide(beta_var.replace(0.0, pd.NA), axis=0).replace([float("inf"), float("-inf")], pd.NA)

        eps_revision = eps.pct_change(self.lookback, fill_method=None)
        op_revision = op.pct_change(self.lookback, fill_method=None)
        consensus_revision = 0.5 * eps_revision + 0.5 * op_revision

        net_oi = foreign.add(inst, fill_value=0.0).sub(retail, fill_value=0.0)
        oi_level = net_oi.rolling(self.flow_lookback, min_periods=max(1, self.flow_lookback // 2)).sum()
        oi_impulse = oi_level - oi_level.shift(self.flow_lookback)

        momentum = close.pct_change(self.momentum_lookback, fill_method=None)
        beta_momentum = beta * momentum

        alpha = (
            0.45 * self._cross_sectional_score(consensus_revision)
            + 0.30 * self._cross_sectional_score(oi_level + oi_impulse)
            + 0.25 * self._cross_sectional_score(beta_momentum)
        )
        alpha = alpha.where(k200).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

        base = market_cap.where(k200)
        benchmark_weights = base.div(base.sum(axis=1).replace(0.0, pd.NA), axis=0).fillna(0.0)
        inclusion = k200 & benchmark_weights.gt(0.0)
        overlay_scale = self._overlay_scale(alpha=alpha, membership=inclusion)

        return SignalBundle(
            alpha=alpha,
            context={
                "sector": sector,
                "benchmark_weights": benchmark_weights,
                "benchmark_membership": inclusion,
                "overlay_scale": overlay_scale,
                "inclusion": inclusion,
            },
            meta={},
        )

    @staticmethod
    def _cross_sectional_score(frame: pd.DataFrame) -> pd.DataFrame:
        ranks = frame.rank(axis=1, pct=True)
        return ranks.sub(0.5).fillna(0.0)

    @staticmethod
    def _rolling_beta(*, stock_ret: pd.DataFrame, bench_ret: pd.Series, window: int, min_periods: int) -> pd.DataFrame:
        stock_mean = stock_ret.rolling(window, min_periods=min_periods).mean()
        bench_mean = bench_ret.rolling(window, min_periods=min_periods).mean()
        cross_mean = stock_ret.mul(bench_ret, axis=0).rolling(window, min_periods=min_periods).mean()
        return cross_mean - stock_mean.mul(bench_mean, axis=0)

    @staticmethod
    def _overlay_scale(*, alpha: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
        positive_breadth = alpha.where(membership).gt(0.0).sum(axis=1)
        member_count = membership.sum(axis=1).replace(0, pd.NA)
        breadth = positive_breadth.divide(member_count).fillna(0.0)
        return breadth.clip(lower=0.25, upper=1.0).astype(float)


