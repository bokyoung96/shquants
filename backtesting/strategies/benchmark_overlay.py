from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.data.benchmarks import benchmark_price_series
from backtesting.signals.base import SignalBundle

from .composable import ComposableStrategy


@dataclass(slots=True)
class BenchmarkOverlay(ComposableStrategy):
    lookback: int = 60
    flow_lookback: int = 20
    momentum_lookback: int = 120
    top_n: int = 18
    active_share_target: float = 0.20
    max_stock_active: float = 0.015
    max_sector_active: float = 0.05
    min_names: int = 25

    def __post_init__(self) -> None:
        self.signal_producer = _BenchmarkOverlaySignal(
            lookback=self.lookback,
            flow_lookback=self.flow_lookback,
            beta_lookback=max(120, self.momentum_lookback),
        )
        self.construction_rule = _BenchmarkOverlayConstruction(
            active_share_target=self.active_share_target,
            max_stock_active=self.max_stock_active,
            max_sector_active=self.max_sector_active,
            min_names=max(self.min_names, self.top_n),
        )


@dataclass(slots=True)
class _BenchmarkOverlaySignal:
    lookback: int = 60
    flow_lookback: int = 20
    beta_lookback: int = 120

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean) < 2:
            return pd.Series(0.0, index=series.index, dtype=float)
        std = float(clean.std())
        if std <= 0.0 or pd.isna(std):
            return pd.Series(0.0, index=series.index, dtype=float)
        return ((pd.to_numeric(series, errors="coerce") - float(clean.mean())) / std).fillna(0.0).astype(float)

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_BM,
            DatasetId.QW_EPS_NFQ1,
            DatasetId.QW_OP_NFQ1,
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
        eps = market.frames["eps_fwd_q1"].ffill()
        op = market.frames["op_fwd_q1"].ffill()
        sector = market.frames["sector_big"].ffill()
        market_cap = market.frames["market_cap"].reindex_like(close).ffill()

        benchmark = benchmark_price_series(benchmark_frame, "IKS200")
        benchmark = benchmark.reindex(close.index).ffill()
        bench_ret = benchmark.pct_change(fill_method=None)
        stock_ret = close.pct_change(fill_method=None)

        min_periods = max(40, self.beta_lookback // 3)
        beta_cov = self._rolling_covariance(
            stock_ret=stock_ret,
            bench_ret=bench_ret,
            window=self.beta_lookback,
            min_periods=min_periods,
        )
        beta_var = bench_ret.rolling(self.beta_lookback, min_periods=min_periods).var(ddof=0)
        beta = beta_cov.divide(beta_var.replace(0.0, pd.NA), axis=0)

        eps_trend = eps.divide(eps.rolling(self.lookback, min_periods=max(20, self.lookback // 3)).mean()) - 1.0
        op_trend = op.divide(op.rolling(self.lookback, min_periods=max(20, self.lookback // 3)).mean()) - 1.0
        eps_delta = eps_trend - eps_trend.shift(self.flow_lookback)
        op_delta = op_trend - op_trend.shift(self.flow_lookback)

        consensus_level = (eps_trend + op_trend) / 2.0
        consensus_flow = (eps_delta + op_delta) / 2.0
        soft_score = consensus_level + 0.6 * consensus_flow
        market_breadth = (soft_score.gt(0.0).mean(axis=1) - soft_score.lt(0.0).mean(axis=1)).fillna(0.0)
        breadth_mean = market_breadth.rolling(self.lookback, min_periods=max(20, self.lookback // 3)).mean().fillna(0.0)
        breadth_delta = market_breadth.diff(self.flow_lookback).fillna(0.0)

        denom = market_cap.where(k200)
        benchmark_weights = denom.div(denom.sum(axis=1).replace(0.0, pd.NA), axis=0).fillna(0.0)

        raw_score = (
            soft_score.reindex(index=close.index, columns=close.columns).fillna(0.0)
            + 0.20 * self._cross_sectional_zscore(beta.reindex(index=close.index, columns=close.columns), k200)
            + 0.15
            * self._cross_sectional_zscore(consensus_level.reindex(index=close.index, columns=close.columns), k200)
            + 0.10 * self._cross_sectional_zscore(consensus_flow.reindex(index=close.index, columns=close.columns), k200)
        )
        raw_score = raw_score.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        weighted_center = raw_score.mul(benchmark_weights, axis=0).sum(axis=1)
        score = raw_score.sub(weighted_center, axis=0).where(k200, 0.0).fillna(0.0)

        active_ranks = score.abs().where(k200).rank(axis=1, ascending=False, method="first", na_option="bottom")
        active_mask = active_ranks.le(60) & k200
        alpha = score.where(active_mask, 0.0).astype(float)
        inclusion = k200.copy()

        strong_breadth = market_breadth.gt(pd.concat((breadth_mean, pd.Series(0.03, index=breadth_mean.index)), axis=1).max(axis=1))
        overlay_scale = pd.Series(0.35, index=close.index, dtype=float)
        overlay_scale.loc[market_breadth.gt(-0.02)] = 0.70
        overlay_scale.loc[strong_breadth & breadth_delta.gt(0.0)] = 1.00
        overlay_scale = overlay_scale.where(score.abs().sum(axis=1).gt(0.0), 0.0)

        return SignalBundle(
            alpha=alpha,
            context={
                "sector": sector,
                "benchmark_weights": benchmark_weights,
                "benchmark_membership": k200,
                "overlay_scale": overlay_scale,
                "inclusion": inclusion,
            },
            meta={},
        )

    @staticmethod
    def _rolling_covariance(
        *,
        stock_ret: pd.DataFrame,
        bench_ret: pd.Series,
        window: int,
        min_periods: int,
    ) -> pd.DataFrame:
        stock_mean = stock_ret.rolling(window, min_periods=min_periods).mean()
        bench_mean = bench_ret.rolling(window, min_periods=min_periods).mean()
        cross_mean = stock_ret.mul(bench_ret, axis=0).rolling(window, min_periods=min_periods).mean()
        return cross_mean - stock_mean.mul(bench_mean, axis=0)

    @staticmethod
    def _cross_sectional_zscore(frame: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
        frame = frame.reindex(index=membership.index, columns=membership.columns)
        membership = membership.fillna(False).astype(bool)
        values = frame.where(membership).fillna(0.0).astype(float)
        counts = membership.sum(axis=1).astype(float)
        sums = values.sum(axis=1)
        means = sums.divide(counts.where(counts.ne(0), np.nan)).fillna(0.0)
        centered = values.sub(means, axis=0).where(membership, 0.0)
        variance = centered.pow(2).sum(axis=1).divide((counts - 1.0).where(counts.gt(1), np.nan))
        std = variance.pow(0.5)
        zscore = centered.div(std.replace(0.0, np.nan), axis=0)
        return zscore.where(membership, 0.0).fillna(0.0).astype(float)


@dataclass(slots=True)
class _BenchmarkOverlayConstruction:
    active_share_target: float = 0.20
    max_stock_active: float = 0.015
    max_sector_active: float = 0.05
    min_names: int = 25

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        alpha = bundle.alpha
        sector = bundle.context.get("sector")
        benchmark_weights = bundle.context.get("benchmark_weights")
        benchmark_membership = bundle.context.get("benchmark_membership")
        overlay_scale = bundle.context.get("overlay_scale")
        inclusion = bundle.context.get("inclusion")
        if not isinstance(sector, pd.DataFrame):
            raise ValueError("benchmark overlay construction requires sector context")
        if not isinstance(benchmark_weights, pd.DataFrame):
            raise ValueError("benchmark overlay construction requires benchmark_weights context")
        if not isinstance(benchmark_membership, pd.DataFrame):
            raise ValueError("benchmark overlay construction requires benchmark_membership context")
        if not isinstance(overlay_scale, pd.Series):
            raise ValueError("benchmark overlay construction requires overlay_scale context")
        if not isinstance(inclusion, pd.DataFrame):
            raise ValueError("benchmark overlay construction requires inclusion context")

        sector = sector.reindex(index=alpha.index, columns=alpha.columns).fillna("unknown").astype(str)
        benchmark_weights = benchmark_weights.reindex(index=alpha.index, columns=alpha.columns).fillna(0.0).astype(float)
        benchmark_membership = benchmark_membership.reindex(index=alpha.index, columns=alpha.columns).fillna(False).astype(bool)
        inclusion = inclusion.reindex(index=alpha.index, columns=alpha.columns).fillna(False).astype(bool)
        overlay_scale = overlay_scale.reindex(alpha.index).fillna(0.0).astype(float)

        alpha_values = alpha.reindex(index=alpha.index, columns=alpha.columns).fillna(0.0).to_numpy(dtype=float)
        benchmark_values = benchmark_weights.to_numpy(dtype=float)
        membership_values = benchmark_membership.to_numpy(dtype=bool)
        inclusion_values = inclusion.to_numpy(dtype=bool)
        sector_values = sector.to_numpy(dtype=object)
        scale_values = overlay_scale.to_numpy(dtype=float)

        weight_values = np.zeros_like(alpha_values, dtype=float)
        for row_index in range(len(alpha.index)):
            base = np.where(membership_values[row_index], benchmark_values[row_index], 0.0)
            base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0).astype(float, copy=False)
            base_sum = float(base.sum())
            if base_sum <= 0.0:
                continue
            base = base / base_sum

            signal = np.where(inclusion_values[row_index], alpha_values[row_index], 0.0)
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(float, copy=False)
            active = self._build_active_overlay_values(
                signal=signal,
                base=base,
                sector=sector_values[row_index],
                scale=float(scale_values[row_index]),
            )
            weights = np.clip(base + active, 0.0, None)
            total = float(weights.sum())
            if total > 0.0:
                weights = weights / total
            weight_values[row_index] = weights

        base_target_weights = pd.DataFrame(weight_values, index=alpha.index, columns=alpha.columns, dtype=float)
        selection_mask = pd.DataFrame(weight_values > 0.0, index=alpha.index, columns=alpha.columns, dtype=bool)
        return ConstructionResult(
            base_target_weights=base_target_weights,
            selection_mask=selection_mask,
            group_long_budget=None,
            group_short_budget=None,
            meta={},
        )

    def _build_active_overlay(
        self,
        *,
        signal: pd.Series,
        base: pd.Series,
        sector_row: pd.Series,
        scale: float,
    ) -> pd.Series:
        zeros = pd.Series(0.0, index=signal.index, dtype=float)
        ranked = signal.abs().sort_values(ascending=False)
        if ranked.empty:
            return zeros

        keep = ranked.head(max(self.min_names, 1)).index
        raw = signal.reindex(keep).fillna(0.0)
        raw = raw - float((raw * base.reindex(keep).fillna(0.0)).sum())
        if raw.abs().sum() <= 0.0:
            return zeros

        active = pd.Series(0.0, index=signal.index, dtype=float)
        gross_budget = max(self.active_share_target * scale, 0.0)
        if gross_budget <= 0.0:
            return active

        pos = raw.clip(lower=0.0)
        neg = (-raw.clip(upper=0.0))
        pos_sum = float(pos.sum())
        neg_sum = float(neg.sum())
        if pos_sum <= 0.0 or neg_sum <= 0.0:
            return active

        pos_budget = gross_budget / 2.0
        neg_budget = gross_budget / 2.0
        active.loc[pos.index] += pos_budget * (pos / pos_sum)
        active.loc[neg.index] -= neg_budget * (neg / neg_sum)

        active = active.clip(lower=-self.max_stock_active, upper=self.max_stock_active)
        active = self._recenter(active, base)
        active = self._cap_sector(active, sector_row)
        active = self._recenter(active, base)

        floor = -base
        active = active.clip(lower=floor, upper=self.max_stock_active)
        active = self._recenter(active, base)
        return active.fillna(0.0)

    def _build_active_overlay_values(
        self,
        *,
        signal: np.ndarray,
        base: np.ndarray,
        sector: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        active = np.zeros(signal.shape, dtype=float)
        if signal.size == 0:
            return active

        ranked = np.argsort(-np.abs(signal), kind="stable")
        keep = ranked[: max(self.min_names, 1)]
        raw = signal[keep].astype(float, copy=True)
        raw = raw - float((raw * base[keep]).sum())
        if float(np.abs(raw).sum()) <= 0.0:
            return active

        gross_budget = max(self.active_share_target * scale, 0.0)
        if gross_budget <= 0.0:
            return active

        pos = np.clip(raw, 0.0, None)
        neg = -np.clip(raw, None, 0.0)
        pos_sum = float(pos.sum())
        neg_sum = float(neg.sum())
        if pos_sum <= 0.0 or neg_sum <= 0.0:
            return active

        active[keep] += (gross_budget / 2.0) * (pos / pos_sum)
        active[keep] -= (gross_budget / 2.0) * (neg / neg_sum)

        active = np.clip(active, -self.max_stock_active, self.max_stock_active)
        active = self._recenter_values(active, base)
        active = self._cap_sector_values(active, sector)
        active = self._recenter_values(active, base)
        active = np.minimum(np.maximum(active, -base), self.max_stock_active)
        active = self._recenter_values(active, base)
        return np.nan_to_num(active, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _recenter(active: pd.Series, base: pd.Series) -> pd.Series:
        total = float(active.sum())
        if abs(total) < 1e-12:
            return active
        adjust = base.fillna(0.0)
        denom = float(adjust.sum())
        if denom <= 0.0:
            return active
        return active - total * (adjust / denom)

    def _cap_sector(self, active: pd.Series, sector_row: pd.Series) -> pd.Series:
        adjusted = active.copy()
        for sector_name, members in sector_row.groupby(sector_row, sort=False):
            names = members.index
            sector_active = float(adjusted.reindex(names).sum())
            if sector_active > self.max_sector_active:
                scale = self.max_sector_active / sector_active
                adjusted.loc[names] = adjusted.reindex(names).fillna(0.0) * scale
            elif sector_active < -self.max_sector_active:
                scale = self.max_sector_active / abs(sector_active)
                adjusted.loc[names] = adjusted.reindex(names).fillna(0.0) * scale
        return adjusted

    @staticmethod
    def _recenter_values(active: np.ndarray, base: np.ndarray) -> np.ndarray:
        total = float(active.sum())
        if abs(total) < 1e-12:
            return active
        denom = float(base.sum())
        if denom <= 0.0:
            return active
        return active - total * (base / denom)

    def _cap_sector_values(self, active: np.ndarray, sector: np.ndarray) -> np.ndarray:
        adjusted = active.copy()
        for sector_name in pd.unique(sector):
            mask = sector == sector_name
            sector_active = float(adjusted[mask].sum())
            if sector_active > self.max_sector_active:
                adjusted[mask] = adjusted[mask] * (self.max_sector_active / sector_active)
            elif sector_active < -self.max_sector_active:
                adjusted[mask] = adjusted[mask] * (self.max_sector_active / abs(sector_active))
        return adjusted


