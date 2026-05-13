from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle

from .composable import ComposableStrategy


@dataclass(slots=True)
class ConsensusBetaSoftParticipationLongOnly(ComposableStrategy):
    lookback: int = 60
    flow_lookback: int = 20
    momentum_lookback: int = 120
    top_n: int = 14

    def __post_init__(self) -> None:
        self.signal_producer = _ConsensusBetaSoftParticipationSignalProducer(
            lookback=self.lookback,
            flow_lookback=self.flow_lookback,
            beta_lookback=max(120, self.momentum_lookback),
            top_n=self.top_n,
        )
        self.construction_rule = _SoftParticipationLongOnlyConstruction(top_n=self.top_n)


@dataclass(slots=True)
class _ConsensusBetaSoftParticipationSignalProducer:
    lookback: int = 60
    flow_lookback: int = 20
    beta_lookback: int = 120
    top_n: int = 14

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
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        benchmark_frame = market.frames["benchmark"]
        eps = market.frames["eps_fwd_q1"].ffill()
        op = market.frames["op_fwd_q1"].ffill()
        sector = market.frames["sector_big"].ffill()

        benchmark = benchmark_frame["IKS200"] if "IKS200" in benchmark_frame.columns else benchmark_frame.iloc[:, 0]
        benchmark = benchmark.reindex(close.index).ffill()
        bench_ret = benchmark.pct_change(fill_method=None)
        stock_ret = close.pct_change(fill_method=None)

        beta = stock_ret.rolling(self.beta_lookback, min_periods=max(40, self.beta_lookback // 3)).cov(bench_ret)
        beta = beta.divide(bench_ret.rolling(self.beta_lookback, min_periods=max(40, self.beta_lookback // 3)).var(), axis=0)
        benchmark_trend = benchmark.divide(benchmark.rolling(self.lookback, min_periods=max(20, self.lookback // 3)).mean()) - 1.0

        eps_trend = eps.divide(eps.rolling(self.lookback, min_periods=max(20, self.lookback // 3)).mean()) - 1.0
        op_trend = op.divide(op.rolling(self.lookback, min_periods=max(20, self.lookback // 3)).mean()) - 1.0
        eps_delta = eps_trend - eps_trend.shift(self.flow_lookback)
        op_delta = op_trend - op_trend.shift(self.flow_lookback)

        consensus_level = (eps_trend + op_trend) / 2.0
        consensus_flow = (eps_delta + op_delta) / 2.0
        soft_score = consensus_level + 0.6 * consensus_flow
        soft_positive = soft_score.gt(0.0)
        soft_negative = soft_score.lt(0.0)

        market_breadth = (soft_positive.mean(axis=1) - soft_negative.mean(axis=1)).fillna(0.0)
        breadth_mean = market_breadth.rolling(self.lookback, min_periods=max(20, self.lookback // 3)).mean().fillna(0.0)
        breadth_delta = market_breadth.diff(self.flow_lookback).fillna(0.0)

        gross_long = pd.Series(0.35, index=close.index, dtype=float)
        alpha = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        selected = pd.DataFrame(False, index=close.index, columns=close.columns)

        for timestamp in close.index:
            beta_row = beta.loc[timestamp].dropna()
            if beta_row.empty:
                continue

            high_beta_cut = float(beta_row.quantile(0.4))
            high_beta_names = beta_row[beta_row.ge(high_beta_cut)].index
            if len(high_beta_names) < max(6, self.top_n):
                continue

            soft_row = soft_score.loc[timestamp].reindex(high_beta_names).dropna()
            if len(soft_row) < max(6, self.top_n // 2):
                continue

            leadership_breadth = float(soft_row.gt(0.0).mean())
            leadership_strength = float(soft_row.clip(lower=0.0).mean())

            breadth_now = float(market_breadth.loc[timestamp])
            breadth_anchor = float(breadth_mean.loc[timestamp])
            breadth_change = float(breadth_delta.loc[timestamp])
            bench_ok = float(benchmark_trend.loc[timestamp]) > -0.03

            if breadth_now > max(breadth_anchor, 0.05) and breadth_change > 0.0 and leadership_breadth >= 0.55 and bench_ok:
                gross_long.loc[timestamp] = 1.0
            elif breadth_now > 0.0 and leadership_breadth >= 0.5 and bench_ok:
                gross_long.loc[timestamp] = 0.80
            elif leadership_breadth >= 0.45:
                gross_long.loc[timestamp] = 0.55
            else:
                gross_long.loc[timestamp] = 0.35

            candidate_pool = soft_row.sort_values(ascending=False).head(self.top_n * 3).index
            if len(candidate_pool) == 0:
                continue

            score = (
                soft_score.loc[timestamp].reindex(candidate_pool).fillna(0.0)
                + 0.20 * self._zscore(beta.loc[timestamp].reindex(candidate_pool).fillna(0.0))
                + 0.15 * self._zscore(consensus_level.loc[timestamp].reindex(candidate_pool).fillna(0.0))
                + 0.10 * self._zscore(consensus_flow.loc[timestamp].reindex(candidate_pool).fillna(0.0))
            )
            ranked = score.dropna().sort_values(ascending=False).head(self.top_n * 2)
            if ranked.empty:
                continue

            alpha.loc[timestamp, ranked.index] = ranked.astype(float)
            selected.loc[timestamp, ranked.index] = True

        return SignalBundle(
            alpha=alpha,
            context={
                "sector": sector,
                "selected": selected,
                "gross_long": gross_long,
            },
            meta={},
        )


@dataclass(slots=True)
class _SoftParticipationLongOnlyConstruction:
    top_n: int = 14
    max_per_sector: int = 4

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        alpha = bundle.alpha
        sector = bundle.context.get("sector")
        selected = bundle.context.get("selected")
        gross_long = bundle.context.get("gross_long")
        if not isinstance(sector, pd.DataFrame) or not isinstance(selected, pd.DataFrame):
            raise ValueError("soft participation long-only construction requires sector and selected context")

        sector = sector.reindex(index=alpha.index, columns=alpha.columns)
        selected = selected.reindex(index=alpha.index, columns=alpha.columns).fillna(False).astype(bool)
        if isinstance(gross_long, pd.Series):
            gross_long = gross_long.reindex(alpha.index).fillna(0.35).astype(float)
        else:
            gross_long = pd.Series(0.35, index=alpha.index, dtype=float)

        rows: dict[pd.Timestamp, pd.Series] = {}
        picked: dict[pd.Timestamp, pd.Series] = {}

        for timestamp in alpha.index:
            weights = pd.Series(0.0, index=alpha.columns, dtype=float)
            active_names = alpha.loc[timestamp][selected.loc[timestamp]].sort_values(ascending=False)
            if active_names.empty:
                rows[timestamp] = weights
                picked[timestamp] = weights.ne(0.0)
                continue

            sector_row = sector.loc[timestamp].reindex(active_names.index).fillna("unknown").astype(str)
            chosen: list[str] = []
            sector_counts: dict[str, int] = {}
            for name in active_names.index:
                sector_name = sector_row.get(name, "unknown")
                count = sector_counts.get(sector_name, 0)
                if count >= self.max_per_sector:
                    continue
                chosen.append(name)
                sector_counts[sector_name] = count + 1
                if len(chosen) >= self.top_n:
                    break

            if not chosen:
                rows[timestamp] = weights
                picked[timestamp] = weights.ne(0.0)
                continue

            gross = float(gross_long.loc[timestamp])
            raw = active_names.reindex(chosen).clip(lower=0.0)
            total = float(raw.sum())
            if total <= 0.0:
                weights.loc[chosen] = gross / len(chosen)
            else:
                weights.loc[chosen] = gross * (raw / total)
            rows[timestamp] = weights
            picked[timestamp] = weights.ne(0.0)

        base_target_weights = (
            pd.DataFrame.from_dict(rows, orient="index")
            .reindex(index=alpha.index, columns=alpha.columns)
            .fillna(0.0)
            .astype(float)
        )
        selection_mask = (
            pd.DataFrame.from_dict(picked, orient="index")
            .reindex(index=alpha.index, columns=alpha.columns)
            .fillna(False)
            .astype(bool)
        )
        return ConstructionResult(
            base_target_weights=base_target_weights,
            selection_mask=selection_mask,
            group_long_budget=None,
            group_short_budget=None,
            meta={},
        )
