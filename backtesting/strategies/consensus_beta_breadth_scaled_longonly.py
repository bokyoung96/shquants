from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle

from .composable import ComposableStrategy


@dataclass(slots=True)
class ConsensusBetaBreadthScaledLongOnly(ComposableStrategy):
    lookback: int = 60
    flow_lookback: int = 20
    momentum_lookback: int = 120
    top_n: int = 12

    def __post_init__(self) -> None:
        self.signal_producer = _ConsensusBetaBreadthScaledSignalProducer(
            lookback=self.lookback,
            flow_lookback=self.flow_lookback,
            beta_lookback=max(120, self.momentum_lookback),
            top_n=self.top_n,
        )
        self.construction_rule = _BreadthScaledLongOnlyConstruction(top_n=self.top_n)


@dataclass(slots=True)
class _ConsensusBetaBreadthScaledSignalProducer:
    lookback: int = 60
    flow_lookback: int = 20
    beta_lookback: int = 120
    top_n: int = 12

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

        positive_state = eps_trend.gt(0.0) & op_trend.gt(0.0) & eps_delta.gt(0.0) & op_delta.gt(0.0)
        negative_state = eps_trend.lt(0.0) & op_trend.lt(0.0) & eps_delta.lt(0.0) & op_delta.lt(0.0)

        market_breadth = (positive_state.mean(axis=1) - negative_state.mean(axis=1)).fillna(0.0)
        breadth_mean = market_breadth.rolling(self.lookback, min_periods=max(20, self.lookback // 3)).mean().fillna(0.0)
        breadth_delta = market_breadth.diff(self.flow_lookback).fillna(0.0)

        gross_long = pd.Series(0.20, index=close.index, dtype=float)
        alpha = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        selected = pd.DataFrame(False, index=close.index, columns=close.columns)

        for timestamp in close.index:
            pos_row = positive_state.loc[timestamp]
            beta_row = beta.loc[timestamp].dropna()
            if beta_row.empty:
                continue

            high_beta_cut = float(beta_row.quantile(0.5))
            high_beta_names = beta_row[beta_row.ge(high_beta_cut)].index
            if len(high_beta_names) < max(4, self.top_n // 2):
                continue

            high_beta_pos = pos_row.reindex(high_beta_names).fillna(False)
            leadership_breadth = float(high_beta_pos.mean())
            leadership_mean = 0.0
            prior_ts = timestamp - pd.Timedelta(days=self.lookback)
            prior_slice = close.index[close.index <= prior_ts]
            if len(prior_slice) > 0:
                window_idx = close.index[(close.index > prior_slice[-1]) & (close.index <= timestamp)]
                leadership_hist = []
                for hist_ts in window_idx:
                    hist_beta = beta.loc[hist_ts].dropna()
                    if hist_beta.empty:
                        continue
                    hist_cut = float(hist_beta.quantile(0.5))
                    hist_high_beta = hist_beta[hist_beta.ge(hist_cut)].index
                    if len(hist_high_beta) == 0:
                        continue
                    hist_pos = positive_state.loc[hist_ts].reindex(hist_high_beta).fillna(False)
                    leadership_hist.append(float(hist_pos.mean()))
                if leadership_hist:
                    leadership_mean = float(pd.Series(leadership_hist, dtype=float).mean())

            risk_on = (
                market_breadth.loc[timestamp] > breadth_mean.loc[timestamp]
                and breadth_delta.loc[timestamp] > 0.0
                and leadership_breadth > leadership_mean
                and benchmark_trend.loc[timestamp] > 0.0
            )
            neutral = (
                market_breadth.loc[timestamp] > 0.0
                and leadership_breadth > max(leadership_mean, 0.5)
            )
            if risk_on:
                gross_long.loc[timestamp] = 1.0
            elif neutral:
                gross_long.loc[timestamp] = 0.60
            else:
                gross_long.loc[timestamp] = 0.20

            candidate_pool = high_beta_names.intersection(pos_row[pos_row].index)
            if len(candidate_pool) == 0:
                continue

            eps_row = eps_trend.loc[timestamp].reindex(candidate_pool).fillna(0.0)
            op_row = op_trend.loc[timestamp].reindex(candidate_pool).fillna(0.0)
            eps_delta_row = eps_delta.loc[timestamp].reindex(candidate_pool).fillna(0.0)
            op_delta_row = op_delta.loc[timestamp].reindex(candidate_pool).fillna(0.0)
            beta_strength = self._zscore(beta.loc[timestamp].reindex(candidate_pool).fillna(0.0))
            score = (
                eps_row
                + op_row
                + 0.5 * eps_delta_row
                + 0.5 * op_delta_row
                + 0.25 * beta_strength
            ) / 3.25
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
class _BreadthScaledLongOnlyConstruction:
    top_n: int = 12
    max_per_sector: int = 3

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        alpha = bundle.alpha
        sector = bundle.context.get("sector")
        selected = bundle.context.get("selected")
        gross_long = bundle.context.get("gross_long")
        if not isinstance(sector, pd.DataFrame) or not isinstance(selected, pd.DataFrame):
            raise ValueError("breadth scaled long-only construction requires sector and selected context")

        sector = sector.reindex(index=alpha.index, columns=alpha.columns)
        selected = selected.reindex(index=alpha.index, columns=alpha.columns).fillna(False).astype(bool)
        if isinstance(gross_long, pd.Series):
            gross_long = gross_long.reindex(alpha.index).fillna(0.20).astype(float)
        else:
            gross_long = pd.Series(0.20, index=alpha.index, dtype=float)

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

            per_name = float(gross_long.loc[timestamp]) / len(chosen)
            weights.loc[chosen] = per_name
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
