from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle

from .composable import ComposableStrategy


@dataclass(slots=True)
class ConsensusBetaSectorTiltLongOnly(ComposableStrategy):
    lookback: int = 60
    flow_lookback: int = 20
    momentum_lookback: int = 120
    top_n: int = 12

    def __post_init__(self) -> None:
        self.signal_producer = _ConsensusBetaSectorTiltSignalProducer(
            lookback=self.lookback,
            flow_lookback=self.flow_lookback,
            beta_lookback=max(120, self.momentum_lookback),
            top_n=self.top_n,
        )
        self.construction_rule = _SectorTiltLongOnlyConstruction(top_n=self.top_n)


@dataclass(slots=True)
class _ConsensusBetaSectorTiltSignalProducer:
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
        market_mean = market_breadth.rolling(self.lookback, min_periods=max(20, self.lookback // 3)).mean().fillna(0.0)
        breadth_delta = market_breadth.diff(self.flow_lookback).fillna(0.0)

        gross_long = pd.Series(0.35, index=close.index, dtype=float)
        risk_on = market_breadth.gt(market_mean) & breadth_delta.gt(0.0) & benchmark_trend.gt(0.0)
        risk_off = market_breadth.lt(market_mean) & breadth_delta.lt(0.0) & benchmark_trend.lt(0.0)
        gross_long.loc[risk_on] = 1.0
        gross_long.loc[~risk_on & ~risk_off] = 0.65

        alpha = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        selected = pd.DataFrame(False, index=close.index, columns=close.columns)
        sector_budget = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for timestamp in close.index:
            sector_row = sector.loc[timestamp].dropna().astype(str)
            if sector_row.empty:
                continue

            pos_row = positive_state.loc[timestamp]
            beta_row = beta.loc[timestamp]
            eps_trend_row = eps_trend.loc[timestamp]
            op_trend_row = op_trend.loc[timestamp]
            eps_delta_row = eps_delta.loc[timestamp]
            op_delta_row = op_delta.loc[timestamp]

            sector_scores: list[tuple[str, float, pd.Index]] = []
            sector_candidates: dict[str, pd.Index] = {}

            for sector_name, members in sector_row.groupby(sector_row, sort=False):
                member_index = members.index
                if len(member_index) < 5:
                    continue

                sector_pos = pos_row.reindex(member_index).fillna(False)
                balance_now = float(sector_pos.mean())

                prior_timestamp = timestamp - pd.Timedelta(days=self.flow_lookback)
                prior_slice = close.index[close.index <= prior_timestamp]
                prev_date = prior_slice[-1] if len(prior_slice) else None
                balance_prev = 0.0
                if prev_date is not None:
                    prev_labels = sector.loc[prev_date].dropna().astype(str)
                    prev_members = prev_labels[prev_labels.eq(sector_name)].index
                    if len(prev_members) > 0:
                        prev_pos = positive_state.loc[prev_date].reindex(prev_members).fillna(False)
                        balance_prev = float(prev_pos.mean())
                balance_delta = balance_now - balance_prev

                beta_slice = beta_row.reindex(member_index).dropna()
                if len(beta_slice) < 4:
                    continue
                high_beta_cut = float(beta_slice.quantile(0.6))
                leadership = beta_slice[beta_slice.ge(high_beta_cut)].index
                if len(leadership) == 0:
                    continue

                candidate_pool = leadership.intersection(member_index[sector_pos.to_numpy()])
                if len(candidate_pool) == 0:
                    continue

                stock_strength = (
                    eps_trend_row.reindex(candidate_pool).fillna(0.0)
                    + op_trend_row.reindex(candidate_pool).fillna(0.0)
                    + 0.5 * eps_delta_row.reindex(candidate_pool).fillna(0.0)
                    + 0.5 * op_delta_row.reindex(candidate_pool).fillna(0.0)
                ) / 3.0
                beta_strength = self._zscore(beta_row.reindex(candidate_pool).fillna(0.0))
                combined = (stock_strength + 0.35 * beta_strength).dropna().sort_values(ascending=False)
                if combined.empty:
                    continue

                sector_score = balance_now + balance_delta
                if sector_score <= 0.0:
                    continue

                keep = combined.head(max(1, min(4, self.top_n)))
                sector_scores.append((sector_name, float(sector_score), keep.index))
                sector_candidates[sector_name] = keep.index
                alpha.loc[timestamp, keep.index] = keep.astype(float)

            if not sector_scores:
                continue

            active = sorted(sector_scores, key=lambda item: item[1], reverse=True)[:3]
            score_total = sum(max(item[1], 0.0) for item in active)
            if score_total <= 0.0:
                continue

            for sector_name, score, names in active:
                budget = max(score, 0.0) / score_total
                sector_budget.loc[timestamp, names] = budget
                selected.loc[timestamp, names] = True

        return SignalBundle(
            alpha=alpha,
            context={
                "sector": sector,
                "selected": selected,
                "sector_budget": sector_budget,
                "gross_long": gross_long,
            },
            meta={},
        )


@dataclass(slots=True)
class _SectorTiltLongOnlyConstruction:
    top_n: int = 12

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        alpha = bundle.alpha
        sector = bundle.context.get("sector")
        selected = bundle.context.get("selected")
        sector_budget = bundle.context.get("sector_budget")
        gross_long = bundle.context.get("gross_long")
        if not isinstance(sector, pd.DataFrame) or not isinstance(selected, pd.DataFrame) or not isinstance(sector_budget, pd.DataFrame):
            raise ValueError("sector tilt long-only construction requires sector, selected, sector_budget context")

        sector = sector.reindex(index=alpha.index, columns=alpha.columns)
        selected = selected.reindex(index=alpha.index, columns=alpha.columns).fillna(False).astype(bool)
        sector_budget = sector_budget.reindex(index=alpha.index, columns=alpha.columns).fillna(0.0).astype(float)
        if isinstance(gross_long, pd.Series):
            gross_long = gross_long.reindex(alpha.index).fillna(0.35).astype(float)
        else:
            gross_long = pd.Series(0.35, index=alpha.index, dtype=float)

        rows: dict[pd.Timestamp, pd.Series] = {}
        picked: dict[pd.Timestamp, pd.Series] = {}

        for timestamp in alpha.index:
            weights = pd.Series(0.0, index=alpha.columns, dtype=float)
            active = selected.loc[timestamp]
            chosen = alpha.loc[timestamp][active].sort_values(ascending=False).head(self.top_n)
            if chosen.empty:
                rows[timestamp] = weights
                picked[timestamp] = weights.ne(0.0)
                continue

            sector_row = sector.loc[timestamp].reindex(chosen.index).astype(str)
            budgets = sector_budget.loc[timestamp].reindex(chosen.index).fillna(0.0)
            sector_names = sector_row.dropna().unique().tolist()
            gross = float(gross_long.loc[timestamp])

            raw_sector_budget: dict[str, float] = {}
            for sector_name in sector_names:
                sector_names_idx = sector_row[sector_row.eq(sector_name)].index
                raw_sector_budget[sector_name] = float(budgets.reindex(sector_names_idx).max())
            total_budget = sum(raw_sector_budget.values())
            if total_budget <= 0.0:
                raw_sector_budget = {name: 1.0 / len(sector_names) for name in sector_names}
            else:
                raw_sector_budget = {name: value / total_budget for name, value in raw_sector_budget.items()}

            for sector_name in sector_names:
                sector_names_idx = chosen.index[sector_row.eq(sector_name)]
                if len(sector_names_idx) == 0:
                    continue
                sector_weight = gross * raw_sector_budget[sector_name]
                per_name = sector_weight / len(sector_names_idx)
                weights.loc[sector_names_idx] = per_name

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
