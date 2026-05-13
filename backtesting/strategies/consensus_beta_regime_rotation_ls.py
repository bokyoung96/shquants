from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle

from .composable import ComposableStrategy
from .revision_asymmetric_relay_hedge_ls import AsymmetricRelayHedgeLongShort


@dataclass(slots=True)
class ConsensusBetaRegimeRotationLs(ComposableStrategy):
    lookback: int = 60
    flow_lookback: int = 20
    momentum_lookback: int = 120

    def __post_init__(self) -> None:
        self.signal_producer = _ConsensusBetaRegimeRotationSignalProducer(
            lookback=self.lookback,
            flow_lookback=self.flow_lookback,
            beta_lookback=max(120, self.momentum_lookback),
        )
        self.construction_rule = AsymmetricRelayHedgeLongShort()


@dataclass(slots=True)
class _ConsensusBetaRegimeRotationSignalProducer:
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

        market_balance = (positive_state.mean(axis=1) - negative_state.mean(axis=1)).fillna(0.0)
        market_mean = market_balance.rolling(self.lookback, min_periods=max(20, self.lookback // 3)).mean().fillna(0.0)
        market_delta = market_balance.diff(self.flow_lookback).fillna(0.0)
        risk_on = market_balance.gt(market_mean) & market_delta.gt(0.0) & benchmark_trend.gt(0.0)
        risk_off = market_balance.lt(market_mean) & market_delta.lt(0.0) & benchmark_trend.lt(0.0)

        gross_long = pd.Series(0.45, index=close.index, dtype=float)
        gross_short = pd.Series(0.45, index=close.index, dtype=float)
        gross_long.loc[risk_on] = 1.0
        gross_short.loc[risk_on] = 0.3
        gross_long.loc[risk_off] = 0.3
        gross_short.loc[risk_off] = 1.0

        long_mask = pd.DataFrame(False, index=close.index, columns=close.columns)
        short_mask = pd.DataFrame(False, index=close.index, columns=close.columns)
        alpha = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for timestamp in close.index:
            sector_row = sector.loc[timestamp].dropna().astype(str)
            if sector_row.empty:
                continue

            pos_row = positive_state.loc[timestamp]
            neg_row = negative_state.loc[timestamp]
            beta_row = beta.loc[timestamp]
            eps_trend_row = eps_trend.loc[timestamp]
            op_trend_row = op_trend.loc[timestamp]

            for sector_name, members in sector_row.groupby(sector_row, sort=False):
                member_index = members.index
                if len(member_index) < 5:
                    continue

                sector_pos = pos_row.reindex(member_index).fillna(False)
                sector_neg = neg_row.reindex(member_index).fillna(False)
                balance_now = float(sector_pos.mean() - sector_neg.mean())

                prior_timestamp = timestamp - pd.Timedelta(days=self.flow_lookback)
                prior_slice = close.index[close.index <= prior_timestamp]
                prev_date = prior_slice[-1] if len(prior_slice) else None
                balance_prev = 0.0
                if prev_date is not None:
                    prev_labels = sector.loc[prev_date].dropna().astype(str)
                    prev_members = prev_labels[prev_labels.eq(sector_name)].index
                    if len(prev_members) > 0:
                        prev_pos = positive_state.loc[prev_date].reindex(prev_members).fillna(False)
                        prev_neg = negative_state.loc[prev_date].reindex(prev_members).fillna(False)
                        balance_prev = float(prev_pos.mean() - prev_neg.mean())
                balance_delta = balance_now - balance_prev

                beta_slice = beta_row.reindex(member_index).dropna()
                if len(beta_slice) < 4:
                    continue
                high_beta_cut = float(beta_slice.quantile(0.6))
                high_beta = beta_slice[beta_slice.ge(high_beta_cut)].index
                if len(high_beta) == 0:
                    continue

                if bool(risk_on.loc[timestamp]) and balance_now > 0.0 and balance_delta > 0.0:
                    longs = high_beta.intersection(member_index[sector_pos.to_numpy()])
                    if len(longs) > 0:
                        trend_strength = (
                            eps_trend_row.reindex(longs).fillna(0.0) + op_trend_row.reindex(longs).fillna(0.0)
                        ) / 2.0
                        beta_strength = self._zscore(beta_row.reindex(longs).fillna(0.0))
                        alpha.loc[timestamp, longs] = (trend_strength + 0.5 * beta_strength).astype(float)
                        long_mask.loc[timestamp, longs] = True

                if bool(risk_off.loc[timestamp]) and balance_now < 0.0 and balance_delta < 0.0:
                    shorts = high_beta.intersection(member_index[sector_neg.to_numpy()])
                    if len(shorts) > 0:
                        trend_strength = (
                            eps_trend_row.reindex(shorts).fillna(0.0).abs() + op_trend_row.reindex(shorts).fillna(0.0).abs()
                        ) / 2.0
                        beta_strength = self._zscore(beta_row.reindex(shorts).fillna(0.0))
                        alpha.loc[timestamp, shorts] = -(trend_strength + 0.5 * beta_strength).astype(float)
                        short_mask.loc[timestamp, shorts] = True

        return SignalBundle(
            alpha=alpha,
            context={
                "sector": sector,
                "long_mask": long_mask,
                "short_mask": short_mask,
                "gross_long": gross_long,
                "gross_short": gross_short,
            },
            meta={},
        )
