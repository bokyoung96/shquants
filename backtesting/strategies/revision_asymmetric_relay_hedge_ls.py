from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle

from .composable import ComposableStrategy


@dataclass(slots=True)
class AsymmetricRelayHedgeLongShort:
    default_long_gross: float = 0.8
    default_short_gross: float = 0.2

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        alpha = bundle.alpha.fillna(0.0)
        sector = bundle.context.get("sector")
        long_mask = bundle.context.get("long_mask")
        short_mask = bundle.context.get("short_mask")
        gross_long = bundle.context.get("gross_long")
        gross_short = bundle.context.get("gross_short")
        if not isinstance(sector, pd.DataFrame) or not isinstance(long_mask, pd.DataFrame) or not isinstance(short_mask, pd.DataFrame):
            raise ValueError("asymmetric relay hedge construction requires sector, long_mask, short_mask context")

        sector = sector.reindex(index=alpha.index, columns=alpha.columns)
        long_mask = long_mask.reindex(index=alpha.index, columns=alpha.columns).fillna(False).astype(bool)
        short_mask = short_mask.reindex(index=alpha.index, columns=alpha.columns).fillna(False).astype(bool)
        if isinstance(gross_long, pd.Series):
            gross_long = gross_long.reindex(alpha.index).fillna(self.default_long_gross)
        else:
            gross_long = pd.Series(self.default_long_gross, index=alpha.index, dtype=float)
        if isinstance(gross_short, pd.Series):
            gross_short = gross_short.reindex(alpha.index).fillna(self.default_short_gross)
        else:
            gross_short = pd.Series(self.default_short_gross, index=alpha.index, dtype=float)

        weights_by_date = {}
        selected_by_date = {}
        selected_long_by_date = {}
        selected_short_by_date = {}

        for timestamp in alpha.index:
            weights = pd.Series(0.0, index=alpha.columns, dtype=float)
            selected = pd.Series(False, index=alpha.columns, dtype=bool)
            selected_long = pd.Series(False, index=alpha.columns, dtype=bool)
            selected_short = pd.Series(False, index=alpha.columns, dtype=bool)

            sector_row = sector.loc[timestamp].dropna().astype(str)
            long_groups = []
            short_groups = []
            for _, members in sector_row.groupby(sector_row, sort=False):
                member_index = members.index
                longs = member_index[long_mask.loc[timestamp].reindex(member_index).fillna(False).to_numpy()]
                shorts = member_index[short_mask.loc[timestamp].reindex(member_index).fillna(False).to_numpy()]
                if len(longs) > 0:
                    long_groups.append(longs)
                if len(shorts) > 0:
                    short_groups.append(shorts)

            if long_groups:
                per_sector_long = float(gross_long.loc[timestamp]) / len(long_groups)
                for longs in long_groups:
                    weights.loc[longs] += per_sector_long / len(longs)
                    selected.loc[longs] = True
                    selected_long.loc[longs] = True

            if short_groups:
                per_sector_short = float(gross_short.loc[timestamp]) / len(short_groups)
                for shorts in short_groups:
                    weights.loc[shorts] -= per_sector_short / len(shorts)
                    selected.loc[shorts] = True
                    selected_short.loc[shorts] = True

            weights_by_date[timestamp] = weights
            selected_by_date[timestamp] = selected
            selected_long_by_date[timestamp] = selected_long
            selected_short_by_date[timestamp] = selected_short

        return ConstructionResult(
            base_target_weights=pd.DataFrame.from_dict(weights_by_date, orient="index").reindex(index=alpha.index, columns=alpha.columns).fillna(0.0).astype(float),
            selection_mask=pd.DataFrame.from_dict(selected_by_date, orient="index").reindex(index=alpha.index, columns=alpha.columns).fillna(False).astype(bool),
            group_long_budget=None,
            group_short_budget=None,
            meta={
                "selected_long": pd.DataFrame.from_dict(selected_long_by_date, orient="index").reindex(index=alpha.index, columns=alpha.columns).fillna(False).astype(bool),
                "selected_short": pd.DataFrame.from_dict(selected_short_by_date, orient="index").reindex(index=alpha.index, columns=alpha.columns).fillna(False).astype(bool),
            },
        )


@dataclass(slots=True)
class RevisionAsymmetricRelayHedgeLs(ComposableStrategy):
    lookback: int = 20
    flow_lookback: int = 20

    def __post_init__(self) -> None:
        self.signal_producer = _RevisionAsymmetricRelayHedgeSignalProducer(
            lookback=self.lookback,
            flow_lookback=self.flow_lookback,
            regime_lookback=60,
        )
        self.construction_rule = AsymmetricRelayHedgeLongShort()


@dataclass(slots=True)
class _RevisionAsymmetricRelayHedgeSignalProducer:
    lookback: int = 20
    flow_lookback: int = 20
    regime_lookback: int = 60

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_EPS_NFQ1,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_WICS_SEC_BIG,
        )

    def build(self, market: MarketData) -> SignalBundle:
        eps = market.frames["eps_fwd_q1"]
        op = market.frames["op_fwd_q1"]
        sector = market.frames["sector_big"]

        eps_rev = eps.pct_change(self.lookback, fill_method=None)
        op_rev = op.pct_change(self.lookback, fill_method=None)
        eps_prev = eps_rev.shift(self.flow_lookback)
        op_prev = op_rev.shift(self.flow_lookback)
        eps_accel = eps_rev - eps_prev
        op_accel = op_rev - op_prev

        positive_now = eps_rev.gt(0.0) & op_rev.gt(0.0)
        negative_now = eps_rev.lt(0.0) & op_rev.lt(0.0)
        long_confirm = eps_accel.gt(0.0) & op_accel.gt(0.0)
        short_confirm = eps_accel.lt(0.0) & op_accel.lt(0.0)

        market_balance = (positive_now.mean(axis=1) - negative_now.mean(axis=1)).fillna(0.0)
        market_mean = market_balance.rolling(self.regime_lookback, min_periods=10).mean().fillna(0.0)
        market_delta = market_balance.diff(self.flow_lookback).fillna(0.0)

        bull_regime = market_balance.gt(market_mean) & market_delta.gt(0.0)
        bear_regime = market_balance.lt(market_mean) & market_delta.lt(0.0)

        gross_long = pd.Series(0.0, index=eps.index, dtype=float)
        gross_short = pd.Series(0.0, index=eps.index, dtype=float)
        gross_long.loc[bull_regime] = 1.0
        gross_short.loc[bull_regime] = 0.35
        gross_long.loc[bear_regime] = 0.35
        gross_short.loc[bear_regime] = 1.0
        neutral = ~(bull_regime | bear_regime)
        gross_long.loc[neutral] = 0.65
        gross_short.loc[neutral] = 0.65

        long_mask = pd.DataFrame(False, index=eps.index, columns=eps.columns)
        short_mask = pd.DataFrame(False, index=eps.index, columns=eps.columns)
        alpha = pd.DataFrame(0.0, index=eps.index, columns=eps.columns)

        for timestamp in eps.index:
            sector_row = sector.loc[timestamp].dropna().astype(str)
            if sector_row.empty:
                continue
            prior_timestamp = timestamp - pd.Timedelta(days=self.flow_lookback)
            prior_slice = eps.index[eps.index <= prior_timestamp]
            prev_date = prior_slice[-1] if len(prior_slice) else None

            for sector_name, members in sector_row.groupby(sector_row, sort=False):
                member_index = members.index
                pos_slice = positive_now.loc[timestamp].reindex(member_index).fillna(False)
                neg_slice = negative_now.loc[timestamp].reindex(member_index).fillna(False)
                long_conf_slice = long_confirm.loc[timestamp].reindex(member_index).fillna(False)
                short_conf_slice = short_confirm.loc[timestamp].reindex(member_index).fillna(False)

                pos_breadth = float(pos_slice.mean())
                neg_breadth = float(neg_slice.mean())
                balance_now = pos_breadth - neg_breadth
                if prev_date is None:
                    balance_prev = 0.0
                else:
                    prev_labels = sector.loc[prev_date].dropna().astype(str)
                    prev_members = prev_labels[prev_labels.eq(sector_name)].index
                    if len(prev_members) == 0:
                        balance_prev = 0.0
                    else:
                        prev_pos = float(positive_now.loc[prev_date].reindex(prev_members).fillna(False).mean())
                        prev_neg = float(negative_now.loc[prev_date].reindex(prev_members).fillna(False).mean())
                        balance_prev = prev_pos - prev_neg
                balance_delta = balance_now - balance_prev

                sector_long_live = balance_now > 0.0 and balance_delta > 0.0
                sector_short_live = balance_now < 0.0 and balance_delta < 0.0

                if sector_long_live:
                    longs = member_index[(pos_slice & long_conf_slice).to_numpy()]
                    if len(longs) > 0:
                        long_mask.loc[timestamp, longs] = True
                        alpha.loc[timestamp, longs] = abs(balance_now) + abs(balance_delta)
                if sector_short_live:
                    shorts = member_index[(neg_slice & short_conf_slice).to_numpy()]
                    if len(shorts) > 0:
                        short_mask.loc[timestamp, shorts] = True
                        alpha.loc[timestamp, shorts] = -(abs(balance_now) + abs(balance_delta))

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
