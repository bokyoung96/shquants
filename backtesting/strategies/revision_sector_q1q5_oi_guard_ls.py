from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle

from .composable import ComposableStrategy


@dataclass(slots=True)
class SectorOiGuardNeutralLongShort:
    gross_long: float = 1.0
    gross_short: float = 1.0
    max_name_weight: float = 0.10
    short_realloc_fraction: float = 0.0

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        alpha = bundle.alpha.fillna(0.0)
        sector = bundle.context.get("sector")
        long_mask = bundle.context.get("long_mask")
        short_mask = bundle.context.get("short_mask")
        short_conf = bundle.context.get("short_confidence")
        if not isinstance(sector, pd.DataFrame) or not isinstance(long_mask, pd.DataFrame) or not isinstance(short_mask, pd.DataFrame):
            raise ValueError("sector oi guard construction requires sector, long_mask, short_mask context")

        sector = sector.reindex(index=alpha.index, columns=alpha.columns)
        long_mask = long_mask.reindex(index=alpha.index, columns=alpha.columns).fillna(False).astype(bool)
        short_mask = short_mask.reindex(index=alpha.index, columns=alpha.columns).fillna(False).astype(bool)
        if isinstance(short_conf, pd.DataFrame):
            short_conf = short_conf.reindex(index=alpha.index, columns=alpha.columns).fillna(0.0).astype(float)

        weights_by_date: dict[pd.Timestamp, pd.Series] = {}
        selected_by_date: dict[pd.Timestamp, pd.Series] = {}
        for timestamp in alpha.index:
            weights = pd.Series(0.0, index=alpha.columns, dtype=float)
            selected = pd.Series(False, index=alpha.columns, dtype=bool)
            sector_row = sector.loc[timestamp].dropna().astype(str)
            qualified: list[tuple[pd.Index, pd.Index]] = []
            for _, members in sector_row.groupby(sector_row, sort=False):
                member_index = members.index
                longs = member_index[long_mask.loc[timestamp].reindex(member_index).fillna(False).to_numpy()]
                shorts = member_index[short_mask.loc[timestamp].reindex(member_index).fillna(False).to_numpy()]
                if len(longs) > 0 and len(shorts) > 0:
                    qualified.append((longs, shorts))

            sector_count = len(qualified)
            if sector_count > 0:
                long_budget = self.gross_long / sector_count
                short_budget = self.gross_short / sector_count
                for longs, shorts in qualified:
                    weights.loc[longs] = long_budget / len(longs)
                    selected.loc[longs] = True

                    if len(shorts) == 0:
                        continue
                    if self.short_realloc_fraction > 0.0 and isinstance(short_conf, pd.DataFrame):
                        conf = short_conf.loc[timestamp].reindex(shorts).fillna(0.0).clip(lower=0.0)
                        if float(conf.sum()) > 0.0:
                            effective_budget = short_budget * min(1.0, len(shorts) * self.short_realloc_fraction)
                            raw = effective_budget * conf / float(conf.sum())
                        else:
                            raw = pd.Series(short_budget / len(shorts), index=shorts, dtype=float)
                    else:
                        raw = pd.Series(short_budget / len(shorts), index=shorts, dtype=float)

                    capped = raw.clip(upper=self.max_name_weight)
                    weights.loc[shorts] = -capped
                    selected.loc[shorts] = True

            weights_by_date[timestamp] = weights
            selected_by_date[timestamp] = selected

        base_target_weights = (
            pd.DataFrame.from_dict(weights_by_date, orient="index")
            .reindex(index=alpha.index, columns=alpha.columns)
            .fillna(0.0)
            .astype(float)
        )
        selection_mask = (
            pd.DataFrame.from_dict(selected_by_date, orient="index")
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


@dataclass(slots=True)
class RevisionSectorQ1Q5OiHardSkipLs(ComposableStrategy):
    lookback: int = 20
    flow_lookback: int = 20
    regime_lookback: int = 60

    def __post_init__(self) -> None:
        self.signal_producer = _RevisionSectorQ1Q5OiGuardSignalProducer(
            lookback=self.lookback,
            flow_lookback=self.flow_lookback,
            regime_lookback=self.regime_lookback,
            confidence_weighted=False,
        )
        self.construction_rule = SectorOiGuardNeutralLongShort(short_realloc_fraction=0.0)


@dataclass(slots=True)
class RevisionSectorQ1Q5OiConfidenceWeightedLs(ComposableStrategy):
    lookback: int = 20
    flow_lookback: int = 20
    regime_lookback: int = 60

    def __post_init__(self) -> None:
        self.signal_producer = _RevisionSectorQ1Q5OiGuardSignalProducer(
            lookback=self.lookback,
            flow_lookback=self.flow_lookback,
            regime_lookback=self.regime_lookback,
            confidence_weighted=True,
        )
        self.construction_rule = SectorOiGuardNeutralLongShort(short_realloc_fraction=0.6)


@dataclass(slots=True)
class _RevisionSectorQ1Q5OiGuardSignalProducer:
    lookback: int = 20
    flow_lookback: int = 20
    regime_lookback: int = 60
    confidence_weighted: bool = False

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_V,
            DatasetId.QW_EPS_NFQ1,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_FOREIGN,
            DatasetId.QW_INSTITUTION,
            DatasetId.QW_RETAIL,
            DatasetId.QW_WICS_SEC_BIG,
        )

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean) < 2:
            return pd.Series(0.0, index=series.index, dtype=float)
        std = float(clean.std())
        if std <= 0.0 or pd.isna(std):
            return pd.Series(0.0, index=series.index, dtype=float)
        return ((pd.to_numeric(series, errors="coerce") - float(clean.mean())) / std).fillna(0.0).astype(float)

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"]
        volume = market.frames["volume"]
        eps = market.frames["eps_fwd_q1"]
        op = market.frames["op_fwd_q1"]
        foreign = market.frames["foreign_flow"]
        inst = market.frames["inst_flow"]
        retail = market.frames["retail_flow"]
        sector = market.frames["sector_big"]

        trading_index = close.index.intersection(volume.index).intersection(foreign.index).intersection(inst.index).intersection(retail.index)
        close = close.reindex(trading_index)
        volume = volume.reindex(trading_index)
        foreign = foreign.reindex(trading_index)
        inst = inst.reindex(trading_index)
        retail = retail.reindex(trading_index)
        eps = eps.reindex(trading_index).ffill()
        op = op.reindex(trading_index).ffill()
        sector = sector.reindex(trading_index).ffill()

        adv = (close * volume).rolling(self.flow_lookback, min_periods=max(5, self.flow_lookback // 2)).mean()
        net_oi = (foreign.add(inst, fill_value=0.0).sub(retail, fill_value=0.0)).rolling(
            self.flow_lookback, min_periods=max(5, self.flow_lookback // 2)
        ).sum()
        oi_intensity = net_oi.divide(adv.replace(0.0, pd.NA))
        oi_impulse = oi_intensity - oi_intensity.shift(self.flow_lookback)

        eps_rev = eps.pct_change(self.lookback, fill_method=None)
        op_rev = op.pct_change(self.lookback, fill_method=None)
        eps_prev = eps_rev.shift(self.flow_lookback)
        op_prev = op_rev.shift(self.flow_lookback)
        eps_accel = eps_rev - eps_prev
        op_accel = op_rev - op_prev

        score = 0.5 * eps_rev.rank(axis=1, pct=True) + 0.5 * op_rev.rank(axis=1, pct=True)
        accel_score = 0.5 * eps_accel.rank(axis=1, pct=True) + 0.5 * op_accel.rank(axis=1, pct=True)
        composite = score + accel_score

        positive_now = eps_rev.gt(0.0) & op_rev.gt(0.0)
        negative_now = eps_rev.lt(0.0) & op_rev.lt(0.0)
        market_balance = (positive_now.mean(axis=1) - negative_now.mean(axis=1)).fillna(0.0)
        market_mean = market_balance.rolling(self.regime_lookback, min_periods=10).mean().fillna(0.0)
        market_live = market_balance.gt(market_mean)

        long_mask = pd.DataFrame(False, index=eps.index, columns=eps.columns)
        short_mask = pd.DataFrame(False, index=eps.index, columns=eps.columns)
        alpha = pd.DataFrame(0.0, index=eps.index, columns=eps.columns)
        short_confidence = pd.DataFrame(0.0, index=eps.index, columns=eps.columns)

        for timestamp in eps.index:
            if not bool(market_live.loc[timestamp]):
                continue

            sector_row = sector.loc[timestamp].dropna().astype(str)
            if sector_row.empty:
                continue
            score_row = composite.loc[timestamp].dropna()
            if score_row.empty:
                continue
            pos_row = positive_now.loc[timestamp].reindex(score_row.index).fillna(False)
            neg_row = negative_now.loc[timestamp].reindex(score_row.index).fillna(False)
            oi_now_row = oi_intensity.loc[timestamp].reindex(score_row.index)
            oi_imp_row = oi_impulse.loc[timestamp].reindex(score_row.index)

            for _, members in sector_row.groupby(sector_row, sort=False):
                member_index = members.index.intersection(score_row.index)
                if len(member_index) < 5:
                    continue
                sector_scores = score_row.reindex(member_index).dropna()
                if len(sector_scores) < 5:
                    continue

                long_candidates = sector_scores[pos_row.reindex(sector_scores.index).fillna(False)]
                short_candidates = sector_scores[neg_row.reindex(sector_scores.index).fillna(False)]
                if len(long_candidates) == 0 or len(short_candidates) == 0:
                    continue

                long_cut = long_candidates.quantile(0.8)
                short_cut = short_candidates.quantile(0.2)
                longs = long_candidates[long_candidates >= long_cut].index
                raw_shorts = short_candidates[short_candidates <= short_cut].index
                if len(longs) == 0 or len(raw_shorts) == 0:
                    continue

                oi_sector = pd.concat(
                    [
                        oi_now_row.reindex(raw_shorts).rename("oi_level"),
                        oi_imp_row.reindex(raw_shorts).rename("oi_impulse"),
                        sector_scores.reindex(raw_shorts).rename("revision_score"),
                    ],
                    axis=1,
                ).dropna()
                if oi_sector.empty:
                    continue

                oi_level_z = self._zscore(oi_sector["oi_level"])
                oi_imp_z = self._zscore(oi_sector["oi_impulse"])
                revision_z = self._zscore(oi_sector["revision_score"])
                squeeze_risk = 0.75 * oi_level_z + 0.75 * oi_imp_z + 0.25 * revision_z
                keep_mask = squeeze_risk <= 0.25
                filtered = oi_sector.index[keep_mask.reindex(oi_sector.index).fillna(False)]
                if len(filtered) == 0:
                    continue

                long_mask.loc[timestamp, longs] = True
                short_mask.loc[timestamp, filtered] = True
                alpha.loc[timestamp, longs] = composite.loc[timestamp, longs].astype(float)
                alpha.loc[timestamp, filtered] = -composite.loc[timestamp, filtered].astype(float)

                if self.confidence_weighted:
                    conf = (-squeeze_risk.reindex(filtered).fillna(0.0) + 0.25).clip(lower=0.0) + revision_z.reindex(filtered).abs().fillna(0.0)
                    short_confidence.loc[timestamp, filtered] = conf.astype(float)
                else:
                    short_confidence.loc[timestamp, filtered] = 1.0

        return SignalBundle(
            alpha=alpha,
            context={"sector": sector, "long_mask": long_mask, "short_mask": short_mask, "short_confidence": short_confidence},
            meta={},
        )
