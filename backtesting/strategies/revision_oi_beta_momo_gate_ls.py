from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle

from .composable import ComposableStrategy
from .revision_sector_q1q5_oi_guard_ls import SectorOiGuardNeutralLongShort


@dataclass(slots=True)
class RevisionOiBetaMomoGateLs(ComposableStrategy):
    lookback: int = 20
    flow_lookback: int = 20
    momentum_lookback: int = 60

    def __post_init__(self) -> None:
        self.signal_producer = _RevisionOiBetaMomoGateSignalProducer(
            lookback=self.lookback,
            flow_lookback=self.flow_lookback,
            momentum_lookback=self.momentum_lookback,
            regime_lookback=max(60, self.momentum_lookback),
        )
        self.construction_rule = SectorOiGuardNeutralLongShort(short_realloc_fraction=0.4)


@dataclass(slots=True)
class _RevisionOiBetaMomoGateSignalProducer:
    lookback: int = 20
    flow_lookback: int = 20
    momentum_lookback: int = 60
    regime_lookback: int = 60

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_V,
            DatasetId.QW_BM,
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
        benchmark_frame = market.frames["benchmark"]
        eps = market.frames["eps_fwd_q1"]
        op = market.frames["op_fwd_q1"]
        foreign = market.frames["foreign_flow"]
        inst = market.frames["inst_flow"]
        retail = market.frames["retail_flow"]
        sector = market.frames["sector_big"]

        benchmark = benchmark_frame["IKS200"] if "IKS200" in benchmark_frame.columns else benchmark_frame.iloc[:, 0]
        benchmark = benchmark.reindex(close.index).ffill()
        bench_ret = benchmark.pct_change(fill_method=None)
        stock_ret = close.pct_change(fill_method=None)

        beta_lookback = max(self.momentum_lookback * 2, 120)
        beta = stock_ret.rolling(beta_lookback, min_periods=max(40, beta_lookback // 3)).cov(bench_ret)
        beta = beta.divide(bench_ret.rolling(beta_lookback, min_periods=max(40, beta_lookback // 3)).var(), axis=0)
        momentum = close.pct_change(self.momentum_lookback, fill_method=None)
        beta_momo = beta * momentum

        trading_index = close.index.intersection(volume.index).intersection(foreign.index).intersection(inst.index).intersection(retail.index)
        close = close.reindex(trading_index)
        volume = volume.reindex(trading_index)
        eps = eps.reindex(trading_index).ffill()
        op = op.reindex(trading_index).ffill()
        foreign = foreign.reindex(trading_index)
        inst = inst.reindex(trading_index)
        retail = retail.reindex(trading_index)
        sector = sector.reindex(trading_index).ffill()
        beta_momo = beta_momo.reindex(trading_index)
        momentum = momentum.reindex(trading_index)

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

        revision_score = 0.5 * eps_rev.rank(axis=1, pct=True) + 0.5 * op_rev.rank(axis=1, pct=True)
        accel_score = 0.5 * eps_accel.rank(axis=1, pct=True) + 0.5 * op_accel.rank(axis=1, pct=True)
        composite = revision_score + accel_score

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
            beta_momo_row = beta_momo.loc[timestamp].reindex(score_row.index)
            momentum_row = momentum.loc[timestamp].reindex(score_row.index)

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

                beta_gate_frame = pd.concat(
                    [
                        sector_scores.reindex(long_candidates.index).rename("revision_score"),
                        beta_momo_row.reindex(long_candidates.index).rename("beta_momo"),
                        momentum_row.reindex(long_candidates.index).rename("momentum"),
                    ],
                    axis=1,
                ).dropna()
                if len(beta_gate_frame) < 4:
                    continue

                gate_cut = beta_gate_frame["beta_momo"].quantile(0.6)
                gated = beta_gate_frame[
                    beta_gate_frame["beta_momo"].ge(gate_cut) & beta_gate_frame["momentum"].ge(0.0)
                ]
                if len(gated) < 2:
                    continue

                long_cut = gated["revision_score"].quantile(0.65)
                longs = gated.index[gated["revision_score"].ge(long_cut)]

                short_cut = short_candidates.quantile(0.2)
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
                revision_short_z = self._zscore(oi_sector["revision_score"])
                squeeze_risk = 0.75 * oi_level_z + 0.75 * oi_imp_z + 0.25 * revision_short_z
                keep_mask = squeeze_risk <= 0.25
                filtered = oi_sector.index[keep_mask.reindex(oi_sector.index).fillna(False)]
                if len(filtered) == 0:
                    continue

                long_mask.loc[timestamp, longs] = True
                short_mask.loc[timestamp, filtered] = True
                alpha.loc[timestamp, longs] = sector_scores.reindex(longs).fillna(0.0).astype(float)
                alpha.loc[timestamp, filtered] = -composite.loc[timestamp, filtered].astype(float)
                conf = (-squeeze_risk.reindex(filtered).fillna(0.0) + 0.25).clip(lower=0.0) + revision_short_z.reindex(filtered).abs().fillna(0.0)
                short_confidence.loc[timestamp, filtered] = conf.astype(float)

        return SignalBundle(
            alpha=alpha,
            context={"sector": sector, "long_mask": long_mask, "short_mask": short_mask, "short_confidence": short_confidence},
            meta={},
        )
