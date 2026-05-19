from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.sector_rotation import SectorRotationLongShort
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle

from .composable import ComposableStrategy


@dataclass(slots=True)
class RrgSectorRotation(ComposableStrategy):
    top_n: int = 25
    bottom_n: int = 25
    lookback: int = 20
    flow_lookback: int = 20
    flow_impulse_lookback: int = 5
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    gross_long: float = 1.0
    gross_short: float = 1.0
    fwd_partial_confidence: float = 0.7
    weighting: str = "equal"

    def __post_init__(self) -> None:
        self.signal_producer = _RrgSectorRotationSignal(
            lookback=self.lookback,
            flow_lookback=self.flow_lookback,
            flow_impulse_lookback=self.flow_impulse_lookback,
            rrg_medium_lookback=self.rrg_medium_lookback,
            rrg_momentum_lookback=self.rrg_momentum_lookback,
            rrg_short_lookback=self.rrg_short_lookback,
            fwd_partial_confidence=self.fwd_partial_confidence,
        )
        self.construction_rule = SectorRotationLongShort(
            long_count=self.top_n,
            short_count=self.bottom_n,
            gross_long=self.gross_long,
            gross_short=self.gross_short,
            weighting=self.weighting,
        )


@dataclass(slots=True)
class _RrgSectorRotationSignal:
    lookback: int = 20
    flow_lookback: int = 20
    flow_impulse_lookback: int = 5
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    fwd_partial_confidence: float = 0.7

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_BM,
            DatasetId.QW_K200_YN,
            DatasetId.QW_WICS_SEC_BIG,
            DatasetId.QW_MKTCAP,
            DatasetId.QW_V,
            DatasetId.QW_EPS_NFQ1,
            DatasetId.QW_EPS_NFQ2,
            DatasetId.QW_EPS_NFY1,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_OP_NFQ2,
            DatasetId.QW_OP_NFY1,
            DatasetId.QW_FOREIGN,
            DatasetId.QW_INSTITUTION,
            DatasetId.QW_RETAIL,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"].astype(float)
        k200 = market.frames["k200_yn"].reindex_like(close).fillna(False).astype(bool)
        active_columns = k200.columns[k200.any(axis=0)]
        if len(active_columns) > 0:
            close = close.loc[:, active_columns]
            k200 = k200.loc[:, active_columns]

        sector = market.frames["sector_big"].reindex(index=close.index, columns=close.columns).ffill()
        market_cap = market.frames["market_cap"].reindex(index=close.index, columns=close.columns).ffill().astype(float)
        benchmark_frame = market.frames["benchmark"]
        benchmark = benchmark_frame["IKS200"] if "IKS200" in benchmark_frame.columns else benchmark_frame.iloc[:, 0]
        benchmark = benchmark.reindex(close.index).ffill().astype(float)

        rrg_state, long_sector, short_sector = _build_rrg_context(
            close=close,
            benchmark=benchmark,
            sector=sector,
            membership=k200,
            market_cap=market_cap,
            medium_lookback=self.rrg_medium_lookback,
            momentum_lookback=self.rrg_momentum_lookback,
            short_lookback=self.rrg_short_lookback,
        )
        fwd_score, fwd_confidence, fwd_coverage = _build_forward_score(
            frames=market.frames,
            index=close.index,
            columns=close.columns,
            sector=sector,
            lookback=self.lookback,
            partial_confidence=self.fwd_partial_confidence,
        )
        flow_score_20d, flow_score_5d = _build_flow_scores(
            frames=market.frames,
            close=close,
            sector=sector,
            flow_lookback=self.flow_lookback,
            impulse_lookback=self.flow_impulse_lookback,
        )

        alpha = (0.5 * fwd_score.mul(fwd_confidence) + 0.5 * flow_score_20d).where(k200 & fwd_score.notna())
        tradable = k200 & fwd_score.notna()

        return SignalBundle(
            alpha=alpha,
            context={
                "tradable": tradable,
                "sector": sector,
                "long_sector": long_sector,
                "short_sector": short_sector,
                "sector_weight_basis": market_cap.where(k200),
            },
            meta={
                "rrg_state": rrg_state,
                "fwd_score": fwd_score,
                "fwd_confidence": fwd_confidence,
                "fwd_coverage": fwd_coverage,
                "flow_score_20d": flow_score_20d,
                "flow_score_5d": flow_score_5d,
            },
        )


def _build_rrg_context(
    *,
    close: pd.DataFrame,
    benchmark: pd.Series,
    sector: pd.DataFrame,
    membership: pd.DataFrame,
    market_cap: pd.DataFrame,
    medium_lookback: int,
    momentum_lookback: int,
    short_lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = close.pct_change(fill_method=None)
    benchmark_returns = benchmark.pct_change(fill_method=None)
    sector_returns = _sector_weighted_returns(
        returns=returns,
        sector=sector,
        membership=membership,
        weights=market_cap,
    )
    sector_index = (1.0 + sector_returns.fillna(0.0)).cumprod()
    benchmark_index = (1.0 + benchmark_returns.fillna(0.0)).cumprod()
    relative = sector_index.divide(benchmark_index, axis=0)

    medium_mean = relative.rolling(medium_lookback, min_periods=max(5, medium_lookback // 3)).mean()
    short_mean = relative.rolling(short_lookback, min_periods=max(5, short_lookback // 3)).mean()
    relative_strength = relative.divide(medium_mean.replace(0.0, np.nan)) - 1.0
    short_relative = relative.divide(short_mean.replace(0.0, np.nan)) - 1.0
    momentum = short_relative - short_relative.shift(momentum_lookback)

    return _classify_rrg_states(
        relative_strength=relative_strength.fillna(0.0),
        momentum=momentum.fillna(0.0),
    )


def _sector_weighted_returns(
    *,
    returns: pd.DataFrame,
    sector: pd.DataFrame,
    membership: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    rows: dict[pd.Timestamp, dict[object, float]] = {}
    for ts in returns.index:
        valid = membership.loc[ts].astype(bool) & returns.loc[ts].notna() & sector.loc[ts].notna()
        row: dict[object, float] = {}
        for sector_name in pd.unique(sector.loc[ts, valid]):
            names = returns.columns[valid & sector.loc[ts].eq(sector_name)]
            sector_weights = weights.loc[ts, names].fillna(0.0).clip(lower=0.0)
            if float(sector_weights.sum()) <= 0.0:
                sector_weights = pd.Series(1.0, index=names, dtype=float)
            sector_weights = sector_weights / float(sector_weights.sum())
            row[sector_name] = float((returns.loc[ts, names] * sector_weights).sum())
        rows[ts] = row
    return pd.DataFrame.from_dict(rows, orient="index").reindex(index=returns.index)


def _build_forward_score(
    *,
    frames: dict[str, pd.DataFrame],
    index: pd.Index,
    columns: pd.Index,
    sector: pd.DataFrame,
    lookback: int,
    partial_confidence: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eps_score, eps_count = _estimate_family_score(
        frames=frames,
        keys=("eps_fwd_q1", "eps_fwd_q2", "eps_fwd"),
        index=index,
        columns=columns,
        sector=sector,
        lookback=lookback,
    )
    op_score, op_count = _estimate_family_score(
        frames=frames,
        keys=("op_fwd_q1", "op_fwd_q2", "op_fwd"),
        index=index,
        columns=columns,
        sector=sector,
        lookback=lookback,
    )

    family_count = eps_score.notna().astype(int) + op_score.notna().astype(int)
    score = (eps_score.fillna(0.0) + op_score.fillna(0.0)).divide(family_count.replace(0, np.nan))
    confidence = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    confidence = confidence.mask(family_count.eq(1), partial_confidence)
    confidence = confidence.mask(family_count.ge(2), 1.0)
    coverage = (eps_count.fillna(0.0) + op_count.fillna(0.0)).astype(float)
    return score, confidence, coverage


def _estimate_family_score(
    *,
    frames: dict[str, pd.DataFrame],
    keys: tuple[str, ...],
    index: pd.Index,
    columns: pd.Index,
    sector: pd.DataFrame,
    lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores: list[pd.DataFrame] = []
    available: list[pd.DataFrame] = []
    for key in keys:
        estimate = frames[key].reindex(index=index, columns=columns).ffill().astype(float)
        delta = _bounded_delta(current=estimate, prior=estimate.shift(lookback), sector=sector)
        ranked = _sector_rank(delta, sector=sector, ascending=True)
        scores.append(ranked)
        available.append(ranked.notna().astype(float))

    score_sum = sum(frame.fillna(0.0) for frame in scores)
    count = sum(frame for frame in available)
    score = score_sum.divide(count.replace(0.0, np.nan))
    return score, count


def _build_flow_scores(
    *,
    frames: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    sector: pd.DataFrame,
    flow_lookback: int,
    impulse_lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    foreign = frames["foreign_flow"].reindex_like(close).fillna(0.0).astype(float)
    inst = frames["inst_flow"].reindex_like(close).fillna(0.0).astype(float)
    retail = frames["retail_flow"].reindex_like(close).fillna(0.0).astype(float)
    volume = frames["volume"].reindex_like(close).fillna(0.0).astype(float)
    trading_value = close.mul(volume).replace(0.0, np.nan)
    flow_pressure = foreign.add(inst, fill_value=0.0).sub(retail, fill_value=0.0).divide(trading_value)
    flow_mean_20d = flow_pressure.rolling(flow_lookback, min_periods=max(2, flow_lookback // 2)).mean()
    flow_mean_5d = flow_pressure.rolling(impulse_lookback, min_periods=max(2, impulse_lookback // 2)).mean()
    flow_score_20d = _sector_rank(_rolling_zscore(flow_mean_20d, flow_lookback), sector=sector, ascending=True)
    flow_score_5d = _sector_rank(_rolling_zscore(flow_mean_5d, impulse_lookback), sector=sector, ascending=True)
    return flow_score_20d, flow_score_5d


def _rolling_zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=max(2, window // 2)).mean()
    std = frame.rolling(window, min_periods=max(2, window // 2)).std(ddof=0)
    return frame.sub(mean).divide(std.replace(0.0, np.nan))


def _bounded_delta(
    *,
    current: pd.DataFrame,
    prior: pd.DataFrame,
    sector: pd.DataFrame | None = None,
) -> pd.DataFrame:
    current = current.astype(float)
    prior = prior.reindex_like(current).astype(float)
    scale = current.abs().combine(prior.abs(), np.maximum)
    if sector is not None:
        sector_scale = _sector_abs_estimate_scale(current=current, prior=prior, sector=sector)
        scale = scale.combine(sector_scale, np.maximum)
    scale = scale.replace(0.0, np.nan)
    delta = current.sub(prior).divide(scale)
    return delta.clip(lower=-1.0, upper=1.0)


def _sector_abs_estimate_scale(
    *,
    current: pd.DataFrame,
    prior: pd.DataFrame,
    sector: pd.DataFrame,
) -> pd.DataFrame:
    scale = pd.DataFrame(np.nan, index=current.index, columns=current.columns, dtype=float)
    aligned_sector = sector.reindex(index=current.index, columns=current.columns)
    absolute_estimate = current.abs().combine(prior.abs(), np.maximum)
    for ts in current.index:
        row_sector = aligned_sector.loc[ts]
        row_abs = absolute_estimate.loc[ts]
        for sector_name in pd.unique(row_sector.dropna()):
            members = row_sector[row_sector.eq(sector_name)].index
            member_abs = row_abs.reindex(members)
            if not bool(member_abs.notna().any()):
                continue
            median_abs = member_abs.median(skipna=True)
            if pd.notna(median_abs):
                scale.loc[ts, members] = float(median_abs)
    return scale


def _sector_rank(values: pd.DataFrame, *, sector: pd.DataFrame, ascending: bool) -> pd.DataFrame:
    result = pd.DataFrame(np.nan, index=values.index, columns=values.columns, dtype=float)
    aligned_sector = sector.reindex(index=values.index, columns=values.columns)
    for ts in values.index:
        row_values = values.loc[ts]
        row_sector = aligned_sector.loc[ts]
        for sector_name in pd.unique(row_sector.dropna()):
            members = row_sector[row_sector.eq(sector_name)].index
            member_values = row_values.reindex(members).dropna()
            if member_values.empty:
                continue
            result.loc[ts, member_values.index] = member_values.rank(
                ascending=ascending,
                pct=True,
                method="average",
            )
    return result


def _classify_rrg_states(
    *,
    relative_strength: pd.DataFrame,
    momentum: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    leading = relative_strength.ge(0.0) & momentum.ge(0.0)
    improving = relative_strength.lt(0.0) & momentum.ge(0.0)
    lagging = relative_strength.lt(0.0) & momentum.lt(0.0)
    weakening = relative_strength.ge(0.0) & momentum.lt(0.0)

    states = pd.DataFrame("Weakening", index=relative_strength.index, columns=relative_strength.columns, dtype=object)
    states = states.mask(leading, "Leading")
    states = states.mask(improving, "Improving")
    states = states.mask(lagging, "Lagging")

    long_sector = leading | improving
    short_sector = lagging | weakening
    return states, long_sector.astype(bool), short_sector.astype(bool)
