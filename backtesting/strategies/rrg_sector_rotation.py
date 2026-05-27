from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle
from backtesting.strategy.base import validate_positive

from .composable import ComposableStrategy


@dataclass(slots=True)
class RrgFwdFlow1LongShort(ComposableStrategy):
    lookback: int = 20
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    rrg_transition_threshold: float = 0.005
    gross_long: float = 1.0
    gross_short: float = 0.5

    def __post_init__(self) -> None:
        validate_positive("lookback", self.lookback)
        if self.gross_long < 0.0:
            raise ValueError("gross_long must be non-negative")
        if self.gross_short < 0.0:
            raise ValueError("gross_short must be non-negative")
        self.signal_producer = _RrgFwdFlow1Signal(
            lookback=self.lookback,
            rrg_medium_lookback=self.rrg_medium_lookback,
            rrg_momentum_lookback=self.rrg_momentum_lookback,
            rrg_short_lookback=self.rrg_short_lookback,
            rrg_transition_threshold=self.rrg_transition_threshold,
        )
        self.construction_rule = _RrgLongShortRankProportionalWeight(
            gross_long=self.gross_long,
            gross_short=self.gross_short,
        )


@dataclass(slots=True)
class _RrgLongShortRankProportionalWeight:
    gross_long: float = 1.0
    gross_short: float = 0.5

    def __post_init__(self) -> None:
        if self.gross_long < 0.0:
            raise ValueError("gross_long must be non-negative")
        if self.gross_short < 0.0:
            raise ValueError("gross_short must be non-negative")

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        long_alpha = bundle.alpha.astype(float)
        short_alpha = _required_frame(bundle, "short_alpha").reindex(index=long_alpha.index, columns=long_alpha.columns).astype(float)
        tradable = _optional_frame(bundle, "tradable", long_alpha.notna()).reindex(index=long_alpha.index, columns=long_alpha.columns)
        tradable = tradable.fillna(False).astype(bool)
        long_entry = _required_frame(bundle, "entry_mask").reindex(index=long_alpha.index, columns=long_alpha.columns)
        long_entry = long_entry.fillna(False).astype(bool)
        long_hold = _required_frame(bundle, "hold_mask").reindex(index=long_alpha.index, columns=long_alpha.columns)
        long_hold = long_hold.fillna(False).astype(bool)
        short_entry = _required_frame(bundle, "short_entry_mask").reindex(index=long_alpha.index, columns=long_alpha.columns)
        short_entry = short_entry.fillna(False).astype(bool)
        short_hold = _required_frame(bundle, "short_hold_mask").reindex(index=long_alpha.index, columns=long_alpha.columns)
        short_hold = short_hold.fillna(False).astype(bool)

        weights = pd.DataFrame(0.0, index=long_alpha.index, columns=long_alpha.columns, dtype=float)
        selected = pd.DataFrame(False, index=long_alpha.index, columns=long_alpha.columns, dtype=bool)
        previous_long = pd.Series(False, index=long_alpha.columns, dtype=bool)
        previous_short = pd.Series(False, index=long_alpha.columns, dtype=bool)

        for ts in long_alpha.index:
            long_candidates = long_entry.loc[ts] | (previous_long & long_hold.loc[ts])
            long_candidates = long_candidates & tradable.loc[ts] & long_alpha.loc[ts].notna() & long_alpha.loc[ts].gt(0.0)
            long_scores = long_alpha.loc[ts, long_candidates]
            long_ranked = long_scores.sort_values(ascending=False, kind="stable")
            if not long_scores.empty and self.gross_long > 0.0:
                long_weights = _proportional_rank_weights(long_scores, gross=float(self.gross_long))
                weights.loc[ts, long_weights.index] = long_weights
                selected.loc[ts, long_weights.index] = True

            short_candidates = short_entry.loc[ts] | (previous_short & short_hold.loc[ts])
            short_candidates = short_candidates & tradable.loc[ts] & short_alpha.loc[ts].notna() & short_alpha.loc[ts].gt(0.0)
            no_long_overlap = pd.Series(True, index=long_alpha.columns, dtype=bool)
            no_long_overlap.loc[long_ranked.index] = False
            short_candidates = short_candidates & no_long_overlap
            short_scores = short_alpha.loc[ts, short_candidates]
            if not short_scores.empty and self.gross_short > 0.0:
                short_weights = _proportional_rank_weights(short_scores, gross=float(self.gross_short))
                weights.loc[ts, short_weights.index] = -short_weights
                selected.loc[ts, short_weights.index] = True

            previous_long = weights.loc[ts].gt(0.0)
            previous_short = weights.loc[ts].lt(0.0)

        return ConstructionResult(
            base_target_weights=weights,
            selection_mask=selected,
            group_long_budget=None,
            group_short_budget=None,
            meta={
                "selected": selected,
            },
        )


def _proportional_rank_weights(scores: pd.Series, *, gross: float) -> pd.Series:
    ranks = scores.rank(method="first", ascending=True)
    total = float(ranks.sum())
    if total <= 0.0:
        return pd.Series(dtype=float)
    return ranks.divide(total).mul(gross).astype(float)


def _required_frame(bundle: SignalBundle, key: str) -> pd.DataFrame:
    frame = bundle.context.get(key)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"context[{key!r}] must be a DataFrame")
    return frame


def _optional_frame(bundle: SignalBundle, key: str, default: pd.DataFrame) -> pd.DataFrame:
    frame = bundle.context.get(key)
    if frame is None:
        return default
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"context[{key!r}] must be a DataFrame")
    return frame


@dataclass(slots=True)
class _RrgFwdFlow1Signal:
    lookback: int = 20
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    rrg_transition_threshold: float = 0.005

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_BM,
            DatasetId.QW_K200_YN,
            DatasetId.QW_WICS_SEC_BIG,
            DatasetId.QW_MKTCAP,
            DatasetId.QW_MKTCAP_FLT,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_OP_NFQ2,
            DatasetId.QW_OP_NFY1,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"].astype(float)
        k200 = market.frames["k200_yn"].reindex_like(close).fillna(False).astype(bool)
        active_columns = k200.columns[k200.any(axis=0)]
        if len(active_columns) > 0:
            close = close.loc[:, active_columns]
            k200 = k200.loc[:, active_columns]

        sector = market.frames["sector_big"].reindex(index=close.index, columns=close.columns).ffill()
        market_cap_source = market.frames.get("float_market_cap", market.frames["market_cap"])
        market_cap = market_cap_source.reindex(index=close.index, columns=close.columns).ffill().astype(float)
        benchmark_frame = market.frames["benchmark"]
        benchmark = benchmark_frame["IKS200"] if "IKS200" in benchmark_frame.columns else benchmark_frame.iloc[:, 0]
        benchmark = benchmark.reindex(close.index).ffill().astype(float)

        rrg_state, _long_sector, _short_sector = _build_rrg_context(
            close=close,
            benchmark=benchmark,
            sector=sector,
            membership=k200,
            market_cap=market_cap,
            medium_lookback=self.rrg_medium_lookback,
            momentum_lookback=self.rrg_momentum_lookback,
            short_lookback=self.rrg_short_lookback,
            transition_threshold=self.rrg_transition_threshold,
        )
        stock_op = _build_stock_op_revision(
            frames=market.frames,
            index=close.index,
            columns=close.columns,
            sector=sector,
            lookback=self.lookback,
        )
        sector_op = _build_sector_op_revision(
            stock_score=stock_op,
            sector=sector,
            membership=k200,
            weights=market_cap,
        )

        sector_confirm_by_symbol = _map_sector_values_to_symbols(
            sector=sector,
            sector_values=sector_op,
        )
        state_by_symbol = _map_sector_state_to_symbols(
            sector=sector.reindex(index=close.index, columns=close.columns),
            rrg_state=rrg_state.reindex(index=close.index),
        )
        confirmed = sector_confirm_by_symbol.gt(0.0) & stock_op.gt(0.0)
        entry_mask = state_by_symbol.isin(("Leading", "Improving", "Weakening")) & confirmed & k200
        hold_mask = state_by_symbol.isin(("Leading", "Improving", "Weakening")) & confirmed & k200
        short_confirmed = sector_confirm_by_symbol.lt(0.0) & stock_op.lt(0.0)
        short_entry_mask = state_by_symbol.eq("Lagging") & short_confirmed & k200
        short_hold_mask = state_by_symbol.eq("Lagging") & short_confirmed & k200

        candidate_score = stock_op.where(stock_op.gt(0.0))
        candidate_score = candidate_score.where((entry_mask | hold_mask) & k200)
        short_candidate_score = stock_op.mul(-1.0).where(stock_op.lt(0.0))
        short_candidate_score = short_candidate_score.where((short_entry_mask | short_hold_mask) & k200)

        return SignalBundle(
            alpha=candidate_score,
            context={
                "tradable": k200,
                "entry_mask": entry_mask.fillna(False).astype(bool),
                "hold_mask": hold_mask.fillna(False).astype(bool),
                "short_entry_mask": short_entry_mask.fillna(False).astype(bool),
                "short_hold_mask": short_hold_mask.fillna(False).astype(bool),
                "short_alpha": short_candidate_score,
                "sector": sector,
                "rrg_state": rrg_state,
                "sector_confirm_score": sector_confirm_by_symbol,
                "stock_confirm_score": stock_op,
                "stock_op_revision": stock_op,
            },
            meta={
                "rrg_state": rrg_state,
                "sector_op_revision": sector_op,
                "stock_op_revision": stock_op,
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
    transition_threshold: float = 0.0,
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

    state, long_sector, short_sector = _classify_rrg_states(
        relative_strength=relative_strength,
        momentum=momentum,
    )
    if transition_threshold > 0.0:
        state = _apply_rrg_transition_hysteresis(
            state=state,
            relative_strength=relative_strength,
            momentum=momentum,
            threshold=transition_threshold,
        )
        long_sector = state.isin(("Leading", "Improving"))
        short_sector = state.isin(("Lagging", "Weakening"))
    return state, long_sector, short_sector


def _apply_rrg_transition_hysteresis(
    *,
    state: pd.DataFrame,
    relative_strength: pd.DataFrame,
    momentum: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    smoothed = state.copy()
    previous = pd.Series("Unclassified", index=state.columns, dtype=object)
    for ts in state.index:
        row = state.loc[ts].copy()
        row_momentum = momentum.loc[ts].reindex(state.columns)
        row_strength = relative_strength.loc[ts].reindex(state.columns)
        weak_lagging_to_improving = previous.eq("Lagging") & row.eq("Improving") & row_momentum.le(threshold)
        weak_lagging_to_leading = previous.eq("Lagging") & row.eq("Leading") & row_strength.le(threshold)
        row.loc[weak_lagging_to_improving] = "Lagging"
        row.loc[weak_lagging_to_leading] = "Improving"
        smoothed.loc[ts] = row
        previous = row
    return smoothed


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


def _build_stock_op_revision(
    *,
    frames: dict[str, pd.DataFrame],
    index: pd.Index,
    columns: pd.Index,
    sector: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    op_delta, _op_count, _op_positive_count = _estimate_family_delta(
        frames=frames,
        keys=("op_fwd_q1", "op_fwd_q2", "op_fwd"),
        index=index,
        columns=columns,
        sector=sector,
        lookback=lookback,
    )
    return op_delta


def _build_sector_op_revision(
    *,
    stock_score: pd.DataFrame,
    sector: pd.DataFrame,
    membership: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    return _sector_weighted_signal(
        values=stock_score,
        sector=sector,
        membership=membership,
        weights=weights,
    )


def _sector_weighted_signal(
    *,
    values: pd.DataFrame,
    sector: pd.DataFrame,
    membership: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    rows: dict[pd.Timestamp, dict[object, float]] = {}
    aligned_sector = sector.reindex(index=values.index, columns=values.columns)
    aligned_membership = membership.reindex(index=values.index, columns=values.columns).fillna(False).astype(bool)
    aligned_weights = weights.reindex(index=values.index, columns=values.columns).ffill().astype(float)
    for ts in values.index:
        valid = aligned_membership.loc[ts] & values.loc[ts].notna() & aligned_sector.loc[ts].notna()
        row: dict[object, float] = {}
        for sector_name in pd.unique(aligned_sector.loc[ts, valid]):
            names = values.columns[valid & aligned_sector.loc[ts].eq(sector_name)]
            if len(names) == 0:
                continue
            sector_weights = aligned_weights.loc[ts, names].fillna(0.0).clip(lower=0.0)
            if float(sector_weights.sum()) <= 0.0:
                sector_weights = pd.Series(1.0, index=names, dtype=float)
            sector_weights = sector_weights / float(sector_weights.sum())
            row[sector_name] = float((values.loc[ts, names] * sector_weights).sum())
        rows[ts] = row
    return pd.DataFrame.from_dict(rows, orient="index").reindex(index=values.index)


def _map_sector_values_to_symbols(*, sector: pd.DataFrame, sector_values: pd.DataFrame) -> pd.DataFrame:
    values_by_symbol = pd.DataFrame(np.nan, index=sector.index, columns=sector.columns, dtype=float)
    for ts in sector.index:
        row_values = sector_values.loc[ts] if ts in sector_values.index else pd.Series(dtype=float)
        values_by_symbol.loc[ts] = sector.loc[ts].map(row_values)
    return values_by_symbol


def _estimate_family_delta(
    *,
    frames: dict[str, pd.DataFrame],
    keys: tuple[str, ...],
    index: pd.Index,
    columns: pd.Index,
    sector: pd.DataFrame,
    lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    deltas: list[pd.DataFrame] = []
    available: list[pd.DataFrame] = []
    positive: list[pd.DataFrame] = []
    for key in keys:
        estimate = frames[key].reindex(index=index, columns=columns).ffill().astype(float)
        delta = _prior_based_revision(current=estimate, prior=estimate.shift(lookback))
        deltas.append(delta)
        available.append(delta.notna().astype(float))
        positive.append(delta.gt(0.0).astype(float).where(delta.notna(), 0.0))

    delta_sum = sum(frame.fillna(0.0) for frame in deltas)
    count = sum(frame for frame in available)
    positive_count = sum(frame for frame in positive)
    average_delta = delta_sum.divide(count.replace(0.0, np.nan))
    return average_delta, count, positive_count


def _map_sector_state_to_symbols(*, sector: pd.DataFrame, rrg_state: pd.DataFrame) -> pd.DataFrame:
    state_by_symbol = pd.DataFrame(index=sector.index, columns=sector.columns, dtype=object)
    for ts in sector.index:
        row_state = rrg_state.loc[ts] if ts in rrg_state.index else pd.Series(dtype=object)
        state_by_symbol.loc[ts] = sector.loc[ts].map(row_state)
    return state_by_symbol


def _prior_based_revision(
    *,
    current: pd.DataFrame,
    prior: pd.DataFrame,
) -> pd.DataFrame:
    current = current.astype(float)
    prior = prior.reindex_like(current).astype(float)
    scale = prior.abs().replace(0.0, np.nan)
    delta = current.sub(prior).divide(scale)
    return delta.clip(lower=-1.0, upper=1.0)


def _classify_rrg_states(
    *,
    relative_strength: pd.DataFrame,
    momentum: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    leading = relative_strength.ge(0.0) & momentum.ge(0.0)
    improving = relative_strength.lt(0.0) & momentum.ge(0.0)
    lagging = relative_strength.lt(0.0) & momentum.lt(0.0)
    weakening = relative_strength.ge(0.0) & momentum.lt(0.0)

    valid = relative_strength.notna() & momentum.notna()
    states = pd.DataFrame("Unclassified", index=relative_strength.index, columns=relative_strength.columns, dtype=object)
    states = states.mask(leading, "Leading")
    states = states.mask(improving, "Improving")
    states = states.mask(lagging, "Lagging")
    states = states.mask(weakening, "Weakening")

    long_sector = (leading | improving) & valid
    short_sector = (lagging | weakening) & valid
    return states, long_sector.astype(bool), short_sector.astype(bool)
