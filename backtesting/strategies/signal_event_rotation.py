from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.data.benchmarks import benchmark_price_series
from backtesting.signals.base import SignalBundle
from backtesting.strategy.base import validate_positive

from .composable import ComposableStrategy
from .rrg_sector_rotation import (
    _build_op_rrg_state,
    _build_rrg_context,
    _compress_sector_op_leaders,
    _estimate_family_delta,
    _map_sector_state_to_symbols,
    _prior_based_revision,
)


SCORE_MODES = ("qavg", "op12", "blend", "accel", "eps_op")
EVENT_MODES = ("cross_up", "accel", "sector_turn", "new_high", "reclaim")
FLOW_GATES = ("none", "smart", "foreign", "inst", "retail_contra")
CONSTRUCTION_MODES = ("k1", "k2", "k3", "breadth")
RISK_MODES = ("lo", "ls02", "ls03", "ls05", "ls07")


@dataclass(slots=True)
class SignalEventRotation(ComposableStrategy):
    lookback: int = 20
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    rrg_transition_threshold: float = 0.005
    flow_lookback: int = 20
    high_lookback: int = 252
    participation_steps: int = 3
    score_mode: str = "qavg"
    event_mode: str = "cross_up"
    flow_gate: str = "smart"
    construction_mode: str = "k2"
    risk_mode: str = "ls03"
    gross_long: float = 1.0

    def __post_init__(self) -> None:
        validate_positive("lookback", self.lookback)
        validate_positive("flow_lookback", self.flow_lookback)
        validate_positive("high_lookback", self.high_lookback)
        validate_positive("participation_steps", self.participation_steps)
        _validate_choice("score_mode", self.score_mode, SCORE_MODES)
        _validate_choice("event_mode", self.event_mode, EVENT_MODES)
        _validate_choice("flow_gate", self.flow_gate, FLOW_GATES)
        _validate_choice("construction_mode", self.construction_mode, CONSTRUCTION_MODES)
        _validate_choice("risk_mode", self.risk_mode, RISK_MODES)
        if self.gross_long < 0.0:
            raise ValueError("gross_long must be non-negative")
        self.signal_producer = _SignalEventRotationSignal(
            lookback=self.lookback,
            rrg_medium_lookback=self.rrg_medium_lookback,
            rrg_momentum_lookback=self.rrg_momentum_lookback,
            rrg_short_lookback=self.rrg_short_lookback,
            rrg_transition_threshold=self.rrg_transition_threshold,
            flow_lookback=self.flow_lookback,
            high_lookback=self.high_lookback,
            participation_steps=self.participation_steps,
            score_mode=self.score_mode,
            event_mode=self.event_mode,
            flow_gate=self.flow_gate,
        )
        self.construction_rule = _SignalEventSectorCompressedWeight(
            gross_long=self.gross_long,
            gross_short=_gross_short_for_risk_mode(self.risk_mode),
            construction_mode=self.construction_mode,
        )


@dataclass(slots=True)
class SignalEventRotationSelected(SignalEventRotation):
    score_mode: str = "op12"
    event_mode: str = "accel"
    flow_gate: str = "retail_contra"
    construction_mode: str = "k2"
    risk_mode: str = "ls03"


@dataclass(slots=True)
class _SignalEventRotationSignal:
    lookback: int = 20
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    rrg_transition_threshold: float = 0.005
    flow_lookback: int = 20
    high_lookback: int = 252
    participation_steps: int = 3
    score_mode: str = "qavg"
    event_mode: str = "cross_up"
    flow_gate: str = "smart"

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
            DatasetId.QW_EPS_NFQ1,
            DatasetId.QW_EPS_NFQ2,
            DatasetId.QW_EPS_NFY1,
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
        market_cap_source = market.frames.get("float_market_cap", market.frames["market_cap"])
        market_cap = market_cap_source.reindex(index=close.index, columns=close.columns).ffill().astype(float)
        benchmark = benchmark_price_series(market.frames["benchmark"], "IKS200")
        benchmark = benchmark.reindex(close.index).ffill().astype(float)

        price_state, _long_sector, _short_sector = _build_rrg_context(
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
        op12 = market.frames["op_fwd"].reindex(index=close.index, columns=close.columns).ffill().astype(float)
        op_state = _build_op_rrg_state(
            op=op12,
            sector=sector,
            membership=k200,
            medium_lookback=self.rrg_medium_lookback,
            momentum_lookback=self.rrg_momentum_lookback,
            short_lookback=self.rrg_short_lookback,
            transition_threshold=self.rrg_transition_threshold,
        )

        score = _score_frame(
            frames=market.frames,
            index=close.index,
            columns=close.columns,
            sector=sector,
            lookback=self.lookback,
            mode=self.score_mode,
        )
        price_by_symbol = _map_sector_state_to_symbols(sector=sector, rrg_state=price_state)
        op_by_symbol = _map_sector_state_to_symbols(sector=sector, rrg_state=op_state)
        long_regime = price_by_symbol.isin(("Leading", "Improving", "Weakening")) & op_by_symbol.isin(("Leading", "Improving"))
        short_regime = price_by_symbol.eq("Lagging") & op_by_symbol.isin(("Lagging", "Weakening"))

        long_hold = long_regime & score.gt(0.0) & k200
        long_event = _event_mask(
            mode=self.event_mode,
            score=score,
            close=close,
            price_state=price_by_symbol,
            op_state=op_by_symbol,
            lookback=self.lookback,
            high_lookback=self.high_lookback,
        )
        long_event &= long_hold
        flow_ok, short_flow_ok = _flow_masks(
            market=market,
            like=close,
            market_cap=market_cap,
            lookback=self.flow_lookback,
            gate=self.flow_gate,
        )
        long_hold &= flow_ok
        long_event &= flow_ok
        participation = _event_participation(
            event=long_event.fillna(False).astype(bool),
            hold=long_hold.fillna(False).astype(bool),
            steps=self.participation_steps,
        )

        short_hold = short_regime & score.lt(0.0) & short_flow_ok & k200
        alpha = score.where(long_hold).mul(participation)
        short_alpha = score.mul(-1.0).where(short_hold)

        return SignalBundle(
            alpha=alpha,
            context={
                "tradable": k200,
                "entry_mask": long_event.fillna(False).astype(bool),
                "hold_mask": long_hold.fillna(False).astype(bool),
                "short_entry_mask": short_hold.fillna(False).astype(bool),
                "short_hold_mask": short_hold.fillna(False).astype(bool),
                "short_alpha": short_alpha,
                "sector": sector,
                "participation": participation,
            },
            meta={
                "score": score,
                "price_rrg_state": price_state,
                "op_rrg_state": op_state,
                "participation": participation,
            },
        )


@dataclass(slots=True)
class _SignalEventSectorCompressedWeight:
    gross_long: float = 1.0
    gross_short: float = 0.3
    construction_mode: str = "k2"

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        alpha = bundle.alpha.astype(float)
        short_alpha = bundle.context["short_alpha"].reindex(index=alpha.index, columns=alpha.columns).astype(float)
        weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns, dtype=float)
        for ts in alpha.index:
            long_scores = alpha.loc[ts].dropna()
            long_scores = long_scores[long_scores.gt(0.0)]
            if not long_scores.empty and self.gross_long > 0.0:
                long_weights = _rank_weights(long_scores, gross=self.gross_long)
                weights.loc[ts, long_weights.index] = long_weights

            short_scores = short_alpha.loc[ts].dropna()
            short_scores = short_scores[short_scores.gt(0.0)]
            short_scores = short_scores.loc[short_scores.index.difference(long_scores.index)]
            if not short_scores.empty and self.gross_short > 0.0:
                short_weights = _rank_weights(short_scores, gross=self.gross_short)
                weights.loc[ts, short_weights.index] = -short_weights

        sector = bundle.context["sector"].reindex(index=weights.index, columns=weights.columns).ffill()
        if self.construction_mode != "breadth":
            per_sector = {"k1": 1, "k2": 2, "k3": 3}[self.construction_mode]
            weights = _compress_sector_op_leaders(
                weights=weights,
                sector=sector,
                long_scores=alpha,
                short_scores=short_alpha,
                long_per_sector=per_sector,
                short_per_sector=1,
            )
        selected = weights.ne(0.0)
        return ConstructionResult(
            base_target_weights=weights,
            selection_mask=selected,
            group_long_budget=None,
            group_short_budget=None,
            meta={"selected": selected},
        )


def _score_frame(
    *,
    frames: dict[str, pd.DataFrame],
    index: pd.Index,
    columns: pd.Index,
    sector: pd.DataFrame,
    lookback: int,
    mode: str,
) -> pd.DataFrame:
    qavg, _q_count, _q_positive = _estimate_family_delta(
        frames=frames,
        keys=("op_fwd_q1", "op_fwd_q2", "op_fwd"),
        index=index,
        columns=columns,
        sector=sector,
        lookback=lookback,
    )
    op12 = _prior_based_revision(
        current=frames["op_fwd"].reindex(index=index, columns=columns).ffill().astype(float),
        prior=frames["op_fwd"].reindex(index=index, columns=columns).ffill().astype(float).shift(lookback),
    )
    eps, _eps_count, _eps_positive = _estimate_family_delta(
        frames=frames,
        keys=("eps_fwd_q1", "eps_fwd_q2", "eps_fwd"),
        index=index,
        columns=columns,
        sector=sector,
        lookback=lookback,
    )
    if mode == "qavg":
        return qavg
    if mode == "op12":
        return op12
    if mode == "blend":
        return qavg.add(op12, fill_value=0.0).divide(2.0)
    if mode == "accel":
        return qavg.sub(qavg.shift(max(1, lookback // 2)))
    if mode == "eps_op":
        same_sign = qavg.mul(eps).gt(0.0)
        return qavg.add(eps, fill_value=0.0).divide(2.0).where(same_sign)
    raise ValueError(f"score_mode must be one of {', '.join(SCORE_MODES)}")


def _event_mask(
    *,
    mode: str,
    score: pd.DataFrame,
    close: pd.DataFrame,
    price_state: pd.DataFrame,
    op_state: pd.DataFrame,
    lookback: int,
    high_lookback: int,
) -> pd.DataFrame:
    positive = score.gt(0.0)
    if mode == "cross_up":
        return positive & score.shift(lookback).le(0.0)
    if mode == "accel":
        return positive & score.diff(max(1, lookback // 2)).gt(0.0)
    if mode == "sector_turn":
        return positive & (price_state.eq("Improving") | op_state.eq("Improving"))
    if mode == "new_high":
        high = close.rolling(high_lookback, min_periods=max(20, high_lookback // 4)).max()
        return positive & close.ge(high)
    if mode == "reclaim":
        ma = close.rolling(60, min_periods=20).mean()
        return positive & close.gt(ma) & close.shift(5).le(ma.shift(5))
    raise ValueError(f"event_mode must be one of {', '.join(EVENT_MODES)}")


def _flow_masks(
    *,
    market: MarketData,
    like: pd.DataFrame,
    market_cap: pd.DataFrame,
    lookback: int,
    gate: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    foreign = market.frames["foreign_flow"].reindex(index=like.index, columns=like.columns).fillna(0.0).astype(float)
    inst = market.frames["inst_flow"].reindex(index=like.index, columns=like.columns).fillna(0.0).astype(float)
    retail = market.frames["retail_flow"].reindex(index=like.index, columns=like.columns).fillna(0.0).astype(float)
    scale = market_cap.where(market_cap.gt(0.0))
    smart = foreign.add(inst, fill_value=0.0).sub(retail, fill_value=0.0)
    flows = {
        "smart": smart,
        "foreign": foreign,
        "inst": inst,
        "retail_contra": retail.mul(-1.0),
    }
    if gate == "none":
        all_true = pd.DataFrame(True, index=like.index, columns=like.columns)
        return all_true, all_true
    if gate not in flows:
        raise ValueError(f"flow_gate must be one of {', '.join(FLOW_GATES)}")
    signal = flows[gate].rolling(lookback, min_periods=max(5, lookback // 2)).sum().divide(scale)
    return signal.gt(0.0), signal.lt(0.0)


def _event_participation(*, event: pd.DataFrame, hold: pd.DataFrame, steps: int) -> pd.DataFrame:
    participation = pd.DataFrame(0.0, index=event.index, columns=event.columns, dtype=float)
    age = pd.Series(0, index=event.columns, dtype=int)
    for ts in event.index:
        event_row = event.loc[ts].fillna(False).astype(bool)
        hold_row = hold.loc[ts].fillna(False).astype(bool)
        age.loc[~hold_row] = 0
        age.loc[event_row] = 1
        continuing = hold_row & ~event_row & age.gt(0)
        age.loc[continuing] = age.loc[continuing] + 1
        active = hold_row & age.gt(0)
        participation.loc[ts, active] = np.minimum(age.loc[active].astype(float) / float(steps), 1.0)
    return participation


def _rank_weights(scores: pd.Series, *, gross: float) -> pd.Series:
    ranks = scores.rank(method="first", ascending=True)
    total = float(ranks.sum())
    if total <= 0.0:
        return pd.Series(dtype=float)
    return ranks.divide(total).mul(float(gross)).astype(float)


def _gross_short_for_risk_mode(mode: str) -> float:
    mapping = {
        "lo": 0.0,
        "ls02": 0.2,
        "ls03": 0.3,
        "ls05": 0.5,
        "ls07": 0.7,
    }
    try:
        return mapping[mode]
    except KeyError as exc:
        raise ValueError(f"risk_mode must be one of {', '.join(RISK_MODES)}") from exc


def _validate_choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}")
