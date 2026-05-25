from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.policy.base import BUCKET_LEDGER_COLUMNS, PositionPlan
from backtesting.signals.base import SignalBundle
from backtesting.strategy.base import validate_positive

from .base import RegisteredStrategy
from .composable import ComposableStrategy


_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class RrgFwdFlow1LongShort(ComposableStrategy):
    max_long_names: int = 10
    max_short_names: int = 10
    lookback: int = 20
    flow_lookback: int = 20
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    rrg_transition_threshold: float = 0.005
    gross_long: float = 1.0
    gross_short: float = 0.5

    def __post_init__(self) -> None:
        validate_positive("max_long_names", self.max_long_names)
        validate_positive("max_short_names", self.max_short_names)
        validate_positive("lookback", self.lookback)
        validate_positive("flow_lookback", self.flow_lookback)
        if self.gross_long < 0.0:
            raise ValueError("gross_long must be non-negative")
        if self.gross_short < 0.0:
            raise ValueError("gross_short must be non-negative")
        self.signal_producer = _RrgFwdFlow1Signal(
            lookback=self.lookback,
            flow_lookback=self.flow_lookback,
            rrg_medium_lookback=self.rrg_medium_lookback,
            rrg_momentum_lookback=self.rrg_momentum_lookback,
            rrg_short_lookback=self.rrg_short_lookback,
            rrg_transition_threshold=self.rrg_transition_threshold,
        )
        self.construction_rule = _RrgConcentratedLongShortEqualWeight(
            max_long_names=self.max_long_names,
            max_short_names=self.max_short_names,
            gross_long=self.gross_long,
            gross_short=self.gross_short,
        )


@dataclass(frozen=True, slots=True)
class _RrgSavedWeightsStrategy(RegisteredStrategy):
    _RUN_DIR_NAME: ClassVar[str]

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (DatasetId.QW_ADJ_C,)

    @property
    def run_dir(self) -> Path:
        return _REPO_ROOT / "results" / "backtests" / self._RUN_DIR_NAME

    def build_signal(self, market: MarketData) -> pd.DataFrame:
        return self._saved_weights(market)

    def target_weights(self, signal: pd.Series) -> pd.Series:
        return signal.fillna(0.0).astype(float)

    def build_plan(self, market: MarketData) -> PositionPlan:
        weights = self._saved_weights(market)
        return PositionPlan(
            target_weights=weights,
            bucket_ledger=self._saved_bucket_ledger(),
            bucket_meta={},
            validation={
                "archived_strategy": True,
                "source_run": str(self.run_dir),
            },
        )

    def _saved_weights(self, market: MarketData) -> pd.DataFrame:
        close = market.frames["close"]
        path = self.run_dir / "positions" / "weights.parquet"
        if not path.exists():
            raise FileNotFoundError(f"saved RRG weights not found: {path}")
        weights = pd.read_parquet(path).astype(float)
        return weights.reindex(index=close.index, columns=close.columns).fillna(0.0).astype(float)

    def _saved_bucket_ledger(self) -> pd.DataFrame:
        path = self.run_dir / "positions" / "bucket_ledger.parquet"
        if not path.exists():
            return pd.DataFrame(columns=BUCKET_LEDGER_COLUMNS)
        ledger = pd.read_parquet(path)
        return ledger.reindex(columns=BUCKET_LEDGER_COLUMNS)


class RrgFwdFlow1LsGs05ListedExitValidated(_RrgSavedWeightsStrategy):
    _RUN_DIR_NAME = "rrg-fwd-flow1-ls-gs0.5-listed-exit-validated_20260525_123450"


class RrgFwdFlow1Ls03Change10EtfShortoffResearch(_RrgSavedWeightsStrategy):
    _RUN_DIR_NAME = "rrg-fwd-flow1-ls03-change10-etf-shortoff-research_20260525_122853"


class RrgFwdFlow1LsLag31MonthlyGs00L5Validated(_RrgSavedWeightsStrategy):
    _RUN_DIR_NAME = "rrg-fwd-flow1-ls-lag31-monthly-gs0.0-l5-validated_20260525_134348"


@dataclass(slots=True)
class _RrgConcentratedLongShortEqualWeight:
    max_long_names: int
    max_short_names: int
    gross_long: float = 1.0
    gross_short: float = 0.5

    def __post_init__(self) -> None:
        validate_positive("max_long_names", self.max_long_names)
        validate_positive("max_short_names", self.max_short_names)
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
            long_ranked = long_alpha.loc[ts, long_candidates].sort_values(ascending=False, kind="stable").head(self.max_long_names)
            if not long_ranked.empty and self.gross_long > 0.0:
                weights.loc[ts, long_ranked.index] = float(self.gross_long) / len(long_ranked)
                selected.loc[ts, long_ranked.index] = True

            short_candidates = short_entry.loc[ts] | (previous_short & short_hold.loc[ts])
            short_candidates = short_candidates & tradable.loc[ts] & short_alpha.loc[ts].notna() & short_alpha.loc[ts].gt(0.0)
            no_long_overlap = pd.Series(True, index=long_alpha.columns, dtype=bool)
            no_long_overlap.loc[long_ranked.index] = False
            short_candidates = short_candidates & no_long_overlap
            short_ranked = short_alpha.loc[ts, short_candidates].sort_values(ascending=False, kind="stable").head(self.max_short_names)
            if not short_ranked.empty and self.gross_short > 0.0:
                weights.loc[ts, short_ranked.index] = -float(self.gross_short) / len(short_ranked)
                selected.loc[ts, short_ranked.index] = True

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
    flow_lookback: int = 20
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
            DatasetId.QW_V,
            DatasetId.QW_FOREIGN,
            DatasetId.QW_INSTITUTION,
            DatasetId.QW_RETAIL,
            DatasetId.QW_EPS_NFQ1,
            DatasetId.QW_EPS_NFQ2,
            DatasetId.QW_EPS_NFY1,
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
        stock_consensus = _build_stock_consensus_confirmation(
            frames=market.frames,
            index=close.index,
            columns=close.columns,
            sector=sector,
            lookback=self.lookback,
        )
        sector_consensus = _build_sector_consensus_confirmation(
            stock_score=stock_consensus,
            sector=sector,
            membership=k200,
            weights=market_cap,
        )
        stock_flow = _build_stock_flow_confirmation(
            frames=market.frames,
            close=close,
            flow_lookback=self.flow_lookback,
        )
        sector_flow = _build_sector_flow_confirmation(
            stock_score=stock_flow,
            sector=sector,
            membership=k200,
            weights=market_cap,
        )

        stock_confirm = stock_consensus.where(stock_consensus.notna(), stock_flow)
        sector_confirm = sector_consensus.where(sector_consensus.notna(), sector_flow)
        sector_confirm_by_symbol = _map_sector_values_to_symbols(
            sector=sector,
            sector_values=sector_confirm,
        )
        state_by_symbol = _map_sector_state_to_symbols(
            sector=sector.reindex(index=close.index, columns=close.columns),
            rrg_state=rrg_state.reindex(index=close.index),
        )
        confirmed = sector_confirm_by_symbol.gt(0.0) & stock_confirm.gt(0.0)
        entry_mask = state_by_symbol.isin(("Leading", "Improving")) & confirmed & k200
        hold_mask = state_by_symbol.isin(("Leading", "Improving", "Weakening")) & confirmed & k200
        short_confirmed = sector_confirm_by_symbol.lt(0.0) & stock_confirm.lt(0.0)
        short_entry_mask = state_by_symbol.isin(("Lagging", "Weakening")) & short_confirmed & k200
        short_hold_mask = state_by_symbol.isin(("Lagging", "Weakening")) & short_confirmed & k200

        sector_rank = sector_confirm.rank(axis=1, ascending=True, pct=True)
        sector_rank_by_symbol = _map_sector_values_to_symbols(
            sector=sector,
            sector_values=sector_rank,
        )
        stock_rank = stock_confirm.rank(axis=1, ascending=True, pct=True)
        candidate_score = sector_rank_by_symbol.combine(stock_rank, np.minimum)
        candidate_score = candidate_score.where((entry_mask | hold_mask) & k200)
        short_sector_rank = sector_confirm.mul(-1.0).rank(axis=1, ascending=True, pct=True)
        short_sector_rank_by_symbol = _map_sector_values_to_symbols(
            sector=sector,
            sector_values=short_sector_rank,
        )
        short_stock_rank = stock_confirm.mul(-1.0).rank(axis=1, ascending=True, pct=True)
        short_candidate_score = short_sector_rank_by_symbol.combine(short_stock_rank, np.minimum)
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
                "stock_confirm_score": stock_confirm,
                "stock_consensus_score": stock_consensus,
                "stock_flow_score": stock_flow,
            },
            meta={
                "rrg_state": rrg_state,
                "sector_consensus_score": sector_consensus,
                "sector_flow_score": sector_flow,
                "stock_consensus_score": stock_consensus,
                "stock_flow_score": stock_flow,
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


def _build_stock_consensus_confirmation(
    *,
    frames: dict[str, pd.DataFrame],
    index: pd.Index,
    columns: pd.Index,
    sector: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    eps_delta, _eps_count, _eps_positive_count = _estimate_family_delta(
        frames=frames,
        keys=("eps_fwd_q1", "eps_fwd_q2", "eps_fwd"),
        index=index,
        columns=columns,
        sector=sector,
        lookback=lookback,
    )
    op_delta, _op_count, _op_positive_count = _estimate_family_delta(
        frames=frames,
        keys=("op_fwd_q1", "op_fwd_q2", "op_fwd"),
        index=index,
        columns=columns,
        sector=sector,
        lookback=lookback,
    )
    return eps_delta.mul(0.5).add(op_delta.mul(0.5), fill_value=np.nan).where(eps_delta.notna() & op_delta.notna())


def _build_stock_flow_confirmation(
    *,
    frames: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    flow_lookback: int,
) -> pd.DataFrame:
    foreign = frames["foreign_flow"].reindex_like(close).fillna(0.0).astype(float)
    inst = frames["inst_flow"].reindex_like(close).fillna(0.0).astype(float)
    retail = frames["retail_flow"].reindex_like(close).fillna(0.0).astype(float)
    volume = frames["volume"].reindex_like(close).fillna(0.0).astype(float)
    trading_value = close.mul(volume).replace(0.0, np.nan)
    flow_pressure = foreign.add(inst, fill_value=0.0).sub(retail, fill_value=0.0).divide(trading_value)
    flow_mean = flow_pressure.rolling(flow_lookback, min_periods=max(2, flow_lookback // 2)).mean()
    return _rolling_zscore(flow_mean, flow_lookback)


def _build_sector_consensus_confirmation(
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


def _build_sector_flow_confirmation(
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
        delta = _bounded_delta(current=estimate, prior=estimate.shift(lookback), sector=sector)
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
