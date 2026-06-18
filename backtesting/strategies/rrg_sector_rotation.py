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


@dataclass(slots=True)
class RrgSectorRotation(ComposableStrategy):
    lookback: int = 20
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    rrg_transition_threshold: float = 0.005
    gross_long: float = 1.0
    gross_short: float = 0.5
    long_quantile: float | None = None
    short_quantile: float | None = None
    min_long_revision: float = 0.0
    min_short_revision: float = 0.0
    max_long_names: int | None = None
    max_short_names: int | None = None

    def __post_init__(self) -> None:
        validate_positive("lookback", self.lookback)
        if self.gross_long < 0.0:
            raise ValueError("gross_long must be non-negative")
        if self.gross_short < 0.0:
            raise ValueError("gross_short must be non-negative")
        _validate_optional_quantile("long_quantile", self.long_quantile)
        _validate_optional_quantile("short_quantile", self.short_quantile)
        _validate_non_negative("min_long_revision", self.min_long_revision)
        _validate_non_negative("min_short_revision", self.min_short_revision)
        _validate_optional_positive_int("max_long_names", self.max_long_names)
        _validate_optional_positive_int("max_short_names", self.max_short_names)
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
            long_quantile=self.long_quantile,
            short_quantile=self.short_quantile,
            min_long_revision=self.min_long_revision,
            min_short_revision=self.min_short_revision,
            max_long_names=self.max_long_names,
            max_short_names=self.max_short_names,
        )


@dataclass(slots=True)
class RrgSectorRotationPrune90(ComposableStrategy):
    lookback: int = 20
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    rrg_transition_threshold: float = 0.005
    gross_long: float = 1.0
    gross_short: float = 0.5
    coverage: float = 0.90
    max_total_names: int = 20

    def __post_init__(self) -> None:
        validate_positive("lookback", self.lookback)
        if self.gross_long < 0.0:
            raise ValueError("gross_long must be non-negative")
        if self.gross_short < 0.0:
            raise ValueError("gross_short must be non-negative")
        _validate_coverage("coverage", self.coverage)
        _validate_optional_positive_int("max_total_names", self.max_total_names)
        self.signal_producer = _RrgFwdFlow1Signal(
            lookback=self.lookback,
            rrg_medium_lookback=self.rrg_medium_lookback,
            rrg_momentum_lookback=self.rrg_momentum_lookback,
            rrg_short_lookback=self.rrg_short_lookback,
            rrg_transition_threshold=self.rrg_transition_threshold,
        )
        self.construction_rule = _RrgK2OpPrune90SectorPreserveWeight(
            gross_long=self.gross_long,
            gross_short=self.gross_short,
            coverage=self.coverage,
            max_total_names=self.max_total_names,
        )


@dataclass(slots=True)
class RrgSectorRotationOpRrg(ComposableStrategy):
    lookback: int = 20
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    rrg_transition_threshold: float = 0.005
    op_rrg_medium_lookback: int = 126
    op_rrg_momentum_lookback: int = 21
    op_rrg_short_lookback: int = 42
    op_rrg_transition_threshold: float = 0.005
    gross_long: float = 1.0
    gross_short: float = 0.5
    long_price_states: tuple[str, ...] = ("Leading", "Improving", "Weakening")
    long_op_states: tuple[str, ...] = ("Leading", "Improving")
    short_price_states: tuple[str, ...] = ("Lagging",)
    short_op_states: tuple[str, ...] = ("Lagging", "Weakening")
    long_per_sector: int = 2
    short_per_sector: int = 1
    op_rrg_exclude_bm_weight_gt: float | None = None

    def __post_init__(self) -> None:
        validate_positive("lookback", self.lookback)
        if self.gross_long < 0.0:
            raise ValueError("gross_long must be non-negative")
        if self.gross_short < 0.0:
            raise ValueError("gross_short must be non-negative")
        _validate_state_names("long_price_states", self.long_price_states)
        _validate_state_names("long_op_states", self.long_op_states)
        _validate_state_names("short_price_states", self.short_price_states)
        _validate_state_names("short_op_states", self.short_op_states)
        _validate_optional_positive_int("long_per_sector", self.long_per_sector)
        _validate_optional_positive_int("short_per_sector", self.short_per_sector)
        _validate_optional_quantile("op_rrg_exclude_bm_weight_gt", self.op_rrg_exclude_bm_weight_gt)
        self.signal_producer = _RrgOpRrgSignal(
            lookback=self.lookback,
            rrg_medium_lookback=self.rrg_medium_lookback,
            rrg_momentum_lookback=self.rrg_momentum_lookback,
            rrg_short_lookback=self.rrg_short_lookback,
            rrg_transition_threshold=self.rrg_transition_threshold,
            op_rrg_medium_lookback=self.op_rrg_medium_lookback,
            op_rrg_momentum_lookback=self.op_rrg_momentum_lookback,
            op_rrg_short_lookback=self.op_rrg_short_lookback,
            op_rrg_transition_threshold=self.op_rrg_transition_threshold,
            long_price_states=self.long_price_states,
            long_op_states=self.long_op_states,
            short_price_states=self.short_price_states,
            short_op_states=self.short_op_states,
            op_rrg_exclude_bm_weight_gt=self.op_rrg_exclude_bm_weight_gt,
        )
        self.construction_rule = _RrgOpRrgSectorCompressedWeight(
            gross_long=self.gross_long,
            gross_short=self.gross_short,
            long_per_sector=self.long_per_sector,
            short_per_sector=self.short_per_sector,
        )


@dataclass(slots=True)
class RrgSectorRotationOpRrgK2(RrgSectorRotationOpRrg):
    long_per_sector: int = 2
    short_per_sector: int = 1


@dataclass(slots=True)
class RrgSectorRotationOpRrgK1(RrgSectorRotationOpRrg):
    long_per_sector: int = 1
    short_per_sector: int = 1


@dataclass(slots=True)
class RrgSectorRotationOpRrgEx10K2(RrgSectorRotationOpRrg):
    long_per_sector: int = 2
    short_per_sector: int = 1
    op_rrg_exclude_bm_weight_gt: float | None = 0.10


@dataclass(slots=True)
class RrgSectorRotationOpRrgEx10K1(RrgSectorRotationOpRrg):
    long_per_sector: int = 1
    short_per_sector: int = 1
    op_rrg_exclude_bm_weight_gt: float | None = 0.10


@dataclass(slots=True)
class _RrgLongShortRankProportionalWeight:
    gross_long: float = 1.0
    gross_short: float = 0.5
    long_quantile: float | None = None
    short_quantile: float | None = None
    min_long_revision: float = 0.0
    min_short_revision: float = 0.0
    max_long_names: int | None = None
    max_short_names: int | None = None

    def __post_init__(self) -> None:
        if self.gross_long < 0.0:
            raise ValueError("gross_long must be non-negative")
        if self.gross_short < 0.0:
            raise ValueError("gross_short must be non-negative")
        _validate_optional_quantile("long_quantile", self.long_quantile)
        _validate_optional_quantile("short_quantile", self.short_quantile)
        _validate_non_negative("min_long_revision", self.min_long_revision)
        _validate_non_negative("min_short_revision", self.min_short_revision)
        _validate_optional_positive_int("max_long_names", self.max_long_names)
        _validate_optional_positive_int("max_short_names", self.max_short_names)

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
            long_scores = _filter_candidate_scores(
                long_scores,
                min_score=self.min_long_revision,
                quantile=self.long_quantile,
                max_names=self.max_long_names,
            )
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
            short_scores = _filter_candidate_scores(
                short_scores,
                min_score=self.min_short_revision,
                quantile=self.short_quantile,
                max_names=self.max_short_names,
            )
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


@dataclass(slots=True)
class _RrgPrune90SectorPreserveWeight:
    gross_long: float = 1.0
    gross_short: float = 0.5
    coverage: float = 0.90
    max_total_names: int = 20

    def __post_init__(self) -> None:
        if self.gross_long < 0.0:
            raise ValueError("gross_long must be non-negative")
        if self.gross_short < 0.0:
            raise ValueError("gross_short must be non-negative")
        _validate_coverage("coverage", self.coverage)
        _validate_optional_positive_int("max_total_names", self.max_total_names)

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        raw = _RrgLongShortRankProportionalWeight(
            gross_long=self.gross_long,
            gross_short=self.gross_short,
        ).build(bundle)
        sector = _required_frame(bundle, "sector").reindex(
            index=raw.base_target_weights.index,
            columns=raw.base_target_weights.columns,
        ).ffill()
        weights = _prune_sector_preserving_weights(
            weights=raw.base_target_weights,
            sector=sector,
            coverage=self.coverage,
            max_total_names=self.max_total_names,
        )
        selected = weights.ne(0.0)
        return ConstructionResult(
            base_target_weights=weights,
            selection_mask=selected,
            group_long_budget=raw.group_long_budget,
            group_short_budget=raw.group_short_budget,
            meta={
                **raw.meta,
                "prune_coverage": self.coverage,
                "prune_max_total_names": self.max_total_names,
            },
        )


@dataclass(slots=True)
class _RrgK2OpPrune90SectorPreserveWeight(_RrgPrune90SectorPreserveWeight):
    long_per_sector: int = 2
    short_per_sector: int = 1

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        raw = _RrgLongShortRankProportionalWeight(
            gross_long=self.gross_long,
            gross_short=self.gross_short,
        ).build(bundle)
        sector = _required_frame(bundle, "sector").reindex(
            index=raw.base_target_weights.index,
            columns=raw.base_target_weights.columns,
        ).ffill()
        alpha = bundle.alpha.reindex(index=raw.base_target_weights.index, columns=raw.base_target_weights.columns).astype(float)
        short_alpha = _required_frame(bundle, "short_alpha").reindex(index=raw.base_target_weights.index, columns=raw.base_target_weights.columns).astype(float)
        compressed = _compress_sector_op_leaders(
            weights=raw.base_target_weights,
            sector=sector,
            long_scores=alpha,
            short_scores=short_alpha,
            long_per_sector=self.long_per_sector,
            short_per_sector=self.short_per_sector,
        )
        weights = _prune_sector_preserving_weights(
            weights=compressed,
            sector=sector,
            coverage=self.coverage,
            max_total_names=self.max_total_names,
        )
        selected = weights.ne(0.0)
        return ConstructionResult(
            base_target_weights=weights,
            selection_mask=selected,
            group_long_budget=raw.group_long_budget,
            group_short_budget=raw.group_short_budget,
            meta={
                **raw.meta,
                "sector_long_per_sector": self.long_per_sector,
                "sector_short_per_sector": self.short_per_sector,
                "prune_coverage": self.coverage,
                "prune_max_total_names": self.max_total_names,
            },
        )


@dataclass(slots=True)
class _RrgOpRrgSectorCompressedWeight:
    gross_long: float = 1.0
    gross_short: float = 0.5
    long_per_sector: int = 2
    short_per_sector: int = 1

    def __post_init__(self) -> None:
        if self.gross_long < 0.0:
            raise ValueError("gross_long must be non-negative")
        if self.gross_short < 0.0:
            raise ValueError("gross_short must be non-negative")
        _validate_optional_positive_int("long_per_sector", self.long_per_sector)
        _validate_optional_positive_int("short_per_sector", self.short_per_sector)

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        raw = _RrgLongShortRankProportionalWeight(
            gross_long=self.gross_long,
            gross_short=self.gross_short,
        ).build(bundle)
        sector = _required_frame(bundle, "sector").reindex(
            index=raw.base_target_weights.index,
            columns=raw.base_target_weights.columns,
        ).ffill()
        alpha = bundle.alpha.reindex(index=raw.base_target_weights.index, columns=raw.base_target_weights.columns).astype(float)
        short_alpha = _required_frame(bundle, "short_alpha").reindex(index=raw.base_target_weights.index, columns=raw.base_target_weights.columns).astype(float)
        weights = _compress_sector_op_leaders(
            weights=raw.base_target_weights,
            sector=sector,
            long_scores=alpha,
            short_scores=short_alpha,
            long_per_sector=self.long_per_sector,
            short_per_sector=self.short_per_sector,
        )
        selected = weights.ne(0.0)
        return ConstructionResult(
            base_target_weights=weights,
            selection_mask=selected,
            group_long_budget=raw.group_long_budget,
            group_short_budget=raw.group_short_budget,
            meta={
                **raw.meta,
                "sector_long_per_sector": self.long_per_sector,
                "sector_short_per_sector": self.short_per_sector,
            },
        )


def _proportional_rank_weights(scores: pd.Series, *, gross: float) -> pd.Series:
    ranks = scores.rank(method="first", ascending=True)
    total = float(ranks.sum())
    if total <= 0.0:
        return pd.Series(dtype=float)
    return ranks.divide(total).mul(gross).astype(float)


def _filter_candidate_scores(
    scores: pd.Series,
    *,
    min_score: float,
    quantile: float | None,
    max_names: int | None,
) -> pd.Series:
    filtered = scores[scores.ge(float(min_score))]
    if filtered.empty:
        return filtered
    if quantile is not None:
        cutoff = float(filtered.quantile(float(quantile)))
        filtered = filtered[filtered.ge(cutoff)]
    if max_names is not None and len(filtered) > max_names:
        filtered = filtered.sort_values(ascending=False, kind="stable").head(max_names)
    return filtered


def _compress_sector_op_leaders(
    *,
    weights: pd.DataFrame,
    sector: pd.DataFrame,
    long_scores: pd.DataFrame,
    short_scores: pd.DataFrame,
    long_per_sector: int,
    short_per_sector: int,
) -> pd.DataFrame:
    compressed = pd.DataFrame(0.0, index=weights.index, columns=weights.columns, dtype=float)
    for ts in weights.index:
        row = weights.loc[ts]
        sector_row = sector.loc[ts]
        _compress_sector_side(
            output=compressed,
            ts=ts,
            side_names=row[row.gt(0.0)].index,
            per_sector=long_per_sector,
            raw=row,
            sector_row=sector_row,
            score_row=long_scores.loc[ts],
        )
        _compress_sector_side(
            output=compressed,
            ts=ts,
            side_names=row[row.lt(0.0)].index,
            per_sector=short_per_sector,
            raw=row,
            sector_row=sector_row,
            score_row=short_scores.loc[ts],
        )
    return compressed


def _compress_sector_side(
    *,
    output: pd.DataFrame,
    ts: pd.Timestamp,
    side_names: pd.Index,
    per_sector: int,
    raw: pd.Series,
    sector_row: pd.Series,
    score_row: pd.Series,
) -> None:
    if len(side_names) == 0:
        return
    for sector_name in pd.unique(sector_row.loc[side_names].dropna()):
        names = side_names[sector_row.loc[side_names].eq(sector_name)]
        exposure = float(raw.loc[names].sum())
        scores = score_row.reindex(names).dropna()
        scores = scores[scores.gt(0.0)].sort_values(ascending=False, kind="stable")
        chosen = scores.head(per_sector).index
        if len(chosen) == 0:
            continue
        basis = raw.loc[chosen].abs()
        if float(basis.sum()) <= 0.0:
            basis = pd.Series(1.0, index=chosen, dtype=float)
        output.loc[ts, chosen] = basis.divide(float(basis.sum())).mul(exposure)


def _prune_sector_preserving_weights(
    *,
    weights: pd.DataFrame,
    sector: pd.DataFrame,
    coverage: float,
    max_total_names: int,
) -> pd.DataFrame:
    pruned = pd.DataFrame(0.0, index=weights.index, columns=weights.columns, dtype=float)
    for ts in weights.index:
        row = weights.loc[ts]
        sector_row = sector.loc[ts]
        keep: set[str] = set()
        keep.update(_prune_side_keep(row[row.gt(0.0)], sector_row=sector_row, coverage=coverage))
        keep.update(_prune_side_keep(row[row.lt(0.0)].abs(), sector_row=sector_row, coverage=coverage))
        keep = _apply_total_name_soft_cap(row=row, sector_row=sector_row, keep=keep, max_total_names=max_total_names)
        pruned.loc[ts] = _redistribute_by_sector_and_side(row=row, sector_row=sector_row, keep=keep)
    return pruned


def _prune_side_keep(side_abs: pd.Series, *, sector_row: pd.Series, coverage: float) -> set[str]:
    side_abs = side_abs.dropna()
    side_abs = side_abs[side_abs.gt(0.0)]
    if side_abs.empty:
        return set()

    keep: set[str] = set()
    gross = float(side_abs.sum())
    cumulative = 0.0
    for symbol, value in side_abs.sort_values(ascending=False, kind="stable").items():
        keep.add(str(symbol))
        cumulative += float(value)
        if cumulative >= gross * coverage:
            break

    sectors = sector_row.reindex(side_abs.index)
    for sector_name in pd.unique(sectors.dropna()):
        names = sectors[sectors.eq(sector_name)].index
        leader = side_abs.loc[names].sort_values(ascending=False, kind="stable").index[0]
        keep.add(str(leader))
    return keep


def _apply_total_name_soft_cap(
    *,
    row: pd.Series,
    sector_row: pd.Series,
    keep: set[str],
    max_total_names: int,
) -> set[str]:
    if len(keep) <= max_total_names:
        return keep

    keep_index = pd.Index(sorted(keep))
    abs_row = row.reindex(keep_index).abs().fillna(0.0)
    sectors = sector_row.reindex(keep_index)
    protected: set[str] = set()
    signs = np.sign(row.reindex(keep_index).fillna(0.0))
    for side in (1.0, -1.0):
        side_names = keep_index[signs.eq(side)]
        side_sectors = sectors.reindex(side_names)
        for sector_name in pd.unique(side_sectors.dropna()):
            names = side_sectors[side_sectors.eq(sector_name)].index
            leader = abs_row.loc[names].sort_values(ascending=False, kind="stable").index[0]
            protected.add(str(leader))

    next_keep = set(keep)
    removable = [symbol for symbol in abs_row.sort_values(ascending=True, kind="stable").index if str(symbol) not in protected]
    for symbol in removable:
        if len(next_keep) <= max_total_names:
            break
        next_keep.discard(str(symbol))
    return next_keep


def _redistribute_by_sector_and_side(*, row: pd.Series, sector_row: pd.Series, keep: set[str]) -> pd.Series:
    redistributed = pd.Series(0.0, index=row.index, dtype=float)
    for side in (1.0, -1.0):
        side_row = row[row.mul(side).gt(0.0)]
        if side_row.empty:
            continue
        sectors = sector_row.reindex(side_row.index)
        for sector_name in pd.unique(sectors.dropna()):
            names = sectors[sectors.eq(sector_name)].index
            exposure = float(side_row.loc[names].sum())
            kept_names = [name for name in names if str(name) in keep]
            if not kept_names:
                continue
            basis = side_row.loc[kept_names].abs()
            if float(basis.sum()) <= 0.0:
                basis = pd.Series(1.0, index=kept_names, dtype=float)
            redistributed.loc[kept_names] = basis.divide(float(basis.sum())).mul(exposure)
    return redistributed


def _validate_coverage(name: str, value: float) -> None:
    if value <= 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _validate_optional_quantile(name: str, value: float | None) -> None:
    if value is None:
        return
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _validate_optional_positive_int(name: str, value: int | None) -> None:
    if value is None:
        return
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_state_names(name: str, values: tuple[str, ...]) -> None:
    allowed = {"Leading", "Improving", "Weakening", "Lagging", "Unclassified"}
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(f"{name} contains unsupported RRG states: {', '.join(invalid)}")


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
            DatasetId.QW_WI_SEC_26_BIG,
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
        benchmark = benchmark_price_series(benchmark_frame, "IKS200")
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


@dataclass(slots=True)
class _RrgOpRrgSignal:
    lookback: int = 20
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    rrg_transition_threshold: float = 0.005
    op_rrg_medium_lookback: int = 126
    op_rrg_momentum_lookback: int = 21
    op_rrg_short_lookback: int = 42
    op_rrg_transition_threshold: float = 0.005
    long_price_states: tuple[str, ...] = ("Leading", "Improving", "Weakening")
    long_op_states: tuple[str, ...] = ("Leading", "Improving")
    short_price_states: tuple[str, ...] = ("Lagging",)
    short_op_states: tuple[str, ...] = ("Lagging", "Weakening")
    op_rrg_exclude_bm_weight_gt: float | None = None

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        datasets = (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_BM,
            DatasetId.QW_K200_YN,
            DatasetId.QW_WI_SEC_26_BIG,
            DatasetId.QW_MKTCAP,
            DatasetId.QW_MKTCAP_FLT,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_OP_NFQ2,
            DatasetId.QW_OP_NFY1,
            DatasetId.QW_OP_FWD_12M,
        )
        if self.op_rrg_exclude_bm_weight_gt is not None:
            return (*datasets, DatasetId.QW_BM_WEIGHTS)
        return datasets

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

        price_state, _price_long_sector, _price_short_sector = _build_rrg_context(
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
        op_for_rrg = market.frames["op_fwd_12m"].reindex(index=close.index, columns=close.columns).ffill().astype(float)
        if self.op_rrg_exclude_bm_weight_gt is not None:
            benchmark_weights = market.frames["bm_weights"].reindex(index=close.index, columns=close.columns).fillna(0.0).astype(float)
            op_for_rrg = _exclude_op_by_benchmark_weight(
                op=op_for_rrg,
                benchmark_weights=benchmark_weights,
                threshold=float(self.op_rrg_exclude_bm_weight_gt),
            )
        op_state = _build_op_rrg_state(
            op=op_for_rrg,
            sector=sector,
            membership=k200,
            medium_lookback=self.op_rrg_medium_lookback,
            momentum_lookback=self.op_rrg_momentum_lookback,
            short_lookback=self.op_rrg_short_lookback,
            transition_threshold=self.op_rrg_transition_threshold,
        )
        stock_op = _build_stock_op_revision(
            frames=market.frames,
            index=close.index,
            columns=close.columns,
            sector=sector,
            lookback=self.lookback,
        )

        price_by_symbol = _map_sector_state_to_symbols(sector=sector, rrg_state=price_state.reindex(index=close.index))
        op_by_symbol = _map_sector_state_to_symbols(sector=sector, rrg_state=op_state.reindex(index=close.index))
        long_ok = (
            price_by_symbol.isin(self.long_price_states)
            & op_by_symbol.isin(self.long_op_states)
            & stock_op.gt(0.0)
            & k200
        )
        short_ok = (
            price_by_symbol.isin(self.short_price_states)
            & op_by_symbol.isin(self.short_op_states)
            & stock_op.lt(0.0)
            & k200
        )
        candidate_score = stock_op.where(long_ok & stock_op.gt(0.0))
        short_candidate_score = stock_op.mul(-1.0).where(short_ok & stock_op.lt(0.0))

        return SignalBundle(
            alpha=candidate_score,
            context={
                "tradable": k200,
                "entry_mask": long_ok.fillna(False).astype(bool),
                "hold_mask": long_ok.fillna(False).astype(bool),
                "short_entry_mask": short_ok.fillna(False).astype(bool),
                "short_hold_mask": short_ok.fillna(False).astype(bool),
                "short_alpha": short_candidate_score,
                "sector": sector,
                "price_rrg_state": price_state,
                "op_rrg_state": op_state,
                "stock_op_revision": stock_op,
            },
            meta={
                "price_rrg_state": price_state,
                "op_rrg_state": op_state,
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


def _build_op_rrg_state(
    *,
    op: pd.DataFrame,
    sector: pd.DataFrame,
    membership: pd.DataFrame,
    medium_lookback: int,
    momentum_lookback: int,
    short_lookback: int,
    transition_threshold: float = 0.0,
) -> pd.DataFrame:
    sector_op = _sector_sum(values=op, sector=sector, membership=membership)
    market_op = op.where(membership).sum(axis=1, min_count=1)
    positive_sector_op = sector_op.where(sector_op.gt(0.0))
    positive_market_op = market_op.where(market_op.gt(0.0))
    op_share = positive_sector_op.divide(positive_market_op, axis=0)

    medium_mean = op_share.rolling(medium_lookback, min_periods=max(5, medium_lookback // 3)).mean()
    short_mean = op_share.rolling(short_lookback, min_periods=max(5, short_lookback // 3)).mean()
    relative_strength = op_share.divide(medium_mean.replace(0.0, np.nan)) - 1.0
    short_relative = op_share.divide(short_mean.replace(0.0, np.nan)) - 1.0
    momentum = short_relative - short_relative.shift(momentum_lookback)

    state, _long_sector, _short_sector = _classify_rrg_states(
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
    return state.where(op_share.notna(), "Unclassified")


def _exclude_op_by_benchmark_weight(
    *,
    op: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    aligned_weights = benchmark_weights.reindex(index=op.index, columns=op.columns).fillna(0.0).astype(float)
    return op.mask(aligned_weights.gt(float(threshold)))


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


def _sector_sum(*, values: pd.DataFrame, sector: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    rows: dict[pd.Timestamp, dict[object, float]] = {}
    for ts in values.index:
        valid = membership.loc[ts].astype(bool) & values.loc[ts].notna() & sector.loc[ts].notna()
        row: dict[object, float] = {}
        for sector_name in pd.unique(sector.loc[ts, valid]):
            names = values.columns[valid & sector.loc[ts].eq(sector_name)]
            row[sector_name] = float(values.loc[ts, names].sum())
        rows[ts] = row
    return pd.DataFrame.from_dict(rows, orient="index").reindex(index=values.index)


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
