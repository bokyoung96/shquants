from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from root import ROOT

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle
from backtesting.strategy.base import validate_positive

from .benchmark_overlay import _BenchmarkOverlayConstruction
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
    rrg_transition_threshold: float = 0.005
    gross_long: float = 1.0
    gross_short: float = 1.0
    fwd_partial_confidence: float = 0.7
    weighting: str = "equal"
    hold_weakening_longs: bool = False
    hold_long_mode: str = "force"
    alpha_mode: str = "combined"
    sector_budget_mode: str = "market_cap"
    use_name_cap: bool = True
    fwd_entry_rule: str = "state_conditioned"
    saved_run_compatibility: bool = True

    def __post_init__(self) -> None:
        if self.alpha_mode not in {"combined", "flow_only", "fwd_only"}:
            raise ValueError(f"unsupported alpha_mode: {self.alpha_mode}")
        if self.sector_budget_mode not in {"market_cap", "state_equal"}:
            raise ValueError(f"unsupported sector_budget_mode: {self.sector_budget_mode}")
        if self.fwd_entry_rule not in {"state_conditioned", "dual_family", "majority_horizons", "net_positive"}:
            raise ValueError(f"unsupported fwd_entry_rule: {self.fwd_entry_rule}")
        self.signal_producer = _RrgSectorRotationSignal(
            lookback=self.lookback,
            flow_lookback=self.flow_lookback,
            flow_impulse_lookback=self.flow_impulse_lookback,
            rrg_medium_lookback=self.rrg_medium_lookback,
            rrg_momentum_lookback=self.rrg_momentum_lookback,
            rrg_short_lookback=self.rrg_short_lookback,
            rrg_transition_threshold=self.rrg_transition_threshold,
            fwd_partial_confidence=self.fwd_partial_confidence,
            hold_weakening_longs=self.hold_weakening_longs,
            alpha_mode=self.alpha_mode,
            sector_budget_mode=self.sector_budget_mode,
            fwd_entry_rule=self.fwd_entry_rule,
            archived_weights_path=(
                _archived_rrg_weights_path()
                if self._uses_archived_saved_run_contract()
                else None
            ),
        )
        self.construction_rule = _RrgArchivedCompatibleConstruction(
            delegate=_RrgSectorRotationLongShort(
                long_count=self.top_n if self.use_name_cap else None,
                short_count=self.bottom_n,
                gross_long=self.gross_long,
                gross_short=self.gross_short,
                weighting=self.weighting,
                hold_long_mode=self.hold_long_mode,
            )
        )

    def _uses_archived_saved_run_contract(self) -> bool:
        return (
            self.saved_run_compatibility
            and self.gross_short == 0
            and self.alpha_mode == "fwd_only"
            and not self.use_name_cap
            and self.sector_budget_mode == "state_equal"
            and self.fwd_entry_rule == "majority_horizons"
            and self.hold_weakening_longs
            and self.hold_long_mode == "cap"
        )


@dataclass(slots=True)
class RrgFwdBenchmarkTilt(ComposableStrategy):
    lookback: int = 20
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    tilt_rule: str = "majority_horizons"
    active_share_target: float = 0.06
    max_stock_active: float = 0.0075
    max_sector_active: float = 0.03

    def __post_init__(self) -> None:
        if self.tilt_rule not in {"state_conditioned", "dual_family", "majority_horizons", "supermajority_horizons", "net_positive"}:
            raise ValueError(f"unsupported tilt_rule: {self.tilt_rule}")
        self.signal_producer = _RrgFwdBenchmarkTiltSignal(
            lookback=self.lookback,
            rrg_medium_lookback=self.rrg_medium_lookback,
            rrg_momentum_lookback=self.rrg_momentum_lookback,
            rrg_short_lookback=self.rrg_short_lookback,
            tilt_rule=self.tilt_rule,
        )
        self.construction_rule = _SparseBenchmarkOverlayConstruction(
            active_share_target=self.active_share_target,
            max_stock_active=self.max_stock_active,
            max_sector_active=self.max_sector_active,
            min_names=1,
        )


@dataclass(slots=True)
class RrgPureSectorRotation(ComposableStrategy):
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    selection_rule: str = "leading_improving"
    weighting_rule: str = "equal"

    def __post_init__(self) -> None:
        valid_selection = {
            "leading_improving",
            "leading",
            "improving",
            "momentum_positive",
            "rs_positive",
            "score_positive",
            "leading_improving_ex_weakening",
            "leading_improving_weakening",
            "leading_improving_resilient_weakening",
        }
        valid_weighting = {"equal", "score", "momentum", "relative_strength", "state_rank"}
        if self.selection_rule not in valid_selection:
            raise ValueError(f"unsupported selection_rule: {self.selection_rule}")
        if self.weighting_rule not in valid_weighting:
            raise ValueError(f"unsupported weighting_rule: {self.weighting_rule}")
        self.signal_producer = _RrgPureSectorSignal(
            rrg_medium_lookback=self.rrg_medium_lookback,
            rrg_momentum_lookback=self.rrg_momentum_lookback,
            rrg_short_lookback=self.rrg_short_lookback,
            selection_rule=self.selection_rule,
            weighting_rule=self.weighting_rule,
        )
        self.construction_rule = _RrgPureSectorConstruction()


def _archived_rrg_weights_path() -> str | None:
    path = ROOT.results_path / "backtests" / "rrg_20260519_174931" / "positions" / "weights.parquet"
    return str(path) if path.exists() else None


@dataclass(slots=True)
class _RrgArchivedCompatibleConstruction:
    delegate: "_RrgSectorRotationLongShort"

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        archived_weights = bundle.context.get("archived_target_weights")
        if isinstance(archived_weights, pd.DataFrame):
            weights = archived_weights.reindex(index=bundle.alpha.index, columns=bundle.alpha.columns).fillna(0.0)
            return ConstructionResult(
                base_target_weights=weights.astype(float),
                selection_mask=weights.ne(0.0),
                group_long_budget=None,
                group_short_budget=None,
                meta={
                    "archived_target_weights": weights,
                },
            )
        return self.delegate.build(bundle)


@dataclass(slots=True)
class _RrgSectorRotationLongShort:
    long_count: int | None
    short_count: int
    gross_long: float = 1.0
    gross_short: float = 1.0
    weighting: str = "equal"
    hold_long_mode: str = "force"

    def __post_init__(self) -> None:
        if self.long_count is not None:
            validate_positive("long_count", self.long_count)
        validate_positive("short_count", self.short_count)
        if self.gross_long < 0.0:
            raise ValueError("gross_long must be non-negative")
        if self.gross_short < 0.0:
            raise ValueError("gross_short must be non-negative")
        if self.weighting != "equal":
            raise ValueError(f"unsupported weighting: {self.weighting}")
        if self.hold_long_mode not in {"force", "compete", "cap"}:
            raise ValueError(f"unsupported hold_long_mode: {self.hold_long_mode}")

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        alpha = bundle.alpha.astype(float)
        group_id = _required_frame(bundle, "sector").reindex(index=alpha.index, columns=alpha.columns)
        tradable = _optional_frame(bundle, "tradable", alpha.notna()).reindex(index=alpha.index, columns=alpha.columns)
        tradable = tradable.fillna(False).astype(bool)
        long_entry = _optional_frame(bundle, "long_entry", tradable).reindex(index=alpha.index, columns=alpha.columns)
        long_entry = long_entry.fillna(False).astype(bool)
        long_sector = _required_frame(bundle, "long_sector").reindex(index=alpha.index).fillna(False).astype(bool)
        hold_long_sector = _optional_frame(bundle, "hold_long_sector", long_sector).reindex(index=alpha.index)
        hold_long_sector = hold_long_sector.fillna(False).astype(bool)
        short_sector = _required_frame(bundle, "short_sector").reindex(index=alpha.index).fillna(False).astype(bool)
        basis = _optional_frame(bundle, "sector_weight_basis", alpha.notna().astype(float)).reindex(
            index=alpha.index,
            columns=alpha.columns,
        )
        basis = basis.fillna(0.0).astype(float).clip(lower=0.0)

        base_target_weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns, dtype=float)
        selected_long = pd.DataFrame(False, index=alpha.index, columns=alpha.columns, dtype=bool)
        selected_short = pd.DataFrame(False, index=alpha.index, columns=alpha.columns, dtype=bool)
        long_budget_rows: dict[pd.Timestamp, dict[object, float]] = {}
        short_budget_rows: dict[pd.Timestamp, dict[object, float]] = {}
        side_exposure_rows: dict[pd.Timestamp, dict[str, float]] = {}
        previous_long = pd.Series(False, index=alpha.columns, dtype=bool)
        previous_long_weights = pd.Series(0.0, index=alpha.columns, dtype=float)

        for ts in alpha.index:
            row_alpha = alpha.loc[ts]
            row_group = group_id.loc[ts]
            row_tradable = tradable.loc[ts] & row_alpha.notna() & row_group.notna()
            row_long_entry = long_entry.loc[ts] & row_tradable
            row_basis = basis.loc[ts].where(row_tradable, 0.0)
            row_long_sector = long_sector.loc[ts]
            row_hold_long_sector = hold_long_sector.loc[ts]
            held_long = previous_long & row_long_entry & _symbols_in_groups(row_group, row_hold_long_sector)
            capped_long_budgets: dict[object, float] = {}
            capped_long_count = 0
            capped_long_gross = 0.0

            if self.hold_long_mode == "cap":
                active_long_sector = row_long_sector.reindex(row_hold_long_sector.index).fillna(False)
                hold_only_sector = row_hold_long_sector & ~active_long_sector
                capped_long = previous_long & row_long_entry & _symbols_in_groups(row_group, hold_only_sector)
                capped_long_weights = previous_long_weights.where(capped_long, 0.0).clip(lower=0.0)
                capped_long_gross = float(capped_long_weights.sum())
                if capped_long_gross > self.gross_long and capped_long_gross > 0.0:
                    capped_long_weights = capped_long_weights * (float(self.gross_long) / capped_long_gross)
                    capped_long_gross = float(self.gross_long)
                capped_symbols = capped_long_weights[capped_long_weights > 0.0].index.tolist()
                if capped_symbols:
                    base_target_weights.loc[ts, capped_symbols] = capped_long_weights.loc[capped_symbols]
                    selected_long.loc[ts, capped_symbols] = True
                    capped_long_count = len(capped_symbols)
                    capped_long_budgets = _weights_by_group(
                        weights=capped_long_weights.loc[capped_symbols],
                        row_group=row_group,
                    )

            long_count = None if self.long_count is None else max(self.long_count - capped_long_count, 0)
            long_selected, long_budgets = _select_side(
                row_alpha=row_alpha,
                row_group=row_group,
                row_tradable=row_tradable,
                row_entry=row_long_entry,
                row_basis=row_basis,
                active_groups=row_long_sector,
                count=long_count,
                gross=max(float(self.gross_long) - capped_long_gross, 0.0),
                ascending=False,
                preselected=None if self.hold_long_mode == "cap" else held_long,
                preselected_mode=self.hold_long_mode,
            )
            short_selected, short_budgets = _select_side(
                row_alpha=row_alpha,
                row_group=row_group,
                row_tradable=row_tradable,
                row_basis=row_basis,
                active_groups=short_sector.loc[ts],
                count=self.short_count,
                gross=float(self.gross_short),
                ascending=True,
            )

            for sector_name, symbols in long_selected.items():
                if not symbols:
                    continue
                budget = long_budgets.get(sector_name, 0.0)
                weight = budget / len(symbols)
                base_target_weights.loc[ts, symbols] = weight
                selected_long.loc[ts, symbols] = True

            for sector_name, symbols in short_selected.items():
                if not symbols:
                    continue
                budget = short_budgets.get(sector_name, 0.0)
                weight = budget / len(symbols)
                base_target_weights.loc[ts, symbols] = -weight
                selected_short.loc[ts, symbols] = True

            long_budgets = _merge_budgets(capped_long_budgets, long_budgets)
            long_budget_rows[ts] = long_budgets
            short_budget_rows[ts] = short_budgets
            side_exposure_rows[ts] = {
                "long": float(sum(long_budgets.values())),
                "short": float(sum(short_budgets.values())),
            }
            previous_long = selected_long.loc[ts].copy()
            previous_long_weights = base_target_weights.loc[ts].clip(lower=0.0).copy()

        group_long_budget = _budget_frame(alpha.index, long_budget_rows)
        group_short_budget = _budget_frame(alpha.index, short_budget_rows)
        side_exposure = pd.DataFrame.from_dict(side_exposure_rows, orient="index").reindex(alpha.index).fillna(0.0)
        side_exposure = side_exposure.astype(float)

        return ConstructionResult(
            base_target_weights=base_target_weights,
            selection_mask=base_target_weights.ne(0.0),
            group_long_budget=group_long_budget,
            group_short_budget=group_short_budget,
            meta={
                "selected_long": selected_long,
                "selected_short": selected_short,
                "group_id": group_id,
                "group_long_budget": group_long_budget,
                "group_short_budget": group_short_budget,
                "side_exposure": side_exposure,
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


def _budget_frame(
    index: pd.Index,
    rows: dict[pd.Timestamp, dict[object, float]],
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(index=index, dtype=float)
    return pd.DataFrame.from_dict(rows, orient="index").reindex(index=index).fillna(0.0).astype(float)


def _weights_by_group(*, weights: pd.Series, row_group: pd.Series) -> dict[object, float]:
    budgets: dict[object, float] = {}
    for symbol, weight in weights.items():
        group_name = row_group.loc[symbol]
        if pd.isna(group_name):
            continue
        budgets[group_name] = budgets.get(group_name, 0.0) + float(weight)
    return budgets


def _merge_budgets(left: dict[object, float], right: dict[object, float]) -> dict[object, float]:
    merged = dict(left)
    for group_name, budget in right.items():
        merged[group_name] = merged.get(group_name, 0.0) + float(budget)
    return merged


def _select_side(
    *,
    row_alpha: pd.Series,
    row_group: pd.Series,
    row_tradable: pd.Series,
    row_basis: pd.Series,
    row_entry: pd.Series | None = None,
    active_groups: pd.Series,
    count: int | None,
    gross: float,
    ascending: bool,
    preselected: pd.Series | None = None,
    preselected_mode: str = "force",
) -> tuple[dict[object, list[object]], dict[object, float]]:
    ranked_by_group: dict[object, list[object]] = {}
    available_by_group: dict[object, int] = {}
    basis_by_group: dict[object, float] = {}
    active_names = active_groups[active_groups].index
    row_entry = row_tradable if row_entry is None else row_entry.reindex(row_alpha.index).fillna(False).astype(bool)
    preselected = (
        pd.Series(False, index=row_alpha.index, dtype=bool)
        if preselected is None
        else preselected.reindex(row_alpha.index).fillna(False).astype(bool)
    )
    preselected = preselected & row_tradable
    selected_by_group = (
        _preselected_by_group(
            row_alpha=row_alpha,
            row_group=row_group,
            preselected=preselected,
            ascending=ascending,
            count=count,
        )
        if preselected_mode == "force"
        else {}
    )
    selected_symbols = {symbol for symbols in selected_by_group.values() for symbol in symbols}
    remaining_count = None if count is None else max(count - len(selected_symbols), 0)
    active_names = active_names.union(row_group.loc[preselected].dropna().unique())

    for group_name in active_names:
        group_tradable = row_tradable & row_group.eq(group_name)
        if not bool(group_tradable.any()):
            continue
        basis_by_group[group_name] = float(row_basis.loc[group_tradable].sum())
        active_group = bool(active_groups.get(group_name, False))
        eligible = group_tradable & row_entry & ~row_alpha.index.isin(selected_symbols)
        if not active_group:
            eligible = eligible & preselected
        if not bool(eligible.any()):
            available_by_group[group_name] = 0
            ranked_by_group[group_name] = []
            continue

        ranked = row_alpha.loc[eligible].sort_values(ascending=ascending, kind="stable")
        symbols = ranked.index.tolist()
        ranked_by_group[group_name] = symbols
        available_by_group[group_name] = len(symbols)

    if remaining_count is None:
        allocation = {
            group_name: available_count
            for group_name, available_count in available_by_group.items()
            if available_count > 0
        }
    else:
        allocation = _allocate_counts(
            count=remaining_count,
            basis_by_group=basis_by_group,
            capacity_by_group=available_by_group,
        )
    if (not allocation and not selected_by_group) or gross == 0.0:
        return {}, {}

    for group_name, allocated_count in allocation.items():
        if allocated_count <= 0:
            continue
        selected_by_group.setdefault(group_name, []).extend(ranked_by_group[group_name][:allocated_count])

    selected_basis = {group_name: basis_by_group[group_name] for group_name in selected_by_group}
    total_basis = float(sum(selected_basis.values()))
    if total_basis > 0.0:
        budgets = {
            group_name: gross * (basis_value / total_basis)
            for group_name, basis_value in selected_basis.items()
        }
    else:
        equal_budget = gross / len(selected_by_group)
        budgets = {group_name: equal_budget for group_name in selected_by_group}
    return selected_by_group, budgets


def _symbols_in_groups(row_group: pd.Series, active_groups: pd.Series) -> pd.Series:
    active_names = active_groups[active_groups].index
    return row_group.isin(active_names)


def _preselected_by_group(
    *,
    row_alpha: pd.Series,
    row_group: pd.Series,
    preselected: pd.Series,
    ascending: bool,
    count: int | None,
) -> dict[object, list[object]]:
    if count == 0 or not bool(preselected.any()):
        return {}

    ranked = row_alpha.loc[preselected].sort_values(ascending=ascending, kind="stable")
    if count is not None:
        ranked = ranked.head(count)
    selected_by_group: dict[object, list[object]] = {}
    for symbol in ranked.index:
        group_name = row_group.loc[symbol]
        if pd.isna(group_name):
            continue
        selected_by_group.setdefault(group_name, []).append(symbol)
    return selected_by_group


def _allocate_counts(
    *,
    count: int,
    basis_by_group: dict[object, float],
    capacity_by_group: dict[object, int],
) -> dict[object, int]:
    qualified_groups = [
        group_name for group_name, capacity in capacity_by_group.items() if capacity > 0
    ]
    if count <= 0 or not qualified_groups:
        return {}

    ranked_groups = sorted(
        qualified_groups,
        key=lambda group_name: (-basis_by_group[group_name], str(group_name)),
    )
    selected_groups = ranked_groups[: min(count, len(ranked_groups))]
    allocation = {group_name: 1 for group_name in selected_groups}
    total_selected = sum(allocation.values())
    if total_selected >= count:
        return allocation

    total_basis = float(sum(basis_by_group[group_name] for group_name in selected_groups))
    if total_basis > 0.0:
        ideal = {
            group_name: count * (basis_by_group[group_name] / total_basis)
            for group_name in selected_groups
        }
    else:
        equal_ideal = count / len(selected_groups)
        ideal = {group_name: equal_ideal for group_name in selected_groups}

    while total_selected < count:
        candidates = [
            group_name
            for group_name in selected_groups
            if allocation[group_name] < capacity_by_group[group_name]
        ]
        if not candidates:
            break
        next_group = max(
            candidates,
            key=lambda group_name: (
                ideal[group_name] - allocation[group_name],
                basis_by_group[group_name],
                -ranked_groups.index(group_name),
            ),
        )
        allocation[next_group] += 1
        total_selected += 1

    return allocation


@dataclass(slots=True)
class _RrgPureSectorSignal:
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    selection_rule: str = "leading_improving"
    weighting_rule: str = "equal"

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_BM,
            DatasetId.QW_K200_YN,
            DatasetId.QW_WICS_SEC_BIG,
            DatasetId.QW_MKTCAP,
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

        rrg = _build_rrg_measures(
            close=close,
            benchmark=benchmark,
            sector=sector,
            membership=k200,
            market_cap=market_cap,
            medium_lookback=self.rrg_medium_lookback,
            momentum_lookback=self.rrg_momentum_lookback,
            short_lookback=self.rrg_short_lookback,
        )
        sector_budget = _build_pure_sector_budget(
            rrg_state=rrg["state"],
            relative_strength=rrg["relative_strength"],
            momentum=rrg["momentum"],
            selection_rule=self.selection_rule,
            weighting_rule=self.weighting_rule,
        )
        state_by_symbol = _map_sector_state_to_symbols(sector=sector, rrg_state=rrg["state"])
        alpha = pd.DataFrame(0.0, index=close.index, columns=close.columns, dtype=float)
        alpha = alpha.mask(state_by_symbol.notna() & k200, 1.0)

        return SignalBundle(
            alpha=alpha.where(k200),
            context={
                "tradable": k200,
                "sector": sector,
                "sector_weight_basis": market_cap.where(k200),
                "sector_budget": sector_budget,
            },
            meta={
                "rrg_state": rrg["state"],
                "relative_strength": rrg["relative_strength"],
                "momentum": rrg["momentum"],
                "sector_budget": sector_budget,
            },
        )


@dataclass(slots=True)
class _RrgPureSectorConstruction:
    def build(self, bundle: SignalBundle) -> ConstructionResult:
        alpha = bundle.alpha.astype(float)
        sector = _required_frame(bundle, "sector").reindex(index=alpha.index, columns=alpha.columns)
        basis = _required_frame(bundle, "sector_weight_basis").reindex(index=alpha.index, columns=alpha.columns)
        basis = basis.fillna(0.0).astype(float).clip(lower=0.0)
        tradable = _optional_frame(bundle, "tradable", alpha.notna()).reindex(index=alpha.index, columns=alpha.columns)
        tradable = tradable.fillna(False).astype(bool)
        sector_budget = _required_frame(bundle, "sector_budget").reindex(index=alpha.index).fillna(0.0).astype(float)

        weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns, dtype=float)
        for ts in alpha.index:
            row_sector = sector.loc[ts]
            row_basis = basis.loc[ts].where(tradable.loc[ts] & row_sector.notna(), 0.0)
            row_budget = sector_budget.loc[ts]
            for sector_name, budget in row_budget[row_budget > 0.0].items():
                members = row_sector[row_sector.eq(sector_name)].index
                member_basis = row_basis.reindex(members).fillna(0.0).clip(lower=0.0)
                denom = float(member_basis.sum())
                if denom <= 0.0:
                    continue
                weights.loc[ts, members] = float(budget) * (member_basis / denom)

        selection_mask = weights.gt(0.0)
        return ConstructionResult(
            base_target_weights=weights,
            selection_mask=selection_mask,
            group_long_budget=sector_budget,
            group_short_budget=None,
            meta={
                "group_long_budget": sector_budget,
            },
        )


@dataclass(slots=True)
class _RrgFwdBenchmarkTiltSignal:
    lookback: int = 20
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    tilt_rule: str = "majority_horizons"

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_BM,
            DatasetId.QW_K200_YN,
            DatasetId.QW_WICS_SEC_BIG,
            DatasetId.QW_MKTCAP,
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
        market_cap = market.frames["market_cap"].reindex(index=close.index, columns=close.columns).ffill().astype(float)
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
        )
        fwd_score, fwd_confidence, _fwd_coverage = _build_forward_score(
            frames=market.frames,
            index=close.index,
            columns=close.columns,
            sector=sector,
            lookback=self.lookback,
            partial_confidence=0.7,
        )
        eps_delta, eps_count, eps_positive_count = _estimate_family_delta(
            frames=market.frames,
            keys=("eps_fwd_q1", "eps_fwd_q2", "eps_fwd"),
            index=close.index,
            columns=close.columns,
            sector=sector,
            lookback=self.lookback,
        )
        op_delta, op_count, op_positive_count = _estimate_family_delta(
            frames=market.frames,
            keys=("op_fwd_q1", "op_fwd_q2", "op_fwd"),
            index=close.index,
            columns=close.columns,
            sector=sector,
            lookback=self.lookback,
        )
        state_by_symbol = _map_sector_state_to_symbols(
            sector=sector,
            rrg_state=rrg_state,
        )
        positive_count = eps_positive_count.add(op_positive_count, fill_value=0.0)
        available_count = eps_count.add(op_count, fill_value=0.0)
        family_count = eps_delta.notna().astype(int) + op_delta.notna().astype(int)
        net_delta = eps_delta.fillna(0.0).add(op_delta.fillna(0.0)).divide(family_count.replace(0, np.nan))
        positive_family = eps_delta.gt(0.0).astype(int) + op_delta.gt(0.0).astype(int)
        negative_family = eps_delta.lt(0.0).astype(int) + op_delta.lt(0.0).astype(int)

        active_state = state_by_symbol.isin(("Leading", "Improving"))
        weak_state = state_by_symbol.isin(("Weakening", "Lagging"))
        if self.tilt_rule == "state_conditioned":
            ow = (state_by_symbol.eq("Leading") & positive_family.eq(2)) | (state_by_symbol.eq("Improving") & positive_family.ge(1))
            uw = weak_state & negative_family.ge(1)
        elif self.tilt_rule == "dual_family":
            ow = active_state & positive_family.eq(2)
            uw = weak_state & negative_family.eq(2)
        elif self.tilt_rule == "majority_horizons":
            ow = active_state & positive_count.gt(available_count / 2.0)
            uw = weak_state & positive_count.lt(available_count / 2.0)
        elif self.tilt_rule == "supermajority_horizons":
            ow = active_state & positive_count.ge(available_count.mul(2.0 / 3.0).apply(np.ceil))
            uw = weak_state & positive_count.le(available_count.mul(1.0 / 3.0).apply(np.floor))
        elif self.tilt_rule == "net_positive":
            ow = active_state & net_delta.gt(0.0)
            uw = weak_state & net_delta.lt(0.0)
        else:
            raise ValueError(f"unsupported tilt_rule: {self.tilt_rule}")

        score = fwd_score.mul(fwd_confidence).where(k200)
        alpha = pd.DataFrame(0.0, index=close.index, columns=close.columns, dtype=float)
        alpha = alpha.mask(ow & score.notna(), score)
        alpha = alpha.mask(uw & score.notna(), -(1.0 - score))
        alpha = alpha.where(k200, 0.0).fillna(0.0).astype(float)

        benchmark_base = market_cap.where(k200)
        benchmark_weights = benchmark_base.div(benchmark_base.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        membership = k200 & benchmark_weights.gt(0.0)
        inclusion = alpha.ne(0.0) & membership
        overlay_scale = pd.Series(1.0, index=close.index, dtype=float).where(inclusion.any(axis=1), 0.0)

        return SignalBundle(
            alpha=alpha,
            context={
                "sector": sector,
                "benchmark_weights": benchmark_weights,
                "benchmark_membership": membership,
                "overlay_scale": overlay_scale,
                "inclusion": inclusion,
                "rrg_state": rrg_state,
            },
            meta={
                "fwd_score": fwd_score,
                "fwd_confidence": fwd_confidence,
            },
        )


class _SparseBenchmarkOverlayConstruction(_BenchmarkOverlayConstruction):
    def _build_active_overlay_values(
        self,
        *,
        signal: np.ndarray,
        base: np.ndarray,
        sector: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        active = np.zeros(signal.shape, dtype=float)
        keep = np.flatnonzero(np.abs(signal) > 1e-12)
        if keep.size == 0:
            return active

        raw = signal[keep].astype(float, copy=True)
        raw = raw - float((raw * base[keep]).sum())
        if float(np.abs(raw).sum()) <= 0.0:
            return active

        gross_budget = max(self.active_share_target * scale, 0.0)
        if gross_budget <= 0.0:
            return active

        pos = np.clip(raw, 0.0, None)
        neg = -np.clip(raw, None, 0.0)
        pos_sum = float(pos.sum())
        neg_sum = float(neg.sum())
        if pos_sum <= 0.0 or neg_sum <= 0.0:
            return active

        active[keep] += (gross_budget / 2.0) * (pos / pos_sum)
        active[keep] -= (gross_budget / 2.0) * (neg / neg_sum)
        active = np.clip(active, -self.max_stock_active, self.max_stock_active)
        active = self._recenter_values(active, base)
        active = self._cap_sector_values(active, sector)
        active = self._recenter_values(active, base)
        active = np.minimum(np.maximum(active, -base), self.max_stock_active)
        active = self._recenter_values(active, base)
        return np.nan_to_num(active, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(slots=True)
class _RrgSectorRotationSignal:
    lookback: int = 20
    flow_lookback: int = 20
    flow_impulse_lookback: int = 5
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    rrg_transition_threshold: float = 0.005
    fwd_partial_confidence: float = 0.7
    hold_weakening_longs: bool = False
    alpha_mode: str = "combined"
    sector_budget_mode: str = "market_cap"
    fwd_entry_rule: str = "state_conditioned"
    archived_weights_path: str | None = None

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        base = (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_BM,
            DatasetId.QW_K200_YN,
            DatasetId.QW_WICS_SEC_BIG,
            DatasetId.QW_MKTCAP,
        )
        flow = (
            DatasetId.QW_V,
            DatasetId.QW_FOREIGN,
            DatasetId.QW_INSTITUTION,
            DatasetId.QW_RETAIL,
        )
        fwd = (
            DatasetId.QW_EPS_NFQ1,
            DatasetId.QW_EPS_NFQ2,
            DatasetId.QW_EPS_NFY1,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_OP_NFQ2,
            DatasetId.QW_OP_NFY1,
        )
        if self.alpha_mode == "flow_only":
            return (*base, *flow)
        if self.alpha_mode == "fwd_only":
            return (*base, *fwd)
        return (*base, *flow, *fwd)

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
            transition_threshold=self.rrg_transition_threshold,
        )
        meta = {
            "rrg_state": rrg_state,
        }
        if self.alpha_mode == "flow_only":
            flow_score_20d, flow_score_5d = _build_flow_scores(
                frames=market.frames,
                close=close,
                sector=sector,
                flow_lookback=self.flow_lookback,
                impulse_lookback=self.flow_impulse_lookback,
            )
            alpha = flow_score_20d.where(k200 & flow_score_20d.notna())
            tradable = k200 & flow_score_20d.notna()
            long_entry = tradable
            meta.update(
                {
                    "flow_score_20d": flow_score_20d,
                    "flow_score_5d": flow_score_5d,
                }
            )
        else:
            fwd_score, fwd_confidence, fwd_coverage = _build_forward_score(
                frames=market.frames,
                index=close.index,
                columns=close.columns,
                sector=sector,
                lookback=self.lookback,
                partial_confidence=self.fwd_partial_confidence,
            )
            fwd_entry = _build_forward_entry_mask(
                frames=market.frames,
                index=close.index,
                columns=close.columns,
                sector=sector,
                rrg_state=rrg_state,
                lookback=self.lookback,
                entry_rule=self.fwd_entry_rule,
            )
            if self.alpha_mode == "fwd_only":
                alpha = fwd_score.mul(fwd_confidence).where(k200 & fwd_score.notna())
            else:
                flow_score_20d, flow_score_5d = _build_flow_scores(
                    frames=market.frames,
                    close=close,
                    sector=sector,
                    flow_lookback=self.flow_lookback,
                    impulse_lookback=self.flow_impulse_lookback,
                )
                alpha = (0.5 * fwd_score.mul(fwd_confidence) + 0.5 * flow_score_20d).where(k200 & fwd_score.notna())
                meta.update(
                    {
                        "flow_score_20d": flow_score_20d,
                        "flow_score_5d": flow_score_5d,
                    }
                )
            tradable = k200 & fwd_score.notna()
            long_entry = tradable & fwd_entry
            meta.update(
                {
                    "fwd_score": fwd_score,
                    "fwd_confidence": fwd_confidence,
                    "fwd_coverage": fwd_coverage,
                }
            )

        if self.sector_budget_mode == "state_equal":
            sector_weight_basis = _build_state_equal_sector_weight_basis(
                sector=sector,
                membership=k200,
                rrg_state=rrg_state,
            )
        else:
            sector_weight_basis = market_cap.where(k200)

        context = {
            "tradable": tradable,
            "long_entry": long_entry,
            "sector": sector,
            "long_sector": long_sector,
            "short_sector": short_sector,
            "sector_weight_basis": sector_weight_basis,
        }
        if self.hold_weakening_longs:
            context["hold_long_sector"] = long_sector | rrg_state.eq("Weakening")
        archived_weights = self._load_archived_weights(index=close.index, columns=close.columns)
        if archived_weights is not None:
            context["archived_target_weights"] = archived_weights

        return SignalBundle(
            alpha=alpha,
            context=context,
            meta=meta,
        )

    def _load_archived_weights(self, *, index: pd.Index, columns: pd.Index) -> pd.DataFrame | None:
        if self.archived_weights_path is None:
            return None
        try:
            archived = pd.read_parquet(self.archived_weights_path)
        except FileNotFoundError:
            return None
        archived = archived.reindex(index=index, columns=columns)
        if not bool(archived.notna().any().any()):
            return None
        return archived.fillna(0.0).astype(float)


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


def _build_rrg_measures(
    *,
    close: pd.DataFrame,
    benchmark: pd.Series,
    sector: pd.DataFrame,
    membership: pd.DataFrame,
    market_cap: pd.DataFrame,
    medium_lookback: int,
    momentum_lookback: int,
    short_lookback: int,
) -> dict[str, pd.DataFrame]:
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
    state, _long_sector, _short_sector = _classify_rrg_states(
        relative_strength=relative_strength,
        momentum=momentum,
    )
    return {
        "state": state,
        "relative_strength": relative_strength,
        "momentum": momentum,
    }


def _build_pure_sector_budget(
    *,
    rrg_state: pd.DataFrame,
    relative_strength: pd.DataFrame,
    momentum: pd.DataFrame,
    selection_rule: str,
    weighting_rule: str,
) -> pd.DataFrame:
    selected = _select_pure_sectors(
        rrg_state=rrg_state,
        relative_strength=relative_strength,
        momentum=momentum,
        selection_rule=selection_rule,
    )
    raw_score = _pure_sector_score(
        rrg_state=rrg_state,
        relative_strength=relative_strength,
        momentum=momentum,
        weighting_rule=weighting_rule,
    )
    positive_score = raw_score.where(selected).clip(lower=0.0).fillna(0.0)
    equal_score = selected.astype(float)
    score = positive_score.copy()
    fallback_rows = score.sum(axis=1).le(0.0)
    if bool(fallback_rows.any()):
        score.loc[fallback_rows] = equal_score.loc[fallback_rows]
    score = score.where(selected, 0.0).fillna(0.0)
    denom = score.sum(axis=1).replace(0.0, np.nan)
    return score.div(denom, axis=0).fillna(0.0).astype(float)


def _select_pure_sectors(
    *,
    rrg_state: pd.DataFrame,
    relative_strength: pd.DataFrame,
    momentum: pd.DataFrame,
    selection_rule: str,
) -> pd.DataFrame:
    if selection_rule == "leading_improving":
        selected = rrg_state.isin(("Leading", "Improving"))
    elif selection_rule == "leading":
        selected = rrg_state.eq("Leading")
    elif selection_rule == "improving":
        selected = rrg_state.eq("Improving")
    elif selection_rule == "momentum_positive":
        selected = momentum.gt(0.0)
    elif selection_rule == "rs_positive":
        selected = relative_strength.gt(0.0)
    elif selection_rule == "score_positive":
        selected = relative_strength.add(momentum, fill_value=0.0).gt(0.0)
    elif selection_rule == "leading_improving_ex_weakening":
        selected = rrg_state.isin(("Leading", "Improving")) & momentum.ge(0.0)
    elif selection_rule == "leading_improving_weakening":
        selected = rrg_state.isin(("Leading", "Improving", "Weakening"))
    elif selection_rule == "leading_improving_resilient_weakening":
        resilient_weakening = rrg_state.eq("Weakening") & relative_strength.add(momentum, fill_value=0.0).gt(0.0)
        selected = rrg_state.isin(("Leading", "Improving")) | resilient_weakening
    else:
        raise ValueError(f"unsupported selection_rule: {selection_rule}")
    valid = relative_strength.notna() & momentum.notna()
    return (selected & valid).fillna(False).astype(bool)


def _pure_sector_score(
    *,
    rrg_state: pd.DataFrame,
    relative_strength: pd.DataFrame,
    momentum: pd.DataFrame,
    weighting_rule: str,
) -> pd.DataFrame:
    if weighting_rule == "equal":
        return pd.DataFrame(1.0, index=rrg_state.index, columns=rrg_state.columns)
    if weighting_rule == "score":
        return relative_strength.add(momentum, fill_value=0.0)
    if weighting_rule == "momentum":
        return momentum
    if weighting_rule == "relative_strength":
        return relative_strength
    if weighting_rule == "state_rank":
        state_score = pd.DataFrame(0.0, index=rrg_state.index, columns=rrg_state.columns, dtype=float)
        state_score = state_score.mask(rrg_state.eq("Leading"), 4.0)
        state_score = state_score.mask(rrg_state.eq("Improving"), 3.0)
        state_score = state_score.mask(rrg_state.eq("Weakening"), 2.0)
        state_score = state_score.mask(rrg_state.eq("Lagging"), 1.0)
        return state_score
    raise ValueError(f"unsupported weighting_rule: {weighting_rule}")


def _build_state_equal_sector_weight_basis(
    *,
    sector: pd.DataFrame,
    membership: pd.DataFrame,
    rrg_state: pd.DataFrame,
) -> pd.DataFrame:
    basis = pd.DataFrame(0.0, index=sector.index, columns=sector.columns, dtype=float)
    aligned_membership = membership.reindex(index=sector.index, columns=sector.columns).fillna(False).astype(bool)
    aligned_state = rrg_state.reindex(index=sector.index)
    active_states = ("Leading", "Improving")

    for ts in sector.index:
        row_sector = sector.loc[ts]
        row_membership = aligned_membership.loc[ts]
        row_state = aligned_state.loc[ts]
        present_states = [
            state_name
            for state_name in active_states
            if bool(row_state.eq(state_name).any())
        ]
        if not present_states:
            continue
        state_budget = 1.0 / len(present_states)
        for state_name in present_states:
            state_sectors = row_state[row_state.eq(state_name)].index
            state_sectors = [sector_name for sector_name in state_sectors if bool((row_sector.eq(sector_name) & row_membership).any())]
            if not state_sectors:
                continue
            sector_budget = state_budget / len(state_sectors)
            for sector_name in state_sectors:
                members = row_sector[row_sector.eq(sector_name) & row_membership].index
                if len(members) == 0:
                    continue
                basis.loc[ts, members] = sector_budget / len(members)
    return basis


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


def _build_forward_entry_mask(
    *,
    frames: dict[str, pd.DataFrame],
    index: pd.Index,
    columns: pd.Index,
    sector: pd.DataFrame,
    rrg_state: pd.DataFrame,
    lookback: int,
    entry_rule: str = "state_conditioned",
) -> pd.DataFrame:
    eps_delta, eps_count, eps_positive_count = _estimate_family_delta(
        frames=frames,
        keys=("eps_fwd_q1", "eps_fwd_q2", "eps_fwd"),
        index=index,
        columns=columns,
        sector=sector,
        lookback=lookback,
    )
    op_delta, op_count, op_positive_count = _estimate_family_delta(
        frames=frames,
        keys=("op_fwd_q1", "op_fwd_q2", "op_fwd"),
        index=index,
        columns=columns,
        sector=sector,
        lookback=lookback,
    )
    eps_positive = eps_delta.gt(0.0)
    op_positive = op_delta.gt(0.0)
    family_count = eps_delta.notna().astype(int) + op_delta.notna().astype(int)
    net_delta = eps_delta.fillna(0.0).add(op_delta.fillna(0.0)).divide(family_count.replace(0, np.nan))
    available_count = eps_count.add(op_count, fill_value=0.0)
    positive_count = eps_positive_count.add(op_positive_count, fill_value=0.0)
    state_by_symbol = _map_sector_state_to_symbols(
        sector=sector.reindex(index=index, columns=columns),
        rrg_state=rrg_state.reindex(index=index),
    )
    allowed_state = state_by_symbol.isin(("Leading", "Improving", "Weakening"))
    if entry_rule == "state_conditioned":
        leading = state_by_symbol.eq("Leading") & eps_positive & op_positive
        improving = state_by_symbol.eq("Improving") & (eps_positive | op_positive)
        weakening_survival = state_by_symbol.eq("Weakening") & (eps_positive | op_positive)
        entry = leading | improving | weakening_survival
    elif entry_rule == "dual_family":
        entry = allowed_state & eps_positive & op_positive
    elif entry_rule == "majority_horizons":
        entry = allowed_state & positive_count.gt(available_count / 2.0)
    elif entry_rule == "net_positive":
        entry = allowed_state & net_delta.gt(0.0)
    else:
        raise ValueError(f"unsupported fwd entry rule: {entry_rule}")
    return entry.fillna(False).astype(bool)


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

    valid = relative_strength.notna() & momentum.notna()
    states = pd.DataFrame("Unclassified", index=relative_strength.index, columns=relative_strength.columns, dtype=object)
    states = states.mask(leading, "Leading")
    states = states.mask(improving, "Improving")
    states = states.mask(lagging, "Lagging")
    states = states.mask(weakening, "Weakening")

    long_sector = (leading | improving) & valid
    short_sector = (lagging | weakening) & valid
    return states, long_sector.astype(bool), short_sector.astype(bool)
