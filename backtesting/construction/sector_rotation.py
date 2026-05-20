from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.signals.base import SignalBundle
from backtesting.strategy.base import validate_positive

from .base import ConstructionResult


@dataclass(slots=True)
class SectorRotationLongShort:
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
                hold_only_sector = row_hold_long_sector & ~row_long_sector.reindex(row_hold_long_sector.index).fillna(False)
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
    preselected = pd.Series(False, index=row_alpha.index, dtype=bool) if preselected is None else preselected.reindex(row_alpha.index).fillna(False).astype(bool)
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
