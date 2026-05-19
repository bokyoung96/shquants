from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.signals.base import SignalBundle
from backtesting.strategy.base import validate_positive

from .base import ConstructionResult


@dataclass(slots=True)
class SectorRotationLongShort:
    long_count: int
    short_count: int
    gross_long: float = 1.0
    gross_short: float = 1.0
    weighting: str = "equal"

    def __post_init__(self) -> None:
        validate_positive("long_count", self.long_count)
        validate_positive("short_count", self.short_count)
        if self.gross_long < 0.0:
            raise ValueError("gross_long must be non-negative")
        if self.gross_short < 0.0:
            raise ValueError("gross_short must be non-negative")
        if self.weighting != "equal":
            raise ValueError(f"unsupported weighting: {self.weighting}")

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        alpha = bundle.alpha.astype(float)
        group_id = _required_frame(bundle, "sector").reindex(index=alpha.index, columns=alpha.columns)
        tradable = _optional_frame(bundle, "tradable", alpha.notna()).reindex(index=alpha.index, columns=alpha.columns)
        tradable = tradable.fillna(False).astype(bool)
        long_sector = _required_frame(bundle, "long_sector").reindex(index=alpha.index).fillna(False).astype(bool)
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

        for ts in alpha.index:
            row_alpha = alpha.loc[ts]
            row_group = group_id.loc[ts]
            row_tradable = tradable.loc[ts] & row_alpha.notna() & row_group.notna()
            row_basis = basis.loc[ts].where(row_tradable, 0.0)

            long_selected, long_budgets = _select_side(
                row_alpha=row_alpha,
                row_group=row_group,
                row_tradable=row_tradable,
                row_basis=row_basis,
                active_groups=long_sector.loc[ts],
                count=self.long_count,
                gross=float(self.gross_long),
                ascending=False,
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

            long_budget_rows[ts] = long_budgets
            short_budget_rows[ts] = short_budgets
            side_exposure_rows[ts] = {
                "long": float(sum(long_budgets.values())),
                "short": float(sum(short_budgets.values())),
            }

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


def _select_side(
    *,
    row_alpha: pd.Series,
    row_group: pd.Series,
    row_tradable: pd.Series,
    row_basis: pd.Series,
    active_groups: pd.Series,
    count: int,
    gross: float,
    ascending: bool,
) -> tuple[dict[object, list[object]], dict[object, float]]:
    ranked_by_group: dict[object, list[object]] = {}
    available_by_group: dict[object, int] = {}
    basis_by_group: dict[object, float] = {}
    active_names = active_groups[active_groups].index

    for group_name in active_names:
        eligible = row_tradable & row_group.eq(group_name)
        if not bool(eligible.any()):
            continue

        ranked = row_alpha.loc[eligible].sort_values(ascending=ascending, kind="stable")
        symbols = ranked.index.tolist()
        ranked_by_group[group_name] = symbols
        available_by_group[group_name] = len(symbols)
        basis_by_group[group_name] = float(row_basis.loc[eligible].sum())

    allocation = _allocate_counts(
        count=count,
        basis_by_group=basis_by_group,
        capacity_by_group=available_by_group,
    )
    if not allocation or gross == 0.0:
        return {}, {}

    selected_by_group = {
        group_name: ranked_by_group[group_name][: allocated_count]
        for group_name, allocated_count in allocation.items()
        if allocated_count > 0
    }

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
