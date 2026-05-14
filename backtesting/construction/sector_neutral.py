from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.signals.base import SignalBundle
from backtesting.strategy.base import validate_positive

from .base import ConstructionResult


@dataclass(slots=True)
class SectorNeutralTopBottom:
    top_n: int
    bottom_n: int
    group_budget: str = "equal_group"

    def __post_init__(self) -> None:
        validate_positive("top_n", self.top_n)
        validate_positive("bottom_n", self.bottom_n)
        if self.group_budget not in {"equal_group", "proportional_selected"}:
            raise ValueError(f"unsupported group_budget: {self.group_budget}")

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        alpha = bundle.alpha
        sector = bundle.context["sector"]
        weights_by_date: dict[pd.Timestamp, pd.Series] = {}
        group_long_budget_by_date: dict[pd.Timestamp, pd.Series] = {}
        group_short_budget_by_date: dict[pd.Timestamp, pd.Series] = {}
        selected_long_by_date: dict[pd.Timestamp, pd.Series] = {}
        selected_short_by_date: dict[pd.Timestamp, pd.Series] = {}
        group_id = (
            sector.reindex(index=alpha.index, columns=alpha.columns)
            if isinstance(sector, pd.DataFrame)
            else pd.DataFrame(index=alpha.index, columns=alpha.columns)
        )

        for timestamp in alpha.index:
            weights = pd.Series(0.0, index=alpha.columns, dtype=float)
            group_long_budget = pd.Series(dtype=float)
            group_short_budget = pd.Series(dtype=float)

            sector_row = sector.loc[timestamp].dropna()
            signal = alpha.loc[timestamp].dropna().reindex(sector_row.index).dropna()
            sector_membership = sector_row.reindex(signal.index).dropna()
            qualified_sectors: list[tuple[object, pd.Index, int, int]] = []

            for sector_name, members in sector_membership.groupby(sector_membership, sort=False):
                long_count, short_count = _leg_sizes(
                    available_count=len(members.index),
                    top_n=self.top_n,
                    bottom_n=self.bottom_n,
                )
                if long_count > 0 and short_count > 0:
                    qualified_sectors.append(
                        (sector_name, members.index, long_count, short_count)
                    )

            group_budgets = _group_budgets(qualified_sectors, self.group_budget)
            for sector_name, member_index, long_count, short_count in qualified_sectors:
                sector_signal = signal.loc[member_index]
                longs = sector_signal.sort_values(ascending=False).head(long_count)
                short_pool = sector_signal.drop(index=longs.index, errors="ignore")
                shorts = short_pool.sort_values(ascending=True).head(short_count)
                group_budget = group_budgets[sector_name]

                weights.loc[longs.index] = group_budget / len(longs)
                weights.loc[shorts.index] = -group_budget / len(shorts)
                group_long_budget.loc[sector_name] = float(weights.loc[longs.index].sum())
                group_short_budget.loc[sector_name] = float(weights.loc[shorts.index].abs().sum())

            weights_by_date[timestamp] = weights
            group_long_budget_by_date[timestamp] = group_long_budget
            group_short_budget_by_date[timestamp] = group_short_budget
            selected_long_by_date[timestamp] = weights.gt(0.0)
            selected_short_by_date[timestamp] = weights.lt(0.0)

        base_target_weights = (
            pd.DataFrame.from_dict(weights_by_date, orient="index")
            .reindex(index=alpha.index, columns=alpha.columns)
            .fillna(0.0)
            .astype(float)
        )
        group_long_budget = (
            pd.DataFrame.from_dict(group_long_budget_by_date, orient="index")
            .reindex(index=alpha.index)
            .fillna(0.0)
            .astype(float)
        )
        group_short_budget = (
            pd.DataFrame.from_dict(group_short_budget_by_date, orient="index")
            .reindex(index=alpha.index)
            .fillna(0.0)
            .astype(float)
        )
        selected_long = (
            pd.DataFrame.from_dict(selected_long_by_date, orient="index")
            .reindex(index=alpha.index, columns=alpha.columns)
            .fillna(False)
            .astype(bool)
        )
        selected_short = (
            pd.DataFrame.from_dict(selected_short_by_date, orient="index")
            .reindex(index=alpha.index, columns=alpha.columns)
            .fillna(False)
            .astype(bool)
        )
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
            },
        )


def _leg_sizes(available_count: int, top_n: int, bottom_n: int) -> tuple[int, int]:
    if available_count < 2:
        return 0, 0

    short_count = min(bottom_n, available_count - 1)
    long_count = min(top_n, available_count - short_count)
    if long_count <= 0 or short_count <= 0:
        return 0, 0
    return long_count, short_count


def _group_budgets(
    qualified_sectors: list[tuple[object, pd.Index, int, int]],
    group_budget: str,
) -> dict[object, float]:
    if not qualified_sectors:
        return {}
    if group_budget == "equal_group":
        budget = 1.0 / len(qualified_sectors)
        return {sector_name: budget for sector_name, *_ in qualified_sectors}
    if group_budget == "proportional_selected":
        selected_counts = {
            sector_name: long_count + short_count
            for sector_name, _, long_count, short_count in qualified_sectors
        }
        total = float(sum(selected_counts.values()))
        return {
            sector_name: selected_count / total
            for sector_name, selected_count in selected_counts.items()
        }
    raise ValueError(f"unsupported group_budget: {group_budget}")
