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
        alpha = bundle.alpha.astype(float)
        sector = bundle.context["sector"]
        group_id = (
            sector.reindex(index=alpha.index, columns=alpha.columns)
            if isinstance(sector, pd.DataFrame)
            else pd.DataFrame(index=alpha.index, columns=alpha.columns)
        )
        valid_alpha = alpha.where(group_id.notna())
        stacked_alpha = valid_alpha.stack()
        if stacked_alpha.empty:
            empty_weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns, dtype=float)
            empty_selection = pd.DataFrame(False, index=alpha.index, columns=alpha.columns, dtype=bool)
            empty_budget = pd.DataFrame(index=alpha.index, dtype=float)
            return ConstructionResult(
                base_target_weights=empty_weights,
                selection_mask=empty_selection,
                group_long_budget=empty_budget,
                group_short_budget=empty_budget.copy(),
                meta={
                    "selected_long": empty_selection.copy(),
                    "selected_short": empty_selection.copy(),
                    "group_id": group_id,
                    "group_long_budget": empty_budget,
                    "group_short_budget": empty_budget.copy(),
                },
            )

        stacked_alpha.index = stacked_alpha.index.set_names(["date", "symbol"])
        stacked_sector = group_id.stack().reindex(stacked_alpha.index)
        records = pd.DataFrame({"alpha": stacked_alpha, "sector": stacked_sector})
        grouped = records.groupby(
            [records.index.get_level_values("date"), records["sector"]],
            sort=False,
        )
        available_count = grouped["alpha"].transform("size").astype(int)
        short_count = (available_count - 1).clip(lower=0, upper=self.bottom_n).astype(int)
        long_count = (available_count - short_count).clip(lower=0, upper=self.top_n).astype(int)
        qualified = (long_count > 0) & (short_count > 0)
        long_count = long_count.where(qualified, 0)
        short_count = short_count.where(qualified, 0)

        long_rank = grouped["alpha"].rank(method="first", ascending=False)
        short_rank = grouped["alpha"].rank(method="first", ascending=True)
        selected_long_values = long_rank.le(long_count) & qualified
        selected_short_values = short_rank.le(short_count) & qualified & ~selected_long_values

        group_summary = (
            pd.DataFrame(
                {
                    "long_count": long_count.to_numpy(),
                    "short_count": short_count.to_numpy(),
                    "qualified": qualified.to_numpy(),
                },
                index=records.index,
            )
            .groupby(
                [records.index.get_level_values("date"), records["sector"].to_numpy()],
                sort=False,
            )
            .first()
        )
        group_summary.index = group_summary.index.set_names(["date", "sector"])
        group_summary = group_summary[group_summary["qualified"]].copy()
        group_summary["budget"] = _group_budget_series(group_summary, self.group_budget).astype(float)

        row_group_index = pd.MultiIndex.from_arrays(
            [records.index.get_level_values("date"), records["sector"]],
            names=["date", "sector"],
        )
        row_budget = pd.Series(
            group_summary["budget"].reindex(row_group_index).to_numpy(),
            index=records.index,
            dtype=float,
        ).fillna(0.0)

        weights = pd.Series(0.0, index=records.index, dtype=float)
        long_denominator = long_count.astype(float).where(long_count.ne(0))
        short_denominator = short_count.astype(float).where(short_count.ne(0))
        weights.loc[selected_long_values] = (row_budget / long_denominator).loc[selected_long_values]
        weights.loc[selected_short_values] = -(row_budget / short_denominator).loc[selected_short_values]

        base_target_weights = (
            weights.unstack("symbol")
            .reindex(index=alpha.index, columns=alpha.columns)
            .fillna(0.0)
            .astype(float)
        )
        group_long_budget = (
            group_summary["budget"]
            .unstack("sector")
            .reindex(index=alpha.index)
            .fillna(0.0)
            .astype(float)
        )
        group_short_budget = (
            group_summary["budget"]
            .unstack("sector")
            .reindex(index=alpha.index)
            .fillna(0.0)
            .astype(float)
        )
        selected_long = (
            selected_long_values.unstack("symbol", fill_value=False)
            .reindex(index=alpha.index, columns=alpha.columns, fill_value=False)
            .astype(bool)
        )
        selected_short = (
            selected_short_values.unstack("symbol", fill_value=False)
            .reindex(index=alpha.index, columns=alpha.columns, fill_value=False)
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


def _group_budget_series(group_summary: pd.DataFrame, group_budget: str) -> pd.Series:
    if group_summary.empty:
        return pd.Series(dtype=float)
    if group_budget == "equal_group":
        sector_count = group_summary.groupby(level="date")["qualified"].transform("sum")
        return 1.0 / sector_count.astype(float)
    if group_budget == "proportional_selected":
        selected_count = group_summary["long_count"] + group_summary["short_count"]
        total_selected = selected_count.groupby(level="date").transform("sum")
        return selected_count.astype(float) / total_selected.astype(float)
    raise ValueError(f"unsupported group_budget: {group_budget}")
