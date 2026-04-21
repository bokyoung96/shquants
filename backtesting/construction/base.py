from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True, slots=True)
class ConstructionResult:
    base_target_weights: pd.DataFrame
    selection_mask: pd.DataFrame
    group_long_budget: pd.DataFrame | None
    group_short_budget: pd.DataFrame | None
    meta: dict[str, pd.DataFrame | pd.Series] = field(default_factory=dict)
