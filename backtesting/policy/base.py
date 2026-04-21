from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from backtesting.construction.base import ConstructionResult
    from backtesting.data import MarketData
    from backtesting.signals.base import SignalBundle


BUCKET_LEDGER_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "side",
    "bucket_id",
    "stage_index",
    "target_weight",
    "actual_weight",
    "target_qty",
    "actual_qty",
    "entry_price",
    "mark_price",
    "bucket_return",
    "state",
    "event",
    "construction_group",
    "budget_id",
)


@dataclass(frozen=True, slots=True)
class PositionPlan:
    target_weights: pd.DataFrame
    bucket_ledger: pd.DataFrame
    bucket_meta: dict[str, pd.DataFrame | pd.Series] = field(default_factory=dict)
    validation: dict[str, object] = field(default_factory=dict)


class PositionPolicy(ABC):
    @abstractmethod
    def apply(
        self,
        construction: ConstructionResult,
        market: MarketData,
        bundle: SignalBundle,
    ) -> PositionPlan:
        raise NotImplementedError
