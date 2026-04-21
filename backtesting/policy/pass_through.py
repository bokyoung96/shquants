from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .base import BUCKET_LEDGER_COLUMNS, PositionPlan, PositionPolicy

if TYPE_CHECKING:
    from backtesting.construction.base import ConstructionResult
    from backtesting.data import MarketData
    from backtesting.signals.base import SignalBundle


class PassThroughPolicy(PositionPolicy):
    def apply(
        self,
        construction: ConstructionResult,
        market: MarketData,
        bundle: SignalBundle,
    ) -> PositionPlan:
        weights = construction.base_target_weights.fillna(0.0).astype(float)
        records: list[dict[str, object]] = []

        for date, row in weights.iterrows():
            active = row[row.ne(0.0)]
            for symbol, value in active.items():
                records.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "side": "long" if value > 0.0 else "short",
                        "bucket_id": "base",
                        "stage_index": 0,
                        "target_weight": float(value),
                        "actual_weight": float(value),
                        "target_qty": 0.0,
                        "actual_qty": 0.0,
                        "entry_price": None,
                        "mark_price": None,
                        "bucket_return": 0.0,
                        "state": "active",
                        "event": "pass_through",
                        "construction_group": None,
                        "budget_id": "base",
                    }
                )

        ledger = pd.DataFrame.from_records(records, columns=BUCKET_LEDGER_COLUMNS)
        return PositionPlan(
            target_weights=weights,
            bucket_ledger=ledger,
            bucket_meta={"policy_name": pd.Series(["pass_through"], name="policy_name")},
            validation={},
        )
