from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.data import MarketData
from backtesting.policy.base import BUCKET_LEDGER_COLUMNS, PositionPlan
from backtesting.signals.base import SignalBundle


@dataclass(frozen=True, slots=True)
class RegimeFilterPolicy:
    regime_key: str
    off_target_weight: float = 0.0
    off_exposure_multiplier: float | None = None

    def apply(self, weights: pd.DataFrame | None = None, *, construction=None, market: MarketData, bundle: SignalBundle) -> PositionPlan:
        if weights is None:
            if construction is None:
                raise ValueError("RegimeFilterPolicy.apply requires weights or construction")
            weights = construction.base_target_weights
        weights = weights.fillna(0.0).astype(float)
        regime = bundle.context[self.regime_key].reindex_like(weights).fillna(False).astype(bool)
        if self.off_exposure_multiplier is not None:
            filtered = weights.where(regime, weights * self.off_exposure_multiplier)
        else:
            filtered = weights.where(regime, self.off_target_weight)
        validation = {
            "regime_key": self.regime_key,
            "active_ratio": float(regime.all(axis=1).mean()) if not regime.empty else 0.0,
        }
        records: list[dict[str, object]] = []
        for date, row in filtered.iterrows():
            active = row[row.ne(0.0)]
            for symbol, value in active.items():
                records.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "side": "long" if value > 0.0 else "short",
                        "bucket_id": "regime_filtered",
                        "stage_index": 0,
                        "target_weight": float(value),
                        "actual_weight": float(value),
                        "target_qty": 0.0,
                        "actual_qty": 0.0,
                        "entry_price": None,
                        "mark_price": None,
                        "bucket_return": 0.0,
                        "state": "active",
                        "event": "regime_filter",
                        "construction_group": None,
                        "budget_id": "base",
                    }
                )
        ledger = pd.DataFrame.from_records(records, columns=BUCKET_LEDGER_COLUMNS)
        return PositionPlan(
            target_weights=filtered,
            bucket_ledger=ledger,
            bucket_meta={"policy_name": pd.Series(["regime_filter"], name="policy_name")},
            validation=validation,
        )
