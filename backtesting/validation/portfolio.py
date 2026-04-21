from __future__ import annotations

import pandas as pd

from backtesting.policy.base import PositionPlan


def validate_position_plan(plan: PositionPlan, tolerance: float = 1e-8) -> None:
    target_weights = plan.target_weights.fillna(0.0).astype(float).copy()
    target_weights.index = pd.to_datetime(target_weights.index)
    target_values = target_weights.stack().astype(float)
    target_values.index = target_values.index.set_names(["date", "symbol"])

    if plan.bucket_ledger.empty:
        ledger_values = pd.Series(dtype=float, index=target_values.index)
    else:
        ledger_values = (
            pd.to_numeric(plan.bucket_ledger["target_weight"], errors="coerce")
            .fillna(0.0)
            .groupby(
                [
                    pd.to_datetime(plan.bucket_ledger["date"]),
                    plan.bucket_ledger["symbol"],
                ]
            )
            .sum()
            .astype(float)
        )
        ledger_values.index = ledger_values.index.set_names(["date", "symbol"])

    all_positions = target_values.index.union(ledger_values.index)
    target_values = target_values.reindex(all_positions, fill_value=0.0)
    ledger_values = ledger_values.reindex(all_positions, fill_value=0.0)
    mismatches = target_values.sub(ledger_values).abs().gt(float(tolerance))
    if not mismatches.any():
        return

    details = ", ".join(
        (
            f"{date.date().isoformat()} {symbol}: "
            f"plan={target_values.loc[(date, symbol)]:.12g} "
            f"ledger={ledger_values.loc[(date, symbol)]:.12g}"
        )
        for date, symbol in all_positions[mismatches]
    )
    raise ValueError(f"bucket target_weight values do not match plan target_weights: {details}")
