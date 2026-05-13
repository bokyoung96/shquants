from __future__ import annotations

import pandas as pd
import numpy as np

from backtesting.policy.base import PositionPlan


def validate_position_plan(plan: PositionPlan, tolerance: float = 1e-8) -> None:
    target_values = _target_weight_values(plan.target_weights, tolerance=tolerance)

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


def _target_weight_values(target_weights: pd.DataFrame, tolerance: float) -> pd.Series:
    frame = target_weights.fillna(0.0).astype(float)
    dates = pd.to_datetime(frame.index)
    columns = frame.columns
    values = frame.to_numpy(dtype=float, copy=False)
    row_index, col_index = np.nonzero(np.abs(values) > float(tolerance))
    if len(row_index) == 0:
        return pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"]),
        )
    index = pd.MultiIndex.from_arrays(
        [dates.take(row_index), columns.take(col_index)],
        names=["date", "symbol"],
    )
    return pd.Series(values[row_index, col_index].astype(float, copy=False), index=index, dtype=float)
