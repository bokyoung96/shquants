from __future__ import annotations

import pandas as pd

from backtesting.execution.schedule import CustomSchedule, DailySchedule, MonthlySchedule, WeeklySchedule

from .models import ExecutionSpec, ScheduleEvaluationSpec


def apply_scheduled_evaluation(
    spec: ExecutionSpec,
    *,
    base_weights: pd.DataFrame,
    selection: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = spec.schedule
    if schedule.kind == "signal_dates":
        if schedule.evaluation is not None:
            return _freeze_to_flags(
                base_weights=base_weights,
                selection=selection,
                flags=_evaluation_flags(schedule.evaluation, base_weights.index),
            )
        return base_weights, selection
    if schedule.kind == "named" and schedule.name == "monthly" and not schedule.evaluate_on_schedule:
        return base_weights, selection

    flags = schedule_flags(schedule, base_weights.index)
    if flags is None:
        return base_weights, selection

    return _freeze_to_flags(base_weights=base_weights, selection=selection, flags=flags)


def schedule_flags(schedule, index: pd.Index) -> pd.Series | None:
    datetime_index = pd.DatetimeIndex(index)
    if schedule.kind == "named":
        if schedule.name == "daily":
            return DailySchedule().flags(datetime_index)
        if schedule.name == "weekly":
            return WeeklySchedule().flags(datetime_index)
        if schedule.name == "monthly":
            return MonthlySchedule().flags(datetime_index)
        return None
    if schedule.kind == "custom_dates":
        return CustomSchedule(pd.to_datetime(list(schedule.dates))).flags(datetime_index)
    return None


def _freeze_to_flags(
    *,
    base_weights: pd.DataFrame,
    selection: pd.DataFrame,
    flags: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flags = flags.reindex(base_weights.index).fillna(False).astype(bool)
    scheduled_weights = base_weights.where(flags, float("nan")).ffill().fillna(0.0).astype(float)
    scheduled_selection = (
        selection.astype(float)
        .where(flags, float("nan"))
        .ffill()
        .fillna(0.0)
        .astype(bool)
    )
    return scheduled_weights, scheduled_selection


def _evaluation_flags(evaluation: ScheduleEvaluationSpec, index: pd.Index) -> pd.Series:
    flags = schedule_flags(evaluation, index)
    if flags is None:
        raise ValueError(f"unsupported schedule evaluation: {evaluation.kind}")
    return flags
