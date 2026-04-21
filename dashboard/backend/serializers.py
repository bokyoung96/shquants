from __future__ import annotations

import math

import pandas as pd

from dashboard.backend.schemas import (
    CategorySeriesModel,
    CategoryPointModel,
    DistributionBinModel,
    DrawdownEpisodeModel,
    HeatmapCellModel,
    HoldingModel,
    HoldingPerformanceModel,
    NamedSeriesModel,
    ValuePointModel,
)


def sanitize_finite_number(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def serialize_value_points(series: pd.Series) -> list[ValuePointModel]:
    points: list[ValuePointModel] = []
    for date, value in series.items():
        numeric = sanitize_finite_number(value)
        if numeric is None:
            continue
        points.append(
            ValuePointModel(
                date=pd.Timestamp(date).date().isoformat(),
                value=numeric,
            )
        )
    return points


def serialize_named_series(series: pd.Series, *, run_id: str, label: str) -> NamedSeriesModel:
    return NamedSeriesModel(
        run_id=run_id,
        label=label,
        points=serialize_value_points(series),
    )


def serialize_latest_holdings(frame: pd.DataFrame | None) -> list[HoldingModel]:
    if frame is None or frame.empty:
        return []
    holdings: list[HoldingModel] = []
    for _, row in frame.iterrows():
        target_weight = sanitize_finite_number(row["target_weight"])
        abs_weight = sanitize_finite_number(row["abs_weight"])
        if target_weight is None or abs_weight is None:
            continue
        holdings.append(
            HoldingModel(
                symbol=str(row["symbol"]),
                target_weight=target_weight,
                abs_weight=abs_weight,
            )
        )
    return holdings


def serialize_latest_holdings_performance(frame: pd.DataFrame | None) -> list[HoldingPerformanceModel]:
    if frame is None or frame.empty:
        return []
    holdings: list[HoldingPerformanceModel] = []
    for _, row in frame.iterrows():
        target_weight = sanitize_finite_number(row["target_weight"])
        abs_weight = sanitize_finite_number(row["abs_weight"])
        return_since_latest_rebalance = sanitize_finite_number(row["return_since_latest_rebalance"])
        if target_weight is None or abs_weight is None or return_since_latest_rebalance is None:
            continue
        holdings.append(
            HoldingPerformanceModel(
                symbol=str(row["symbol"]),
                target_weight=target_weight,
                abs_weight=abs_weight,
                return_since_latest_rebalance=return_since_latest_rebalance,
            )
        )
    return holdings


def serialize_named_values(series: pd.Series) -> list[CategoryPointModel]:
    if series.empty:
        return []
    values: list[CategoryPointModel] = []
    for name, value in series.items():
        numeric = sanitize_finite_number(value)
        if numeric is None:
            continue
        values.append(CategoryPointModel(name=str(name), value=numeric))
    return values


def serialize_category_series(frame: pd.DataFrame) -> list[CategorySeriesModel]:
    if frame.empty:
        return []

    def _latest_abs_value(series: pd.Series) -> float:
        points = series.dropna()
        if points.empty:
            return 0.0
        return abs(float(points.iloc[-1]))

    categories: list[CategorySeriesModel] = []
    columns = sorted(frame.columns, key=lambda name: (-_latest_abs_value(frame[name]), str(name)))
    for name in columns:
        points = serialize_value_points(frame[name])
        if not points:
            continue
        categories.append(CategorySeriesModel(name=str(name), points=points))
    return categories


def serialize_heatmap(frame: pd.DataFrame) -> list[HeatmapCellModel]:
    if frame.empty:
        return []

    cells: list[HeatmapCellModel] = []
    for year in frame.index:
        for month in frame.columns:
            numeric = sanitize_finite_number(frame.loc[year, month])
            if numeric is None:
                continue
            cells.append(HeatmapCellModel(year=int(year), month=int(month), value=numeric))
    return cells


def serialize_distribution(frame: pd.DataFrame) -> list[DistributionBinModel]:
    if frame.empty:
        return []

    bins: list[DistributionBinModel] = []
    for _, row in frame.iterrows():
        start = sanitize_finite_number(row["start"])
        end = sanitize_finite_number(row["end"])
        frequency = sanitize_finite_number(row["frequency"])
        count = sanitize_finite_number(row["count"])
        if start is None or end is None or frequency is None or count is None:
            continue
        bins.append(
            DistributionBinModel(
                start=start,
                end=end,
                count=int(count),
                frequency=frequency,
            )
        )
    return bins


def serialize_drawdown_episodes(frame: pd.DataFrame) -> list[DrawdownEpisodeModel]:
    if frame.empty:
        return []

    episodes: list[DrawdownEpisodeModel] = []
    for _, row in frame.iterrows():
        drawdown = sanitize_finite_number(row["drawdown"])
        duration_days = sanitize_finite_number(row["duration_days"])
        time_to_trough_days = sanitize_finite_number(row["time_to_trough_days"])
        if drawdown is None or duration_days is None or time_to_trough_days is None:
            continue
        recovery_days = sanitize_finite_number(row.get("recovery_days"))
        episodes.append(
            DrawdownEpisodeModel(
                peak=pd.Timestamp(row["peak"]).date().isoformat(),
                start=pd.Timestamp(row["start"]).date().isoformat(),
                trough=pd.Timestamp(row["trough"]).date().isoformat(),
                end=pd.Timestamp(row["end"]).date().isoformat(),
                drawdown=drawdown,
                duration_days=int(duration_days),
                time_to_trough_days=int(time_to_trough_days),
                recovery_days=int(recovery_days) if recovery_days is not None else None,
                recovered=bool(row["recovered"]),
            )
        )
    return episodes
