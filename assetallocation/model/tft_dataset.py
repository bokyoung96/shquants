from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class SequenceArrays:
    values: np.ndarray
    targets: np.ndarray
    target_index: pd.DatetimeIndex
    feature_columns: list[str]


def make_supervised_sequences(features: pd.DataFrame, target: pd.Series, lookback: int) -> SequenceArrays:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if not features.index.equals(target.index):
        raise ValueError("features and target must share the same index")

    aligned = features.join(target.rename("__target__"), how="inner").dropna(subset=["__target__"])
    if len(aligned) < lookback:
        return SequenceArrays(
            values=np.empty((0, lookback, features.shape[1]), dtype=np.float32),
            targets=np.empty((0,), dtype=np.float32),
            target_index=pd.DatetimeIndex([], name=features.index.name),
            feature_columns=list(features.columns),
        )

    x = aligned.drop(columns=["__target__"]).astype(float)
    y = aligned["__target__"].astype(float)
    values = []
    targets = []
    target_dates = []
    for end in range(lookback - 1, len(aligned)):
        start = end - lookback + 1
        values.append(x.iloc[start : end + 1].to_numpy(dtype=np.float32))
        targets.append(float(y.iloc[end]))
        target_dates.append(aligned.index[end])

    return SequenceArrays(
        values=np.asarray(values, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        target_index=pd.DatetimeIndex(target_dates, name=features.index.name),
        feature_columns=list(x.columns),
    )


def make_prediction_sequences(
    context_features: pd.DataFrame,
    prediction_index: pd.Index,
    lookback: int,
    feature_columns: list[str],
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    context = context_features.loc[:, feature_columns].astype(float)
    sequences = []
    dates = []
    for date in prediction_index:
        if date not in context.index:
            continue
        end = context.index.get_loc(date)
        if isinstance(end, slice):
            raise ValueError("context_features index must be unique")
        start = int(end) - lookback + 1
        if start < 0:
            continue
        sequences.append(context.iloc[start : int(end) + 1].to_numpy(dtype=np.float32))
        dates.append(date)
    return np.asarray(sequences, dtype=np.float32), pd.DatetimeIndex(dates, name=context_features.index.name)
