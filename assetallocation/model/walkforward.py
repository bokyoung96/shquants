from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .ridge import RidgeForecaster


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    train_window: int = 1260
    test_window: int = 252
    step: int = 252
    alpha: float = 10.0
    purge_window: int = 20


class ForecastModel(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> "ForecastModel":
        ...

    def predict(self, features: pd.DataFrame) -> pd.Series:
        ...


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    predictions: pd.Series
    folds: pd.DataFrame
    diagnostics: pd.DataFrame


def _validate_walk_forward_inputs(features: pd.DataFrame, target: pd.Series, config: WalkForwardConfig) -> None:
    if config.train_window <= 0 or config.test_window <= 0 or config.step <= 0:
        raise ValueError("walk-forward windows must be positive")
    if config.purge_window < 0:
        raise ValueError("purge_window must not be negative")
    if not features.index.equals(target.index):
        raise ValueError("features and target must share the same index")


def walk_forward_predict_with_model(
    features: pd.DataFrame,
    target: pd.Series,
    config: WalkForwardConfig,
    model_factory: Callable[[], ForecastModel],
    prediction_name: str = "prediction",
) -> WalkForwardResult:
    _validate_walk_forward_inputs(features, target, config)

    predictions: list[pd.Series] = []
    fold_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    start = config.train_window
    fold = 0
    while start < len(features):
        train_end = start - config.purge_window
        if train_end <= start - config.train_window:
            raise ValueError("purge_window leaves no train observations")
        train_slice = slice(start - config.train_window, train_end)
        test_end = min(start + config.test_window, len(features))
        test_slice = slice(start, test_end)

        model = model_factory().fit(features.iloc[train_slice], target.iloc[train_slice])
        predictions.append(model.predict(features.iloc[test_slice]).rename(prediction_name))
        fold_rows.append(
            {
                "fold": fold,
                "train_start": features.index[start - config.train_window],
                "train_end": features.index[train_end - 1],
                "purge_start": features.index[train_end] if config.purge_window else pd.NaT,
                "purge_end": features.index[start - 1] if config.purge_window else pd.NaT,
                "test_start": features.index[start],
                "test_end": features.index[test_end - 1],
                "train_rows": train_end - (start - config.train_window),
                "purge_rows": config.purge_window,
                "test_rows": test_end - start,
            }
        )
        diagnostic_rows.append(
            {
                "fold": fold,
                "training_rows": getattr(model, "training_rows_", train_end - (start - config.train_window)),
                "validation_rows": getattr(model, "validation_rows_", 0),
                "best_iteration": getattr(model, "best_iteration_", pd.NA),
            }
        )
        fold += 1
        start += config.step

    if predictions:
        prediction = pd.concat(predictions).sort_index().rename(prediction_name)
    else:
        prediction = pd.Series(dtype=float, name=prediction_name)
    return WalkForwardResult(
        predictions=prediction,
        folds=pd.DataFrame(fold_rows),
        diagnostics=pd.DataFrame(diagnostic_rows),
    )


def walk_forward_predict(features: pd.DataFrame, target: pd.Series, config: WalkForwardConfig) -> pd.Series:
    result = walk_forward_predict_with_model(
        features=features,
        target=target,
        config=config,
        model_factory=lambda: RidgeForecaster(alpha=config.alpha),
        prediction_name="prediction",
    )
    return result.predictions
