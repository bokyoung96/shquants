from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class LightGBMConfig:
    num_boost_round: int = 1500
    learning_rate: float = 0.01
    num_leaves: int = 15
    max_depth: int = 4
    min_data_in_leaf: int = 60
    feature_fraction: float = 0.75
    bagging_fraction: float = 0.75
    bagging_freq: int = 1
    lambda_l1: float = 0.01
    lambda_l2: float = 5.0
    validation_window: int = 252
    min_train_rows: int = 500
    early_stopping_rounds: int = 100
    seed: int = 42
    num_threads: int = 1

    def params(self) -> dict[str, Any]:
        return {
            "objective": "regression",
            "metric": "l2",
            "boosting_type": "gbdt",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "seed": self.seed,
            "num_threads": self.num_threads,
            "force_col_wise": True,
            "verbosity": -1,
        }


@dataclass(slots=True)
class LightGBMForecaster:
    config: LightGBMConfig = LightGBMConfig()
    columns_: list[str] = field(init=False)
    target_min_: float = field(init=False)
    target_max_: float = field(init=False)
    booster_: lgb.Booster = field(init=False)
    training_rows_: int = field(init=False)
    validation_rows_: int = field(init=False)
    best_iteration_: int = field(init=False)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "LightGBMForecaster":
        aligned = features.join(target.rename("__target__"), how="inner").dropna(subset=["__target__"])
        if aligned.empty:
            raise ValueError("cannot fit lightgbm forecaster without target observations")

        x = aligned.drop(columns=["__target__"]).astype(float)
        y = aligned["__target__"].astype(float)
        self.columns_ = list(x.columns)
        self.target_min_ = float(y.min())
        self.target_max_ = float(y.max())

        x_train, y_train, x_valid, y_valid = self._split_validation_tail(x, y)
        self.training_rows_ = len(x_train)
        self.validation_rows_ = len(x_valid)

        train_set = lgb.Dataset(
            x_train,
            label=y_train.to_numpy(dtype=float),
            feature_name=self.columns_,
            free_raw_data=False,
        )
        valid_sets = None
        valid_names = None
        callbacks = [lgb.log_evaluation(period=0)]
        if len(x_valid):
            valid_set = lgb.Dataset(
                x_valid,
                label=y_valid.to_numpy(dtype=float),
                feature_name=self.columns_,
                reference=train_set,
                free_raw_data=False,
            )
            valid_sets = [valid_set]
            valid_names = ["validation"]
            callbacks.append(lgb.early_stopping(self.config.early_stopping_rounds, verbose=False))

        self.booster_ = lgb.train(
            self.config.params(),
            train_set,
            num_boost_round=self.config.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self.best_iteration_ = int(self.booster_.best_iteration or self.booster_.num_trees())
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        missing = [column for column in self.columns_ if column not in features.columns]
        if missing:
            raise ValueError(f"missing feature columns for prediction: {', '.join(missing)}")

        x = features.loc[:, self.columns_].astype(float)
        prediction = self.booster_.predict(x, num_iteration=self.best_iteration_)
        prediction = np.clip(prediction, self.target_min_, self.target_max_)
        return pd.Series(prediction, index=features.index, name="prediction")

    def _split_validation_tail(
        self, features: pd.DataFrame, target: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        validation_window = int(self.config.validation_window)
        min_train_rows = int(self.config.min_train_rows)
        if validation_window <= 0 or len(features) < min_train_rows + validation_window:
            return features, target, features.iloc[0:0], target.iloc[0:0]

        train_end = len(features) - validation_window
        return (
            features.iloc[:train_end],
            target.iloc[:train_end],
            features.iloc[train_end:],
            target.iloc[train_end:],
        )
