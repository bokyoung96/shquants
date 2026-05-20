from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RidgeForecaster:
    alpha: float = 1.0

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "RidgeForecaster":
        aligned = features.join(target.rename("__target__"), how="inner")
        aligned = aligned.dropna(subset=["__target__"])
        if aligned.empty:
            raise ValueError("cannot fit ridge forecaster without target observations")

        x = aligned.drop(columns=["__target__"]).astype(float)
        y = aligned["__target__"].astype(float)
        self.columns_ = list(x.columns)
        self.fill_ = x.median().fillna(0.0)
        x = x.fillna(self.fill_)
        self.mean_ = x.mean()
        self.std_ = x.std(ddof=0).replace(0.0, 1.0)
        x_values = ((x - self.mean_) / self.std_).to_numpy(dtype=float)
        y_values = y.to_numpy(dtype=float)
        self.target_min_ = float(y.min())
        self.target_max_ = float(y.max())

        design = np.column_stack([np.ones(len(x_values)), x_values])
        penalty = np.eye(design.shape[1]) * float(self.alpha)
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(design.T @ design + penalty, design.T @ y_values)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        missing = [column for column in self.columns_ if column not in features.columns]
        if missing:
            raise ValueError(f"missing feature columns for prediction: {', '.join(missing)}")

        x = features.loc[:, self.columns_].astype(float).fillna(self.fill_)
        x_values = ((x - self.mean_) / self.std_).to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x_values)), x_values])
        prediction = np.clip(design @ self.coef_, self.target_min_, self.target_max_)
        return pd.Series(prediction, index=features.index, name="prediction")
