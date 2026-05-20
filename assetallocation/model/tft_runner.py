from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from assetallocation.backtesting.engine import BacktestResult, TwoAssetBacktester
from assetallocation.backtesting.report import BacktestReportWriter

from .allocation import rebalance_weekly, score_to_two_asset_weights
from .runner import DEFAULT_TARGET, ModelRunResult
from .tft_model import TFTConfig, TemporalFusionTransformerForecaster
from .walkforward import WalkForwardConfig


@dataclass(slots=True)
class TFTAllocationRunner:
    feature_dir: Path
    output_dir: Path
    target_column: str = DEFAULT_TARGET
    walk_forward: WalkForwardConfig = WalkForwardConfig()
    tft: TFTConfig = TFTConfig()

    def run(self) -> ModelRunResult:
        self.feature_dir = Path(self.feature_dir)
        self.output_dir = Path(self.output_dir)
        features = pd.read_parquet(self.feature_dir / "features.parquet", engine="pyarrow")
        targets = pd.read_parquet(self.feature_dir / "targets.parquet", engine="pyarrow")
        if self.target_column not in targets.columns:
            raise ValueError(f"missing target column: {self.target_column}")

        target = targets[self.target_column]
        predictions, folds, diagnostics = self._walk_forward_predict(features, target)
        weights = rebalance_weekly(score_to_two_asset_weights(predictions))
        daily_returns = features.loc[:, ["spy_ret_1d", "ief_ret_1d"]].rename(
            columns={"spy_ret_1d": "SPY US Equity", "ief_ret_1d": "IEF US Equity"}
        )

        evaluation = TwoAssetBacktester().run(weights, daily_returns)
        self._attach_metrics(evaluation, predictions)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = self.output_dir / "tft_predictions.parquet"
        weight_path = self.output_dir / "tft_weights.parquet"
        fold_path = self.output_dir / "folds.parquet"
        diagnostic_path = self.output_dir / "training_diagnostics.parquet"
        predictions.to_frame().to_parquet(prediction_path, engine="pyarrow")
        weights.to_parquet(weight_path, engine="pyarrow")
        folds.to_parquet(fold_path, engine="pyarrow", index=False)
        diagnostics.to_parquet(diagnostic_path, engine="pyarrow", index=False)
        backtest = BacktestReportWriter(self.output_dir / "backtests").write("tft", evaluation)

        return ModelRunResult(prediction_path, weight_path, fold_path, diagnostic_path, backtest, evaluation)

    def _walk_forward_predict(
        self, features: pd.DataFrame, target: pd.Series
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        config = self.walk_forward
        if config.train_window <= 0 or config.test_window <= 0 or config.step <= 0:
            raise ValueError("walk-forward windows must be positive")
        if config.purge_window < 0:
            raise ValueError("purge_window must not be negative")
        if not features.index.equals(target.index):
            raise ValueError("features and target must share the same index")

        predictions: list[pd.Series] = []
        fold_rows: list[dict[str, object]] = []
        diagnostic_rows: list[dict[str, object]] = []
        start = config.train_window
        fold = 0
        while start < len(features):
            train_end = start - config.purge_window
            if train_end <= start - config.train_window:
                raise ValueError("purge_window leaves no train observations")
            test_end = min(start + config.test_window, len(features))
            train_slice = slice(start - config.train_window, train_end)
            test_index = features.index[start:test_end]
            context_start = max(0, start - self.tft.lookback + 1)
            context = features.iloc[context_start:test_end]

            model = TemporalFusionTransformerForecaster(config=self.tft).fit(
                features.iloc[train_slice],
                target.iloc[train_slice],
            )
            predictions.append(model.predict_with_context(context, test_index).rename("tft_prediction"))
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
                    "lookback": self.tft.lookback,
                    "training_rows": model.training_rows_,
                    "validation_rows": model.validation_rows_,
                    "best_iteration": model.best_iteration_,
                    "best_validation_loss": model.best_validation_loss_,
                }
            )
            fold += 1
            start += config.step

        if predictions:
            prediction = pd.concat(predictions).sort_index().rename("tft_prediction")
        else:
            prediction = pd.Series(dtype=float, name="tft_prediction")
        return prediction, pd.DataFrame(fold_rows), pd.DataFrame(diagnostic_rows)

    def _attach_metrics(self, evaluation: BacktestResult, predictions: pd.Series) -> None:
        evaluation.metrics["model"] = "tft"
        evaluation.metrics["rebalance_frequency"] = "W-FRI"
        evaluation.metrics["purge_window"] = float(self.walk_forward.purge_window)
        evaluation.metrics["train_window"] = float(self.walk_forward.train_window)
        evaluation.metrics["test_window"] = float(self.walk_forward.test_window)
        evaluation.metrics["oos_start"] = predictions.index.min().date().isoformat() if len(predictions) else None
        evaluation.metrics["oos_end"] = predictions.index.max().date().isoformat() if len(predictions) else None
        evaluation.metrics["tft_params"] = asdict(self.tft)


def default_runner() -> TFTAllocationRunner:
    root = Path(__file__).resolve().parents[1]
    return TFTAllocationRunner(
        feature_dir=root / "feature",
        output_dir=root / "results" / "tft",
    )


def main() -> None:
    result = default_runner().run()
    print(f"predictions={result.prediction_path}")
    print(f"weights={result.weight_path}")
    print(f"folds={result.fold_path}")
    print(f"diagnostics={result.diagnostic_path}")
    print(f"metrics={result.backtest.metrics_path}")
    print(f"equity_plot={result.backtest.equity_plot_path}")
    print(f"drawdown_plot={result.backtest.drawdown_plot_path}")
    print(f"weights_plot={result.backtest.weights_plot_path}")
    for key, value in result.evaluation.metrics.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
