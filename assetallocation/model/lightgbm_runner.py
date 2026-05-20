from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from assetallocation.backtesting.engine import TwoAssetBacktester
from assetallocation.backtesting.report import BacktestReportWriter

from .allocation import rebalance_weekly, score_to_two_asset_weights
from .lightgbm_model import LightGBMConfig, LightGBMForecaster
from .runner import DEFAULT_TARGET, ModelRunResult
from .walkforward import WalkForwardConfig, walk_forward_predict_with_model


@dataclass(slots=True)
class LightGBMAllocationRunner:
    feature_dir: Path
    output_dir: Path
    target_column: str = DEFAULT_TARGET
    walk_forward: WalkForwardConfig = WalkForwardConfig()
    lightgbm: LightGBMConfig = LightGBMConfig()

    def run(self) -> ModelRunResult:
        self.feature_dir = Path(self.feature_dir)
        self.output_dir = Path(self.output_dir)
        features = pd.read_parquet(self.feature_dir / "features.parquet", engine="pyarrow")
        targets = pd.read_parquet(self.feature_dir / "targets.parquet", engine="pyarrow")
        if self.target_column not in targets.columns:
            raise ValueError(f"missing target column: {self.target_column}")

        target = targets[self.target_column]
        walk_forward = walk_forward_predict_with_model(
            features=features,
            target=target,
            config=self.walk_forward,
            model_factory=lambda: LightGBMForecaster(config=self.lightgbm),
            prediction_name="lightgbm_prediction",
        )
        predictions = walk_forward.predictions
        weights = rebalance_weekly(score_to_two_asset_weights(predictions))
        daily_returns = features.loc[:, ["spy_ret_1d", "ief_ret_1d"]].rename(
            columns={"spy_ret_1d": "SPY US Equity", "ief_ret_1d": "IEF US Equity"}
        )

        evaluation = TwoAssetBacktester().run(weights, daily_returns)
        evaluation.metrics["model"] = "lightgbm"
        evaluation.metrics["rebalance_frequency"] = "W-FRI"
        evaluation.metrics["purge_window"] = float(self.walk_forward.purge_window)
        evaluation.metrics["train_window"] = float(self.walk_forward.train_window)
        evaluation.metrics["test_window"] = float(self.walk_forward.test_window)
        evaluation.metrics["oos_start"] = predictions.index.min().date().isoformat() if len(predictions) else None
        evaluation.metrics["oos_end"] = predictions.index.max().date().isoformat() if len(predictions) else None
        evaluation.metrics["num_boost_round"] = float(self.lightgbm.num_boost_round)
        evaluation.metrics["lightgbm_params"] = asdict(self.lightgbm)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = self.output_dir / "lightgbm_predictions.parquet"
        weight_path = self.output_dir / "lightgbm_weights.parquet"
        fold_path = self.output_dir / "folds.parquet"
        diagnostic_path = self.output_dir / "training_diagnostics.parquet"
        predictions.to_frame().to_parquet(prediction_path, engine="pyarrow")
        weights.to_parquet(weight_path, engine="pyarrow")
        walk_forward.folds.to_parquet(fold_path, engine="pyarrow", index=False)
        walk_forward.diagnostics.to_parquet(diagnostic_path, engine="pyarrow", index=False)
        backtest = BacktestReportWriter(self.output_dir / "backtests").write("lightgbm", evaluation)

        return ModelRunResult(prediction_path, weight_path, fold_path, diagnostic_path, backtest, evaluation)


def default_runner() -> LightGBMAllocationRunner:
    root = Path(__file__).resolve().parents[1]
    return LightGBMAllocationRunner(
        feature_dir=root / "feature",
        output_dir=root / "results" / "lightgbm",
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
