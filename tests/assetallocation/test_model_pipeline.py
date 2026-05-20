import numpy as np
import pandas as pd

from assetallocation.model.allocation import rebalance_weekly, score_to_two_asset_weights
from assetallocation.model.lightgbm_model import LightGBMConfig, LightGBMForecaster
from assetallocation.model.ridge import RidgeForecaster
from assetallocation.model.runner import RidgeAllocationRunner, default_runner
from assetallocation.model.tft_dataset import make_supervised_sequences
from assetallocation.model.tft_model import TFTConfig, TemporalFusionTransformerForecaster
from assetallocation.model.walkforward import WalkForwardConfig, walk_forward_predict, walk_forward_predict_with_model


def test_ridge_forecaster_learns_linear_signal_with_missing_values() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="D", name="date")
    features = pd.DataFrame(
        {
            "signal": [0.0, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0],
            "noise": [1.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )
    target = pd.Series([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0], index=index, name="target")

    model = RidgeForecaster(alpha=0.01).fit(features.iloc[:6], target.iloc[:6])
    prediction_features = pd.DataFrame(
        {"signal": [4.5, 5.0], "noise": [1.0, 1.0]},
        index=index[6:],
    )
    prediction = model.predict(prediction_features)

    assert prediction.index.tolist() == index[6:].tolist()
    assert prediction.iloc[0] > 8.0
    assert prediction.iloc[1] >= prediction.iloc[0]


def test_ridge_forecaster_clips_predictions_to_training_target_range() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="D", name="date")
    features = pd.DataFrame({"signal": [0, 1, 2, 3, 100, -100]}, index=index, dtype=float)
    target = pd.Series([-0.02, -0.01, 0.0, 0.02], index=index[:4], name="target")

    model = RidgeForecaster(alpha=0.01).fit(features.iloc[:4], target)
    prediction = model.predict(features.iloc[4:])

    assert prediction.max() <= 0.02
    assert prediction.min() >= -0.02


def test_lightgbm_forecaster_uses_validation_tail_for_early_stopping() -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="D", name="date")
    features = pd.DataFrame(
        {
            "signal": np.linspace(-1.0, 1.0, len(index)),
            "noise": np.sin(np.arange(len(index))),
        },
        index=index,
    )
    target = pd.Series(np.linspace(-0.02, 0.02, len(index)), index=index, name="target")

    model = LightGBMForecaster(
        config=LightGBMConfig(
            num_boost_round=30,
            early_stopping_rounds=5,
            validation_window=8,
            min_train_rows=20,
            min_data_in_leaf=2,
        )
    ).fit(features, target)
    prediction = model.predict(features.tail(3))

    assert model.training_rows_ == 32
    assert model.validation_rows_ == 8
    assert 1 <= model.best_iteration_ <= 30
    assert prediction.index.tolist() == index[-3:].tolist()
    assert prediction.notna().all()


def test_make_supervised_sequences_uses_trailing_lookback_window() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D", name="date")
    features = pd.DataFrame({"signal": np.arange(5, dtype=float), "other": np.arange(10, 15, dtype=float)}, index=index)
    target = pd.Series(np.arange(100, 105, dtype=float), index=index, name="target")

    sequences = make_supervised_sequences(features, target, lookback=3)

    assert sequences.values.shape == (3, 3, 2)
    assert sequences.target_index.tolist() == index[2:].tolist()
    assert sequences.values[0, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert sequences.values[-1, :, 0].tolist() == [2.0, 3.0, 4.0]
    assert sequences.targets.tolist() == [102.0, 103.0, 104.0]


def test_tft_forecaster_predicts_with_prior_context_window() -> None:
    index = pd.date_range("2024-01-01", periods=36, freq="D", name="date")
    signal = np.linspace(-1.0, 1.0, len(index))
    features = pd.DataFrame({"signal": signal, "noise": np.cos(np.arange(len(index)))}, index=index)
    target = pd.Series(signal * 0.02, index=index, name="target")

    model = TemporalFusionTransformerForecaster(
        config=TFTConfig(
            lookback=5,
            hidden_size=8,
            num_heads=2,
            max_epochs=3,
            batch_size=8,
            validation_window=5,
            min_train_sequences=8,
            early_stopping_patience=2,
        )
    ).fit(features.iloc[:28], target.iloc[:28])
    prediction = model.predict_with_context(features.iloc[23:32], prediction_index=index[28:32])

    assert prediction.index.tolist() == index[28:32].tolist()
    assert prediction.notna().all()
    assert model.training_rows_ > 0
    assert model.best_iteration_ >= 1


def test_walk_forward_predict_trains_only_on_past_windows() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="D", name="date")
    features = pd.DataFrame({"signal": np.arange(10, dtype=float)}, index=index)
    target = pd.Series(np.arange(10, dtype=float), index=index, name="target")

    predictions = walk_forward_predict(
        features,
        target,
        config=WalkForwardConfig(train_window=4, test_window=2, step=2, alpha=0.01, purge_window=0),
    )

    assert predictions.index.tolist() == index[4:10].tolist()
    assert predictions.notna().all()
    assert predictions.loc[index[4]] < predictions.loc[index[9]]


def test_walk_forward_predict_purges_train_rows_before_test_window() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="D", name="date")
    features = pd.DataFrame({"signal": np.arange(10, dtype=float)}, index=index)
    target = pd.Series([0.0, 1.0, 2.0, 100.0, 100.0, 5.0, 6.0, 7.0, 8.0, 9.0], index=index, name="target")

    purged = walk_forward_predict(
        features,
        target,
        config=WalkForwardConfig(train_window=5, test_window=1, step=1, alpha=0.01, purge_window=2),
    )
    unpurged = walk_forward_predict(
        features,
        target,
        config=WalkForwardConfig(train_window=5, test_window=1, step=1, alpha=0.01, purge_window=0),
    )

    assert purged.index[0] == index[5]
    assert purged.iloc[0] < unpurged.iloc[0]


def test_walk_forward_result_records_is_purge_and_oos_windows() -> None:
    index = pd.date_range("2024-01-01", periods=12, freq="D", name="date")
    features = pd.DataFrame({"signal": np.arange(12, dtype=float)}, index=index)
    target = pd.Series(np.arange(12, dtype=float), index=index, name="target")

    result = walk_forward_predict_with_model(
        features,
        target,
        config=WalkForwardConfig(train_window=6, test_window=3, step=3, purge_window=2),
        model_factory=lambda: RidgeForecaster(alpha=0.01),
        prediction_name="ridge_prediction",
    )

    folds = result.folds
    assert result.predictions.name == "ridge_prediction"
    assert folds.columns.tolist() == [
        "fold",
        "train_start",
        "train_end",
        "purge_start",
        "purge_end",
        "test_start",
        "test_end",
        "train_rows",
        "purge_rows",
        "test_rows",
    ]
    assert folds.iloc[0].to_dict() == {
        "fold": 0,
        "train_start": index[0],
        "train_end": index[3],
        "purge_start": index[4],
        "purge_end": index[5],
        "test_start": index[6],
        "test_end": index[8],
        "train_rows": 4,
        "purge_rows": 2,
        "test_rows": 3,
    }
    assert folds.iloc[1]["train_start"] == index[3]
    assert folds.iloc[1]["test_start"] == index[9]


def test_walk_forward_result_records_model_training_diagnostics() -> None:
    class DiagnosticForecaster:
        def fit(self, features: pd.DataFrame, target: pd.Series) -> "DiagnosticForecaster":
            self.training_rows_ = len(features)
            self.validation_rows_ = 0
            self.best_iteration_ = 7
            return self

        def predict(self, features: pd.DataFrame) -> pd.Series:
            return pd.Series(0.0, index=features.index, name="prediction")

    index = pd.date_range("2024-01-01", periods=8, freq="D", name="date")
    features = pd.DataFrame({"signal": np.arange(8, dtype=float)}, index=index)
    target = pd.Series(np.arange(8, dtype=float), index=index, name="target")

    result = walk_forward_predict_with_model(
        features,
        target,
        config=WalkForwardConfig(train_window=4, test_window=2, step=2, purge_window=1),
        model_factory=DiagnosticForecaster,
    )

    assert result.diagnostics.loc[0, "fold"] == 0
    assert result.diagnostics.loc[0, "training_rows"] == 3
    assert result.diagnostics.loc[0, "validation_rows"] == 0
    assert result.diagnostics.loc[0, "best_iteration"] == 7


def test_score_to_two_asset_weights_clips_and_sums_to_one() -> None:
    scores = pd.Series([-0.10, 0.0, 0.10, np.nan], index=pd.date_range("2024-01-01", periods=4), name="score")

    weights = score_to_two_asset_weights(scores, scale=10.0, min_spy_weight=0.0, max_spy_weight=1.0)

    assert weights.columns.tolist() == ["SPY US Equity", "IEF US Equity"]
    assert weights.iloc[0].to_dict() == {"SPY US Equity": 0.0, "IEF US Equity": 1.0}
    assert weights.iloc[1].to_dict() == {"SPY US Equity": 0.5, "IEF US Equity": 0.5}
    assert weights.iloc[2].to_dict() == {"SPY US Equity": 1.0, "IEF US Equity": 0.0}
    assert weights.iloc[3].to_dict() == {"SPY US Equity": 0.5, "IEF US Equity": 0.5}
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_rebalance_weekly_keeps_only_weekly_weight_updates() -> None:
    index = pd.bdate_range("2024-01-01", periods=8, name="date")
    weights = pd.DataFrame(
        {
            "SPY US Equity": np.linspace(0.1, 0.8, len(index)),
            "IEF US Equity": 1.0 - np.linspace(0.1, 0.8, len(index)),
        },
        index=index,
    )

    weekly = rebalance_weekly(weights)

    pd.testing.assert_series_equal(weekly.iloc[0], weights.iloc[0], check_names=False)
    pd.testing.assert_series_equal(weekly.loc["2024-01-02"], weights.iloc[0], check_names=False)
    pd.testing.assert_series_equal(weekly.loc["2024-01-05"], weights.loc["2024-01-05"], check_names=False)
    pd.testing.assert_series_equal(weekly.loc["2024-01-08"], weights.loc["2024-01-05"], check_names=False)


def test_ridge_allocation_runner_writes_prediction_weight_and_metric_files(tmp_path) -> None:
    index = pd.date_range("2024-01-01", periods=12, freq="D", name="date")
    features = pd.DataFrame(
        {
            "signal": np.linspace(-1.0, 1.0, len(index)),
            "spy_ret_1d": [0.0, 0.01, 0.02, -0.01, 0.01, 0.02, -0.01, 0.01, 0.02, -0.01, 0.01, 0.02],
            "ief_ret_1d": [0.0, 0.002, 0.001, 0.003, 0.001, 0.002, 0.001, 0.003, 0.001, 0.002, 0.001, 0.003],
        },
        index=index,
    )
    targets = pd.DataFrame(
        {"target_spy_excess_ief_fwd_20d": np.linspace(-0.03, 0.03, len(index))},
        index=index,
    )
    feature_dir = tmp_path / "feature"
    output_dir = tmp_path / "runs"
    feature_dir.mkdir()
    features.to_parquet(feature_dir / "features.parquet", engine="pyarrow")
    targets.to_parquet(feature_dir / "targets.parquet", engine="pyarrow")

    result = RidgeAllocationRunner(
        feature_dir=feature_dir,
        output_dir=output_dir,
        config=WalkForwardConfig(train_window=6, test_window=3, step=3, alpha=0.01, purge_window=1),
    ).run()

    assert result.prediction_path.exists()
    assert result.weight_path.exists()
    assert result.fold_path.exists()
    assert result.diagnostic_path.exists()
    assert result.backtest.metrics_path.exists()
    assert result.backtest.equity_plot_path.exists()
    assert result.evaluation.metrics["rebalance_frequency"] == "W-FRI"
    assert result.evaluation.metrics["purge_window"] == 1
    assert "oos_start" in result.evaluation.metrics
    assert "oos_end" in result.evaluation.metrics
    assert result.evaluation.metrics["observations"] > 0


def test_lightgbm_allocation_runner_writes_oos_artifacts_with_fold_boundaries(tmp_path) -> None:
    from assetallocation.model.lightgbm_runner import LightGBMAllocationRunner, LightGBMConfig

    index = pd.date_range("2024-01-01", periods=18, freq="D", name="date")
    features = pd.DataFrame(
        {
            "signal": np.linspace(-1.0, 1.0, len(index)),
            "spy_ret_1d": [0.001, 0.01, -0.002, 0.003, 0.004, -0.005] * 3,
            "ief_ret_1d": [0.0, 0.002, 0.001, -0.001, 0.001, 0.002] * 3,
        },
        index=index,
    )
    targets = pd.DataFrame(
        {"target_spy_excess_ief_fwd_20d": np.linspace(-0.02, 0.02, len(index))},
        index=index,
    )
    feature_dir = tmp_path / "feature"
    output_dir = tmp_path / "lightgbm"
    feature_dir.mkdir()
    features.to_parquet(feature_dir / "features.parquet", engine="pyarrow")
    targets.to_parquet(feature_dir / "targets.parquet", engine="pyarrow")

    result = LightGBMAllocationRunner(
        feature_dir=feature_dir,
        output_dir=output_dir,
        walk_forward=WalkForwardConfig(train_window=8, test_window=4, step=4, purge_window=2),
        lightgbm=LightGBMConfig(num_boost_round=3, min_data_in_leaf=1),
    ).run()

    folds = pd.read_parquet(result.fold_path, engine="pyarrow")
    assert result.prediction_path.name == "lightgbm_predictions.parquet"
    assert result.weight_path.name == "lightgbm_weights.parquet"
    assert result.fold_path.name == "folds.parquet"
    assert result.prediction_path.exists()
    assert result.weight_path.exists()
    assert result.diagnostic_path.exists()
    assert result.backtest.metrics_path.exists()
    diagnostics = pd.read_parquet(result.diagnostic_path, engine="pyarrow")
    assert folds.loc[0, "train_end"] < folds.loc[0, "purge_start"]
    assert folds.loc[0, "purge_end"] < folds.loc[0, "test_start"]
    assert diagnostics.loc[0, "training_rows"] > 0
    assert result.evaluation.metrics["model"] == "lightgbm"
    assert result.evaluation.metrics["oos_start"] == index[8].date().isoformat()


def test_tft_allocation_runner_writes_sequence_oos_artifacts(tmp_path) -> None:
    from assetallocation.model.tft_runner import TFTAllocationRunner

    index = pd.date_range("2024-01-01", periods=36, freq="D", name="date")
    features = pd.DataFrame(
        {
            "signal": np.linspace(-1.0, 1.0, len(index)),
            "spy_ret_1d": [0.001, 0.01, -0.002, 0.003, 0.004, -0.005] * 6,
            "ief_ret_1d": [0.0, 0.002, 0.001, -0.001, 0.001, 0.002] * 6,
        },
        index=index,
    )
    targets = pd.DataFrame(
        {"target_spy_excess_ief_fwd_20d": np.linspace(-0.02, 0.02, len(index))},
        index=index,
    )
    feature_dir = tmp_path / "feature"
    output_dir = tmp_path / "tft"
    feature_dir.mkdir()
    features.to_parquet(feature_dir / "features.parquet", engine="pyarrow")
    targets.to_parquet(feature_dir / "targets.parquet", engine="pyarrow")

    result = TFTAllocationRunner(
        feature_dir=feature_dir,
        output_dir=output_dir,
        walk_forward=WalkForwardConfig(train_window=18, test_window=6, step=6, purge_window=2),
        tft=TFTConfig(
            lookback=5,
            hidden_size=8,
            num_heads=2,
            max_epochs=2,
            batch_size=8,
            validation_window=4,
            min_train_sequences=6,
            early_stopping_patience=1,
        ),
    ).run()

    folds = pd.read_parquet(result.fold_path, engine="pyarrow")
    diagnostics = pd.read_parquet(result.diagnostic_path, engine="pyarrow")
    assert result.prediction_path.name == "tft_predictions.parquet"
    assert result.weight_path.name == "tft_weights.parquet"
    assert result.prediction_path.exists()
    assert result.backtest.summary_plot_path.exists()
    assert folds.loc[0, "test_start"] == index[18]
    assert diagnostics.loc[0, "lookback"] == 5
    assert result.evaluation.metrics["model"] == "tft"


def test_default_runner_writes_under_assetallocation_results() -> None:
    runner = default_runner()

    assert runner.output_dir.name == "ridge"
    assert runner.output_dir.parent.name == "results"
    assert runner.output_dir.parent.parent.name == "assetallocation"
