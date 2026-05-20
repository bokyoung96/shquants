# assetallocation Model Map

## Scope

This package is intentionally separate from the repository-wide `backtesting` module.

The existing `backtesting` package is a broader strategy framework with catalog, execution, and Korean-market assumptions. The first assetallocation model is a small two-asset research loop for:

- `SPY US Equity`
- `IEF US Equity`

## Model Stack

The first model is a dependency-light ridge regression walk-forward baseline.
The next model is a LightGBM gradient boosting experiment using the same
purged rolling IS/OOS split.
The sequence model is a compact PyTorch temporal transformer/TFT-style
experiment that consumes trailing feature windows instead of a single feature row.

Data flow:

```text
assetallocation/feature/features.parquet
assetallocation/feature/targets.parquet
        |
        v
purged rolling walk-forward model
        |
        v
target_spy_excess_ief_fwd_20d prediction
        |
        v
SPY / IEF weights
        |
        v
assetallocation/backtesting report
```

## Modules

- `ridge.py`: standardized ridge regression with median imputation.
- `lightgbm_model.py`: native LightGBM regression wrapper with validation-tail early stopping.
- `tft_dataset.py`: builds trailing lookback windows for sequence training and prediction.
- `tft_model.py`: compact PyTorch temporal transformer forecaster.
- `walkforward.py`: rolling train/test prediction windows with explicit IS, purge, and OOS fold records.
- `allocation.py`: converts predicted SPY excess return scores into two-asset weights.
- `runner.py`: wires Ridge, allocation, and the assetallocation backtesting report.
- `lightgbm_runner.py`: wires LightGBM, allocation, and the assetallocation backtesting report.
- `tft_runner.py`: wires the sequence model, allocation, and the assetallocation backtesting report.

## Rolling IS/OOS Contract

Default model runs use:

- IS train window: previous 1,260 rows, roughly five trading years.
- Purge window: final 20 rows immediately before the OOS test block are removed from training.
- OOS test window: next 252 rows, roughly one trading year.
- Step: 252 rows.

For each fold, the model trains only on the IS rows ending before the purge
block. It then predicts the following OOS block. `folds.parquet` records:

- `train_start`, `train_end`
- `purge_start`, `purge_end`
- `test_start`, `test_end`
- row counts for each segment

LightGBM additionally splits the post-purge IS rows into:

- sub-train rows: first 988 rows by default
- validation rows: final 252 IS rows by default

It can search up to 1,500 boosting rounds, but early stopping selects the
fold-specific `best_iteration` without using OOS data.
`training_diagnostics.parquet` records the sub-train rows, validation rows,
and selected best iteration by fold.

TFT uses the same outer fold dates, but prediction is sequence-aware:

- default lookback: 252 rows ending at the decision date
- target: `target_spy_excess_ief_fwd_20d`
- target timing: features through `t`, realized outcome from `t + 1` to `t + 21`
- validation: final sequence block inside the post-purge IS window
- OOS rows are not used for training or early stopping

## Backtesting Path

Model modules stop at final weight generation. Performance and graphs are produced only through `assetallocation/backtesting`:

- `assetallocation/backtesting/engine.py`: applies model weights to next-day SPY/IEF returns.
- `assetallocation/backtesting/report.py`: writes performance tables and graph files.
- `assetallocation/backtesting/plots.py`: creates equity curve, drawdown, and weight charts.

Current ridge output path:

```text
assetallocation/results/ridge/ridge_predictions.parquet
assetallocation/results/ridge/ridge_weights.parquet
assetallocation/results/ridge/folds.parquet
assetallocation/results/ridge/training_diagnostics.parquet
assetallocation/results/ridge/backtests/ridge/performance.json
assetallocation/results/ridge/backtests/ridge/returns.parquet
assetallocation/results/ridge/backtests/ridge/equity.parquet
assetallocation/results/ridge/backtests/ridge/drawdown.parquet
assetallocation/results/ridge/backtests/ridge/weights.parquet
assetallocation/results/ridge/backtests/ridge/equity_curve.png
assetallocation/results/ridge/backtests/ridge/drawdown.png
assetallocation/results/ridge/backtests/ridge/weights.png
```

Current LightGBM output path:

```text
assetallocation/results/lightgbm/lightgbm_predictions.parquet
assetallocation/results/lightgbm/lightgbm_weights.parquet
assetallocation/results/lightgbm/folds.parquet
assetallocation/results/lightgbm/training_diagnostics.parquet
assetallocation/results/lightgbm/backtests/lightgbm/performance.json
assetallocation/results/lightgbm/backtests/lightgbm/returns.parquet
assetallocation/results/lightgbm/backtests/lightgbm/equity.parquet
assetallocation/results/lightgbm/backtests/lightgbm/drawdown.parquet
assetallocation/results/lightgbm/backtests/lightgbm/weights.parquet
assetallocation/results/lightgbm/backtests/lightgbm/summary.png
```

Current TFT output path:

```text
assetallocation/results/tft/tft_predictions.parquet
assetallocation/results/tft/tft_weights.parquet
assetallocation/results/tft/folds.parquet
assetallocation/results/tft/training_diagnostics.parquet
assetallocation/results/tft/backtests/tft/performance.json
assetallocation/results/tft/backtests/tft/returns.parquet
assetallocation/results/tft/backtests/tft/equity.parquet
assetallocation/results/tft/backtests/tft/drawdown.parquet
assetallocation/results/tft/backtests/tft/weights.parquet
assetallocation/results/tft/backtests/tft/summary.png
```

## macOS Execution

From a MacBook checkout:

```bash
uv sync
uv run python -m assetallocation.feature.builder
uv run python -m assetallocation.model.tft_runner
```

The PyTorch dependency is declared as `torch>=2.3,<3` without a Windows-only
or CPU-only package index, so macOS should resolve the appropriate PyTorch
wheel through the normal package resolver.

## Current Limitations

- `USGG10YR Index` is now a rate-context feature; the tradable bond leg is `IEF US Equity`.
- ETF source fields should be reviewed before production use because unadjusted `PX_LAST` does not fully capture distributions.
- The assetallocation backtest is deliberately simple and applies weights to next-day returns.
- Ridge and LightGBM are still tabular ML baselines before adding TFT sequence modeling.
