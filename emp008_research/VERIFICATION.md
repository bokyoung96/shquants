# Verification

The handoff was executed from `emp008_research` using only its local package
code and its `data/parquet` snapshot.

## Fresh local workflow

- Raw snapshot: 14 required source files under `data/raw`.
- Conversion: `scripts/convert_data.py` completed all 12 required datasets,
  including KRX benchmark-weight construction from `krx_ks200_weight.xlsx`.
- Weights: `scripts/generate_weights.py` for 2020-01-31 through 2020-03-31
  produced 3 rebalance rows × 200 symbols; all optimizer diagnostics succeeded
  and row sums were within 3e-15 of 1.0.
- Backtest: `scripts/run_backtest.py` completed 43 daily rows and produced
  final equity `83,093,624.8892673254`.
- Tests: `uv run --project . pytest tests -q` → `8 passed`.

The unified `run.py` entrypoint was then executed with its editable settings
block. It produced the same 3 × 200 weights and completed the 43-row backtest;
final equity was `83,093,624.88926733`. After adding the entrypoint tests, the
full handoff suite passed with `10 passed`.

## Reference parity

The same dates, source parquet, config, and target-weight schedule were run
through the original `backtesting.strategies.emp008` implementation. Target
weights had shape `(3, 200)` and maximum absolute difference from the handoff
was `6.16e-11` (mean absolute difference `6.00e-12`). The original backtest
final equity was `83,093,624.88926733`, matching the handoff to floating-point
precision; active-share mean was `7.198478726%` in both runs.

The original repository files were not moved or edited. The package is an
additive handoff surface; the large raw/parquet snapshot remains local and is
regenerated with the converter when transferring the package through source
control.
