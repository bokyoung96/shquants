# EMP008 Research Handoff Design

## Goal

Create an additive `emp008_research/` handoff package without modifying the existing `backtesting/strategies/emp008`, experiments, tests, or data surfaces. The package must reproduce the existing EMP008 target weights and backtest results from raw inputs through parquet conversion.

## Scope and invariants

- Existing repository files remain the read-only reference implementation.
- The handoff package owns its imports and accepts explicit `data_dir` and `output_dir` paths.
- `target_weights.csv` is the stable contract between the EMP008 calculation layer and the backtest layer.
- Default factor definitions, date alignment, warmup, benchmark completion, risk model, optimizer constraints, fill mode, fees, sell tax, and slippage remain unchanged.
- Raw-to-parquet conversion is reproducible and records dataset metadata and checksums.
- Generated research outputs are separated from source tests and experiment definitions.

## Package layout

```text
emp008_research/
  emp008/       # copied strategy calculation surface
  backtest/     # target-weight execution and reporting surface
  data/         # raw inputs, parquet outputs, conversion and loading
  experiments/  # copied/reorganized research drivers and manifests
  tests/        # unit, integration, and parity tests
  scripts/      # handoff CLI entrypoints
  results/      # generated outputs, ignored by default
  README.md
```

The existing `backtesting` package is not imported by the handoff runtime. The copied modules retain the same calculations, while path and infrastructure dependencies are replaced by the handoff-local data and execution adapters.

## Data flow

```text
data/raw -> data/ingest/convert.py -> data/parquet
data/parquet -> emp008/run_weights.py -> target_weights.csv
target_weights.csv + data/parquet -> backtest/run_backtest.py -> report artifacts
```

The default MFBT dataset manifest includes `QW_ADJ_C`, `QW_BM_WEIGHTS`, `QW_OP_FWD_12M`, `QW_DPS_TTM`, `QW_RETAIL`, sector labels, market cap, float market cap, value inputs, and `QW_K200_YN`. Dataset names and stems are recorded in `data/manifest.json`.

## Parity verification

The parity suite runs the original implementation and the handoff implementation over the same parquet snapshot and configuration. It compares:

1. target-weight dates and ticker axes;
2. target and active weights within a documented numerical tolerance;
3. optimizer diagnostics and constraint values;
4. backtest equity, returns, turnover, costs, and summary metrics;
5. raw-to-parquet frame axes and values for each required dataset.

The test fixture uses a bounded date window suitable for local execution. A full-range verification command is documented separately for operators with the complete dataset snapshot.

## Handoff commands

```powershell
python data/ingest/convert.py --raw-dir data/raw --parquet-dir data/parquet
python scripts/generate_weights.py --data-dir data/parquet --output-dir results/run_001
python scripts/run_backtest.py --data-dir data/parquet --weights results/run_001/weights/target_weights.csv --output-dir results/run_001
python -m pytest tests -q
```

All commands use explicit paths and do not depend on the repository root's global parquet path.

## Risks and decisions

- Numerical parity requires the same Python dependency and SciPy versions; tolerances are used for cross-environment solver variation.
- Raw data may be licensed or sensitive. The manifest supports a parquet-only handoff when raw files cannot be redistributed.
- Full historical runs remain computationally expensive because the optimizer solves once per monthly rebalance. The parity suite therefore uses a bounded fixture while preserving a full-run command.
