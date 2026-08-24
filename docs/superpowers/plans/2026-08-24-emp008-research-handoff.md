# EMP008 Research Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Add an independent `emp008_research/` handoff package that converts raw data to parquet, generates EMP008 target weights, runs target-weight backtests, and proves parity with the unchanged repository implementation.

**Architecture:** Keep the current EMP008 package, experiments, tests, and data untouched as the oracle. Build a new package with explicit path arguments, local data conversion/loading, copied strategy calculations, a local target-weight backtest engine, organized experiments/tests, and parity fixtures comparing both implementations.

**Tech Stack:** Python 3.12, pandas, NumPy, SciPy, PyArrow, openpyxl, pytest.

---

### Task 1: Create handoff package boundaries and immutable source manifest

**Files:**
- Create: `emp008_research/README.md`
- Create: `emp008_research/pyproject.toml`
- Create: `emp008_research/emp008/__init__.py`
- Create: `emp008_research/backtest/__init__.py`
- Create: `emp008_research/data/__init__.py`
- Create: `emp008_research/experiments/README.md`
- Create: `emp008_research/tests/__init__.py`
- Create: `emp008_research/scripts/__init__.py`
- Create: `emp008_research/.gitignore`

- [ ] Add the directory tree and package metadata without importing the existing `backtesting` package.
- [ ] Document the three commands: raw-to-parquet conversion, weight generation, and backtest.
- [ ] Ignore generated `data/parquet/`, `results/`, and caches while keeping `data/manifest.json` tracked.
- [ ] Verify imports with `python -m compileall emp008_research`.

### Task 2: Add raw-to-parquet conversion and explicit data loading

**Files:**
- Create: `emp008_research/data/catalog.py`
- Create: `emp008_research/data/normalize.py`
- Create: `emp008_research/data/convert.py`
- Create: `emp008_research/data/loader.py`
- Create: `emp008_research/data/manifest.json`
- Create: `emp008_research/tests/unit/test_data_conversion.py`

- [ ] Copy the catalog dataset names/stems required by the default MFBT set and the special KOSPI200 benchmark-weight reader into the local catalog.
- [ ] Implement CSV/XLSX discovery, normalization, `QW_BM_WEIGHTS` parsing, parquet writing, and JSON metadata generation with explicit `raw_dir` and `parquet_dir` arguments.
- [ ] Implement `load_market_data(parquet_dir, config, start, end)` returning the local market-data bundle required by the copied factor pipeline.
- [ ] Write a failing test for normalized axes and manifest entries, run it, then implement the converter and rerun it.
- [ ] Run `pytest emp008_research/tests/unit/test_data_conversion.py -q`.

### Task 3: Copy EMP008 calculation logic behind local data interfaces

**Files:**
- Create: `emp008_research/emp008/config.py`
- Create: `emp008_research/emp008/registry.py`
- Create: `emp008_research/emp008/factor_builders.py`
- Create: `emp008_research/emp008/factors.py`
- Create: `emp008_research/emp008/preprocess.py`
- Create: `emp008_research/emp008/factor_pipeline.py`
- Create: `emp008_research/emp008/risk.py`
- Create: `emp008_research/emp008/optimize.py`
- Create: `emp008_research/emp008/factor_timing.py`
- Create: `emp008_research/emp008/strategy.py`
- Create: `emp008_research/emp008/run_weights.py`
- Create: `emp008_research/tests/unit/test_emp008_contract.py`

- [ ] Preserve the original factor formulas, factor-set order, warmup, benchmark completion, preprocessing, expected-alpha estimators, factor timing, risk models, and optimizer constraints.
- [ ] Replace `backtesting.*` imports with local `emp008_research.data` and `emp008` imports.
- [ ] Make `run_weights.py` accept `--data-dir`, `--start`, `--end`, `--factor-set`, `--risk-model`, `--tracking-error-annual`, and `--output-dir`.
- [ ] Write failing contract tests for default factor set, target-weight schema, and explicit data path, then implement and run them.
- [ ] Run focused unit tests and compare a bounded output against the original implementation.

### Task 4: Add independent target-weight backtest execution

**Files:**
- Create: `emp008_research/backtest/engine.py`
- Create: `emp008_research/backtest/spec.py`
- Create: `emp008_research/backtest/report.py`
- Create: `emp008_research/backtest/run_backtest.py`
- Create: `emp008_research/tests/unit/test_backtest_engine.py`

- [ ] Define an explicit `BacktestConfig` with `data_dir`, `weights_csv`, date range, capital, fill mode, fee, sell tax, slippage, and fractional-share policy.
- [ ] Load adjusted close and KOSPI200 membership from the supplied data directory, apply the target-weight schedule, calculate daily equity/returns/turnover/costs, and write JSON/CSV/Parquet summaries.
- [ ] Write a failing synthetic two-stock test covering target-weight rebalancing, close-fill return, turnover, and transaction costs.
- [ ] Implement the minimal engine and verify the synthetic test before connecting real EMP008 weights.
- [ ] Run the engine against the bounded real-data fixture and compare its gross/costed results with the original `BacktestRunner` output within documented tolerances.

### Task 5: Reorganize experiments and parity tests without changing originals

**Files:**
- Create: `emp008_research/experiments/README.md`
- Create: `emp008_research/experiments/factor_quantiles/README.md`
- Create: `emp008_research/experiments/weight_grid/README.md`
- Create: `emp008_research/experiments/size_value/README.md`
- Create: `emp008_research/tests/parity/test_weights_parity.py`
- Create: `emp008_research/tests/parity/test_backtest_parity.py`
- Create: `emp008_research/tests/fixtures/README.md`
- Create: `emp008_research/scripts/generate_weights.py`
- Create: `emp008_research/scripts/run_backtest.py`

- [ ] Copy experiment entrypoints and manifests into the new folders; keep generated outputs under `emp008_research/results/`.
- [ ] Add a bounded parity fixture configuration using the same start/end, factor set, risk model, TE, and cost assumptions as the original research run.
- [ ] Run the original and handoff weight generators from the test and compare axes, target weights, active weights, and diagnostics.
- [ ] Run the original and handoff backtests from the same target-weight CSV and compare equity, returns, turnover, cost, and summary metrics.
- [ ] Ensure parity tests fail clearly when data, configuration, or numerical tolerance changes.

### Task 6: Verify the complete handoff workflow and document evidence

**Files:**
- Modify: `emp008_research/README.md`
- Create: `emp008_research/results/.gitkeep`
- Create: `emp008_research/VERIFICATION.md`

- [ ] Run raw-to-parquet conversion into a temporary handoff data directory.
- [ ] Run weight generation from that parquet directory only.
- [ ] Run the backtest from the generated target weights and the same parquet directory only.
- [ ] Run `pytest emp008_research/tests -q` and the original focused EMP008 tests that do not mutate source artifacts.
- [ ] Record exact commands, dates, data manifest hashes, output paths, tolerances, and pass/fail counts in `VERIFICATION.md`.
- [ ] Confirm `git diff` contains no modifications to existing EMP008, backtesting, experiments, tests, raw, or parquet files.
