# EMP008 Unified Run Entrypoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a beginner-friendly `run.py` that executes the complete EMP008 workflow from one editable settings block without requiring command-line argument knowledge.

**Architecture:** Keep the existing converter, EMP008 weight generator, and standalone backtest unchanged. Add a thin orchestrator that stores settings in a dataclass, optionally converts raw data, calls the existing modules, and writes one run directory with a JSON summary. Tests will validate settings translation and a dry-run path without running a long optimizer.

**Tech Stack:** Python 3.12, dataclasses, pathlib, pandas, existing EMP008 package modules, pytest.

---

### Task 1: Add the user-editable settings contract

**Files:**
- Create: `emp008_research/run.py`
- Test: `emp008_research/tests/unit/test_run_entrypoint.py`

- [ ] Write a failing test that imports `RunSettings`, checks beginner defaults, and verifies `run_settings()` rejects an end date before the start date.
- [ ] Implement `RunSettings` with explicit fields for dates, factor set, risk model, estimator, timing, conversion flag, backtest flag, costs, capital, and run name.
- [ ] Add `run_settings(settings, project_dir=None)` that resolves `data/raw`, `data/parquet`, and `results/<run_name>` relative to `run.py` and validates settings.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Wire the complete workflow without argparse

**Files:**
- Modify: `emp008_research/run.py`
- Test: `emp008_research/tests/unit/test_run_entrypoint.py`

- [ ] Add `run_settings()` stages: optional `convert_required`, `emp008.run_weights.run_emp008`, CSV export, and optional `backtest.run_backtest.run_backtest`.
- [ ] Use `BacktestSpec` to pass costs and execution settings directly; do not shell out or parse CLI arguments.
- [ ] Write `run_summary.json` containing settings, generated paths, and backtest summary.
- [ ] Add a `--dry-run`-free `main()` that simply calls `run_settings(RunSettings())`; the only normal edit surface remains the settings block.
- [ ] Run unit tests and a synthetic dry workflow.

### Task 3: Document operator workflow and verify real data

**Files:**
- Modify: `emp008_research/README.md`
- Modify: `emp008_research/VERIFICATION.md`

- [ ] Document the three settings that operators most commonly change: `START`, `END`, and `FACTOR_SET`, plus the optional conversion/backtest switches.
- [ ] Run `uv run python run.py` from `emp008_research` against the local parquet snapshot.
- [ ] Compare the resulting target-weight shape, row sums, and final equity with the already verified standalone commands.
- [ ] Run `uv run --project . pytest tests -q` and record the final result.

### Task 4: Commit the unified entrypoint

- [ ] Review the diff for accidental changes outside `emp008_research`.
- [ ] Commit with Lore trailers describing the single-file operator interface and verification evidence.
