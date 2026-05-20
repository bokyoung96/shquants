# Advanced 3D RRG Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested `rrg/` package that loads KOSPI200 WICS sector data, computes multi-horizon RS/MOM/ACC metrics, and exports a Plotly 3D phase-space RRG.

**Architecture:** Keep the framework separate from backtesting strategies. `rrg.core` owns pure calculations, `rrg.data` adapts existing backtesting loaders into sector-level RRG inputs, and `rrg.plot`/`rrg.dashboard` own Plotly figure and HTML export.

**Tech Stack:** Python 3.11, pandas, numpy, plotly, pytest, existing `backtesting.data` and `backtesting.catalog` modules.

---

## File Structure

- Create `rrg/__init__.py`: public exports.
- Create `rrg/core.py`: dataclasses, multi-horizon formulas, classification, persistence.
- Create `rrg/data.py`: KOSPI200 WICS data adapter and sector return/index construction.
- Create `rrg/filters.py`: smoothing and rolling z-score helpers.
- Create `rrg/plot.py`: Plotly 3D RRG figure construction.
- Create `rrg/dashboard.py`: multi-horizon HTML export wrapper.
- Create `rrg/examples.py`: runnable KOSPI200 WICS example function and CLI entry point.
- Create `rrg/README.md`: formulas, usage, interpretation.
- Modify `pyproject.toml`: include `rrg*` in package discovery.
- Create `tests/rrg/test_core.py`: formula and label tests.
- Create `tests/rrg/test_data.py`: sector weighted return tests.
- Create `tests/rrg/test_plot.py`: Plotly/export tests.

## Task 1: Core Formula Tests And Implementation

**Files:**
- Create: `tests/rrg/test_core.py`
- Create: `rrg/core.py`
- Create: `rrg/filters.py`
- Create: `rrg/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing core tests**

Add tests that call `compute_horizon_rrg`, `compute_multi_horizon_rrg`, and `classify_rrg_state` before those modules exist. Tests must verify:

- log RS formula
- MOM formula
- ACC formula
- all four state labels
- turning labels for exhaustion and recovery
- persistence run counts
- tidy multi-horizon output

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/rrg/test_core.py -q`

Expected: FAIL because `rrg.core` does not exist.

- [ ] **Step 3: Implement minimal core**

Implement:

- `RrgConfig`
- `HorizonSpec`
- `compute_horizon_rrg`
- `compute_multi_horizon_rrg`
- `classify_rrg_state`
- `classify_turning_point`
- `add_state_persistence`
- `smooth_frame`
- `rolling_zscore`

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/rrg/test_core.py -q`

Expected: PASS.

## Task 2: KOSPI200 WICS Data Adapter

**Files:**
- Create: `tests/rrg/test_data.py`
- Create/modify: `rrg/data.py`

- [ ] **Step 1: Write failing data tests**

Tests must verify:

- `build_sector_return_index` applies market-cap weights inside each sector.
- zero market-cap weight basis falls back to equal weight.
- missing sector labels are excluded.
- `load_kospi200_wics_sector_rrg_input` declares and uses the expected dataset IDs through the existing loader shape.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/rrg/test_data.py -q`

Expected: FAIL because `rrg.data` does not exist.

- [ ] **Step 3: Implement minimal data adapter**

Implement:

- `RrgInputData`
- `required_kospi200_wics_datasets`
- `build_sector_return_index`
- `load_kospi200_wics_sector_rrg_input`

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/rrg/test_data.py -q`

Expected: PASS.

## Task 3: Plotly 3D Figure And HTML Export

**Files:**
- Create: `tests/rrg/test_plot.py`
- Create/modify: `rrg/plot.py`
- Create/modify: `rrg/dashboard.py`

- [ ] **Step 1: Write failing plot tests**

Tests must verify:

- `make_rrg_3d_figure` returns a Plotly figure with `scatter3d` traces.
- horizon dropdown includes configured horizons.
- `export_multi_horizon_rrg` writes an HTML file.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/rrg/test_plot.py -q`

Expected: FAIL because plotting modules do not exist.

- [ ] **Step 3: Implement minimal Plotly code**

Implement:

- latest-point markers per sector
- trail lines per sector
- horizon dropdown visibility controls
- HTML export

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/rrg/test_plot.py -q`

Expected: PASS.

## Task 4: Example And Documentation

**Files:**
- Create/modify: `rrg/examples.py`
- Create/modify: `rrg/README.md`

- [ ] **Step 1: Add example entry point**

Implement `run_kospi200_wics_example(start, end, output_path)` that loads data, computes metrics, and writes an HTML export.

- [ ] **Step 2: Add README**

Document formulas, axes, turning labels, example usage, and interpretation notes.

- [ ] **Step 3: Run import smoke test**

Run: `uv run python -c "from rrg.examples import run_kospi200_wics_example; print(run_kospi200_wics_example)"`

Expected: prints the function object without import errors.

## Task 5: Verification

**Files:**
- All files above.

- [ ] **Step 1: Run focused RRG tests**

Run: `uv run pytest tests/rrg -q`

Expected: PASS.

- [ ] **Step 2: Run adjacent existing tests**

Run: `uv run pytest tests/strategies/test_rrg_sector_rotation.py tests/data/test_loader.py -q`

Expected: PASS.

- [ ] **Step 3: Check package discovery**

Run: `uv run python -c "import rrg; print(rrg.__all__)"`

Expected: imports successfully and prints public exports.

## Self-Review

- The plan covers the approved spec: core formulas, data adapter, 3D Plotly export, example, and docs.
- ETF input support is deliberately excluded.
- Existing backtesting strategy code is not modified.
- Tests are required before production modules for each behavior group.
