# EMP008 Report Package Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move EMP008-specific report generation and report execution code into a focused `reports` subpackage without changing behavior.

**Architecture:** Keep strategy construction and general pipeline runners at `backtesting/strategies/emp008/`. Place report-only comparison, attribution, and model-comparison modules under `backtesting/strategies/emp008/reports/`, then update all repository-local imports to the new canonical paths. Do not add compatibility wrappers because they would leave report code at the old boundary and duplicate the public surface.

**Tech Stack:** Python 3.12, pandas, matplotlib, pytest, Ruff, mypy

---

### Task 1: Lock the package boundary

**Files:**
- Modify: `tests/strategies/test_emp008_module_names.py`

- [ ] **Step 1: Add a failing structure test**

Add a test that imports `backtesting.strategies.emp008.reports` modules and asserts the old report module files are absent from the `emp008` root.

- [ ] **Step 2: Run the structure test and verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/strategies/test_emp008_module_names.py -q`

Expected: FAIL because `backtesting.strategies.emp008.reports` does not exist yet.

### Task 2: Create the report package and move report-only modules

**Files:**
- Create: `backtesting/strategies/emp008/reports/__init__.py`
- Move: `backtesting/strategies/emp008/adjust_comparison_report.py` to `backtesting/strategies/emp008/reports/adjust_comparison.py`
- Move: `backtesting/strategies/emp008/attribution.py` to `backtesting/strategies/emp008/reports/attribution.py`
- Move: `backtesting/strategies/emp008/comparison.py` to `backtesting/strategies/emp008/reports/comparison.py`
- Move: `backtesting/strategies/emp008/model_comparison_report.py` to `backtesting/strategies/emp008/reports/model_comparison.py`
- Move: `backtesting/strategies/emp008/run_model_comparison_report.py` to `backtesting/strategies/emp008/reports/run_model_comparison.py`

- [ ] **Step 1: Move the five report-only modules**

Preserve their contents, then change package-relative imports such as `from .comparison` and `from .strategy` to imports valid from the nested `reports` package.

- [ ] **Step 2: Export only the report package modules**

Create an empty `reports/__init__.py` so importing the package has no plotting or data-loading side effects.

- [ ] **Step 3: Run the structure test and verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/strategies/test_emp008_module_names.py -q`

Expected: PASS.

### Task 3: Update canonical imports

**Files:**
- Modify: `backtesting/strategies/emp008/run_full.py`
- Modify: `backtesting/strategies/emp008/experiments/value_factor_robustness.py`
- Modify: `tests/scripts/test_run_emp008_full.py`
- Modify: `tests/scripts/test_run_emp008_model_comparison_report.py`
- Modify: `tests/strategies/test_emp008_adjust_comparison_report.py`
- Modify: `tests/strategies/test_emp008_factor_pipeline.py`
- Modify: `tests/strategies/test_emp008_model_comparison_report.py`

- [ ] **Step 1: Replace old module paths with `backtesting.strategies.emp008.reports.*`**

Update direct imports and pytest monkeypatch target strings. Keep imported symbols and runtime behavior unchanged.

- [ ] **Step 2: Run focused report and runner tests**

Run: `.venv/Scripts/python.exe -m pytest tests/strategies/test_emp008_adjust_comparison_report.py tests/strategies/test_emp008_model_comparison_report.py tests/strategies/test_emp008_factor_pipeline.py tests/scripts/test_run_emp008_full.py tests/scripts/test_run_emp008_model_comparison_report.py -q`

Expected: PASS.

### Task 4: Verify the cleanup

**Files:**
- Modify only if verification exposes a scoped import or formatting defect.

- [ ] **Step 1: Confirm no old imports or root report modules remain**

Run: `rg -n "backtesting\.strategies\.emp008\.(adjust_comparison_report|attribution|comparison|model_comparison_report|run_model_comparison_report)" backtesting tests scripts`

Expected: no matches.

- [ ] **Step 2: Run lint and static analysis**

Run: `.venv/Scripts/python.exe -m ruff check backtesting/strategies/emp008 tests/strategies tests/scripts`

Run: `.venv/Scripts/python.exe -m mypy backtesting/strategies/emp008`

Expected: PASS, or document pre-existing repository-wide configuration gaps separately.

- [ ] **Step 3: Run the EMP008 test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/strategies/test_emp008_*.py tests/scripts/test_run_emp008_*.py -q`

Expected: PASS.
