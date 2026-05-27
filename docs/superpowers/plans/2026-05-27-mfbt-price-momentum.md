# MFBT Price Momentum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first `mfbt` multi-factor strategy with a `price_momentum` factor based on close divided by the 252-day rolling close high.

**Architecture:** `mfbt` is a new composable strategy module. The first version has one signal producer that emits binary `price_momentum` alpha and uses the existing `LongOnlyTopN` construction rule for equal-weight long-only selection.

**Tech Stack:** Python, pandas, pytest, existing `backtesting.strategies` registry and `ComposableStrategy`.

---

### Task 1: Add MFBT Tests

**Files:**
- Create: `tests/strategies/test_mfbt.py`
- Modify: `tests/strategies/test_registry.py`

- [ ] **Step 1: Write the failing tests**

Add tests that expect `build_strategy("mfbt")` to exist, `build_signal` to return `1.0` only when `close / close.rolling(252).max() > 0.8`, and `build_plan` to equal-weight selected names.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/test_mfbt.py tests/strategies/test_registry.py::test_registry_lists_default_strategies tests/strategies/test_registry.py::test_registry_lists_screened_strategy_names_only -q`

Expected: fail because `mfbt` is not registered yet.

### Task 2: Implement Strategy

**Files:**
- Create: `backtesting/strategies/mfbt.py`
- Modify: `backtesting/strategies/registry.py`
- Modify: `backtesting/strategies/__init__.py`

- [ ] **Step 1: Create `Mfbt` and `_MfbtSignal`**

Implement `price_momentum` as `close / close.rolling(252).max() > 0.8`, cast to `1.0 / 0.0`, expose `DatasetId.QW_ADJ_C`, and use `LongOnlyTopN`.

- [ ] **Step 2: Register `mfbt`**

Import `Mfbt` in the strategy registry and register it under the exact id `mfbt`.

- [ ] **Step 3: Run tests to verify green**

Run: `uv run pytest tests/strategies/test_mfbt.py tests/strategies/test_registry.py::test_registry_lists_default_strategies tests/strategies/test_registry.py::test_registry_lists_screened_strategy_names_only -q`

Expected: all selected tests pass.

### Task 3: Document Strategy

**Files:**
- Modify: `backtesting/strategies/README.md`

- [ ] **Step 1: Add README entry**

Document `mfbt`, its current `price_momentum` factor, `close` data dependency, and long-only top-N construction.

- [ ] **Step 2: Run targeted verification**

Run: `uv run pytest tests/strategies/test_mfbt.py tests/strategies/test_registry.py::test_registry_lists_default_strategies tests/strategies/test_registry.py::test_registry_lists_screened_strategy_names_only -q`

Expected: all selected tests pass.
