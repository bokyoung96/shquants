# RRG Fwd Flow1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `rrg-fwd-flow1` stock strategy and run a final backtest with performance output.

**Architecture:** Implement the strategy inside `backtesting/strategies/rrg_sector_rotation.py` to reuse existing RRG, forward revision, and flow helpers. Register the strategy as `rrg-fwd-flow1`, export the class, and add focused tests for registry exposure, max-10 concentration, consensus-first flow fallback, equal weighting, and cash when sparse.

**Tech Stack:** Python, pandas, pytest, existing shquants backtesting strategy registry and composable strategy APIs.

---

### Task 1: Lock Strategy Contract With Tests

**Files:**
- Modify: `tests/strategies/test_registry.py`

- [ ] **Step 1: Add failing registry and behavior tests**

Add tests that assert:

- `rrg-fwd-flow1` appears in `list_strategies()`.
- Building `rrg-fwd-flow1` returns a strategy whose weights never exceed 10 names.
- The strategy equal-weights selected names.
- A stock with missing consensus can enter only through positive flow fallback.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
uv run pytest tests/strategies/test_registry.py -q
```

Expected: failure because `rrg-fwd-flow1` is not registered yet.

### Task 2: Implement Signal-Confirmed Concentration

**Files:**
- Modify: `backtesting/strategies/rrg_sector_rotation.py`
- Modify: `backtesting/strategies/registry.py`
- Modify: `backtesting/strategies/__init__.py`
- Modify: `backtesting/strategies/README.md`

- [ ] **Step 1: Add `RrgFwdFlow1`**

Add a dataclass strategy that uses a dedicated signal builder and concentrated equal-weight construction.

- [ ] **Step 2: Add signal helpers**

Reuse existing helpers for:

- RRG state.
- stock consensus score.
- stock flow score.

Add sector-level confirmation by cap-weighting stock consensus/flow scores by sector.

- [ ] **Step 3: Add construction**

Select at most 10 positive candidates, equal-weight selected names, and leave cash when fewer than 10 names qualify.

- [ ] **Step 4: Register and document**

Register the strategy under `rrg-fwd-flow1`, export the class, and document the scheme.

### Task 3: Verify And Backtest

**Files:**
- No source edits expected.

- [ ] **Step 1: Run focused tests**

```bash
uv run pytest tests/strategies/test_registry.py tests/strategies/test_strategy_contracts.py -q
```

- [ ] **Step 2: Run final backtest**

```bash
uv run python -m backtesting.run --strategy rrg-fwd-flow1 --name rrg-fwd-flow1-final --start 2020-01-02 --end 2026-03-25 --top-n 10 --lookback 20 --schedule weekly --fill-mode close --out-root /tmp/shquants-rrg-fwd-flow1
```

- [ ] **Step 3: Inspect output**

Read `summary.json` and `positions/weights.parquet`. Confirm holdings count never exceeds 10.
