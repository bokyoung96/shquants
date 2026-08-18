# KOSPI200 Double-Bottom Staged-Buy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible KOSPI200 daily double-bottom event study and non-overlapping staged-buy backtest with a Korean result report.

**Architecture:** Use one focused research module for daily aggregation, causal pattern detection, execution schedules, portfolio valuation, and metric calculation. A thin CLI script loads the local parquet source and writes CSV/Markdown/PNG artifacts. Tests exercise pure functions with synthetic daily bars before the source-file smoke run.

**Tech Stack:** Python 3.11+, pandas, numpy, matplotlib, pytest, uv.

---

### Task 1: Lock the public behavior with unit tests

**Files:**
- Create: `tests/test_kospi200_double_bottom.py`

- [ ] **Step 1: Write tests for daily aggregation and causal pivots**

Assert that UTC bars crossing Seoul midnight are assigned to the correct local date, OHLC uses first/max/min/last, a low pivot is confirmed only after two future sessions, and no signal is emitted before confirmation.

- [ ] **Step 2: Write tests for pattern filters and execution**

Use a synthetic series with lows at sessions 5 and 20 and a neckline between them. Assert the ±3% tolerance and 10% base rebound filter, next-session-open entry, and 50/25/25 fills at offsets 0/5/10.

- [ ] **Step 3: Write tests for missing fills, portfolio non-overlap, and metrics**

Assert that a fill beyond the data end is omitted rather than backfilled, overlapping signals select only the first eligible trade, and a known equity curve returns expected total return and drawdown.

- [ ] **Step 4: Run the new tests and confirm the expected RED failure**

Run: `uv run pytest tests/test_kospi200_double_bottom.py -q`

Expected: collection fails because `scripts.kospi200_double_bottom` does not yet exist.

### Task 2: Implement the research engine

**Files:**
- Create: `scripts/kospi200_double_bottom.py`
- Modify: `tests/test_kospi200_double_bottom.py` only as needed to correct test fixtures, never expected behavior

- [ ] **Step 1: Implement typed configuration and source aggregation**

Expose `DoubleBottomConfig`, `aggregate_daily_ohlc`, and a parquet loader. Keep all defaults from the design document in one configuration object.

- [ ] **Step 2: Implement confirmed pivots and pattern detection**

Implement `find_pivot_lows` and `detect_double_bottoms` using integer positional indices. Only use future bars for pivot confirmation; action dates must be strictly after the second pivot and the first close above the neckline.

- [ ] **Step 3: Implement fill schedules and event returns**

Implement `build_fills`, `run_event_study`, and explicit cost deductions. Each fill's notional contribution is based on target capital; staged weights must sum to the amount actually filled.

- [ ] **Step 4: Implement non-overlapping portfolio valuation and metrics**

Implement `select_non_overlapping_signals`, `run_portfolio`, and `summarize_equity`. Value open positions at close, subtract costs at fills, liquidate at the configured horizon, and include cash when later staged fills cannot occur.

- [ ] **Step 5: Run the unit tests and confirm GREEN**

Run: `uv run pytest tests/test_kospi200_double_bottom.py -q`

Expected: all new tests pass.

### Task 3: Add the reproducible CLI and report

**Files:**
- Create: `scripts/run_kospi200_double_bottom.py`
- Create: `tests/test_run_kospi200_double_bottom.py`

- [ ] **Step 1: Add CLI smoke test**

Run the CLI against a temporary synthetic parquet file and assert that it writes `signals.csv`, `event_summary.csv`, `portfolio_summary.csv`, `portfolio_equity.csv`, and `report.md`.

- [ ] **Step 2: Implement CLI output and Korean report**

Add `--input`, `--output-dir`, `--cost-bps`, and pattern/horizon options. Use a 10% base rebound threshold and report 15/20/25/30% sensitivity counts so the strict 30% zero-signal result remains visible. Report source period, assumptions, signal count, strategy comparison, limitations, and output paths.

- [ ] **Step 3: Add the equity/pattern chart**

Write a single PNG with KOSPI200 close plus selected entry markers and normalized portfolio equity. If no signals exist, still produce the summary/report and skip only the chart with a clear note.

- [ ] **Step 4: Run CLI smoke tests**

Run: `uv run pytest tests/test_run_kospi200_double_bottom.py -q`

Expected: all smoke tests pass.

### Task 4: Run the real KOSPI200 backtest and verify artifacts

**Files:**
- Create: `results/kospi200_double_bottom/` generated artifacts

- [ ] **Step 1: Execute the real data run**

Run: `uv run python scripts/run_kospi200_double_bottom.py --input parquet/KOSPI200_1m.parquet --output-dir results/kospi200_double_bottom`

- [ ] **Step 2: Inspect summary and ledger**

Confirm the source period is populated, the signal ledger has internally consistent dates, fills are causal, and both strategy rows exist even if the sample is sparse.

- [ ] **Step 3: Run focused and repository verification**

Run: `uv run pytest tests/test_kospi200_double_bottom.py tests/test_run_kospi200_double_bottom.py -q` and `uv run ruff check scripts/kospi200_double_bottom.py scripts/run_kospi200_double_bottom.py tests/test_kospi200_double_bottom.py tests/test_run_kospi200_double_bottom.py`.

- [ ] **Step 4: Review the report for evidence-vs-inference boundaries**

Ensure the report calls results historical, identifies the continuous-index limitation, and does not present the result as a guaranteed profitable trading strategy.
