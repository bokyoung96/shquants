# Backtesting and Reporting Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the local backtesting/reporting workflow around the provided raw data with minimal code churn.

**Architecture:** Keep the current backtesting/reporting structure, remove out-of-scope Graphify tests, add only missing dependencies, generate local parquet artifacts from `raw/`, and then fix only the remaining scoped failures revealed by the backtesting/reporting test subset.

**Tech Stack:** Python, uv, pytest, pandas, scipy, openpyxl, pyarrow

---

### Task 1: Remove repo-local Graphify scope from `shquants`

**Files:**
- Delete: `tests/test_graphify_generate.py`
- Test: `uv run python -m pytest -q tests/test_graphify_generate.py`

- [ ] **Step 1: Use the existing failing Graphify test as the red phase**

Run: `uv run python -m pytest -q tests/test_graphify_generate.py`

Expected: FAIL because `graphify-out/generate_graphify.py` is not part of the repo checkout.

- [ ] **Step 2: Apply the minimal implementation**

```bash
rm tests/test_graphify_generate.py
```

- [ ] **Step 3: Verify Graphify is no longer part of the scoped suite**

Run: `test ! -e tests/test_graphify_generate.py`

Expected: exit code `0`.

### Task 2: Add only the missing backtesting/reporting dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: targeted failing tests

- [ ] **Step 1: Use the existing failing tests as the red phase**

Run:

```bash
uv run python -m pytest -q \
  tests/analytics/test_factor.py::test_rank_ic_uses_common_overlap_and_returns_nan_when_empty \
  tests/ingest/test_pipeline.py::test_ingest_reads_xlsx_sources
```

Expected: FAIL with missing `scipy` and `openpyxl`.

- [ ] **Step 2: Add the minimal dependency changes**

Add these dependencies to `pyproject.toml`:

```toml
"openpyxl>=3.1,<4",
"scipy>=1.13,<2",
```

- [ ] **Step 3: Regenerate and sync the environment**

Run:

```bash
uv lock
uv sync
```

Expected: lock and sync succeed.

- [ ] **Step 4: Verify the targeted tests go green**

Run:

```bash
uv run python -m pytest -q \
  tests/analytics/test_factor.py::test_rank_ic_uses_common_overlap_and_returns_nan_when_empty \
  tests/ingest/test_pipeline.py::test_ingest_reads_xlsx_sources
```

Expected: PASS.

### Task 3: Generate local parquet artifacts from raw files

**Files:**
- Modify: `.gitignore`
- Test: local artifact generation command

- [ ] **Step 1: Ignore the local parquet artifact directory**

Add to `.gitignore`:

```gitignore
parquet/
results/
```

- [ ] **Step 2: Generate parquet artifacts using the existing ingest flow**

Run:

```bash
uv run python - <<'PY'
from backtesting.catalog import DataCatalog
from backtesting.ingest.pipeline import IngestJob
from backtesting.ingest.io import find_raw_path
from backtesting.reporting.benchmarks import _read_quantwise_benchmark_frame
from backtesting.data.store import ParquetStore
from root import ROOT

catalog = DataCatalog.default()
job = IngestJob(catalog=catalog, raw_dir=ROOT.raw_path, parquet_dir=ROOT.parquet_path)
store = ParquetStore(ROOT.parquet_path)

for dataset_id in catalog.ids():
    try:
        raw_path = find_raw_path(ROOT.raw_path, dataset_id.value)
    except FileNotFoundError:
        continue
    if dataset_id.value == "qw_BM" and raw_path.suffix == ".xlsx":
        frame = _read_quantwise_benchmark_frame(raw_path)
        store.write(dataset_id.value, frame)
        continue
    job.run(dataset_id)
PY
```

- [ ] **Step 3: Verify the critical parquet artifact exists**

Run:

```bash
uv run python - <<'PY'
from root import ROOT
print((ROOT.parquet_path / "qw_BM.parquet").exists())
PY
```

Expected: prints `True`.

### Task 4: Fix the remaining scoped reporting drift

**Files:**
- Modify: `tests/reporting/test_figures.py`
- Modify: `tests/reporting/test_tables.py`
- Test: targeted reporting tests

- [ ] **Step 1: Use the existing failing tests as the red phase**

Run:

```bash
uv run python -m pytest -q \
  tests/reporting/test_figures.py \
  tests/reporting/test_tables.py
```

Expected: FAIL because `RollingMetrics` now requires a `window` field.

- [ ] **Step 2: Apply the minimal implementation**

Update test fixtures to instantiate `RollingMetrics` with:

```python
RollingMetrics(
    window=252,
    series={"rolling_sharpe": rolling_sharpe, "rolling_beta": rolling_beta},
)
```

- [ ] **Step 3: Verify green**

Run:

```bash
uv run python -m pytest -q \
  tests/reporting/test_figures.py \
  tests/reporting/test_tables.py
```

Expected: PASS.

### Task 5: Verify the scoped backtesting/reporting suite

**Files:**
- Modify: tracked files from earlier tasks
- Test: scoped pytest command

- [ ] **Step 1: Run the scoped suite**

Run:

```bash
uv run python -m pytest -q \
  tests/analytics \
  tests/catalog \
  tests/construction \
  tests/data \
  tests/engine \
  tests/execution \
  tests/ingest \
  tests/policy \
  tests/reporting \
  tests/strategies \
  tests/strategy \
  tests/test_report_cli.py \
  tests/test_run.py \
  tests/test_smoke.py
```

Expected: PASS for the scoped backtesting/reporting suite.

### Task 6: Commit and push tracked changes

**Files:**
- Create: `docs/superpowers/specs/2026-04-22-backtesting-reporting-stabilization-design.md`
- Create: `docs/superpowers/plans/2026-04-22-backtesting-reporting-stabilization.md`
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `tests/reporting/test_figures.py`
- Modify: `tests/reporting/test_tables.py`
- Delete: `tests/test_graphify_generate.py`

- [ ] **Step 1: Commit**

```bash
git add .gitignore pyproject.toml uv.lock \
  docs/superpowers/specs/2026-04-22-backtesting-reporting-stabilization-design.md \
  docs/superpowers/plans/2026-04-22-backtesting-reporting-stabilization.md \
  tests/reporting/test_figures.py \
  tests/reporting/test_tables.py \
  tests/test_graphify_generate.py
git commit -m "Stabilize backtesting and reporting workflow"
```

- [ ] **Step 2: Push**

Run: `git push origin main`

Expected: push succeeds.
