# WRDS Folder Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `wrds/` into core, downloads, marketdata, universes, and derivatives packages while preserving current WRDS CLI behavior.

**Architecture:** Add focused packages and keep root-level compatibility shims for existing direct-script imports. Domain services produce DataFrames or named output bundles; shared download/output helpers handle CSV persistence.

**Tech Stack:** Python 3.11+, pandas, pytest, tqdm, existing `backtesting.data` source/download primitives.

---

### Task 1: Lock Current Behavior

**Files:**
- Test: `wrds/tests/test_wrds.py`

- [ ] **Step 1: Run existing WRDS tests**

Run: `uv run pytest wrds/tests/test_wrds.py -q`
Expected: existing tests pass before production refactor edits.

- [ ] **Step 2: Record failures before editing**

If failures appear, inspect whether they are environment-only or real regressions. Do not restructure code until the current baseline is understood.

### Task 2: Add Red Tests For The New Structure

**Files:**
- Modify: `wrds/tests/test_wrds.py`

- [ ] **Step 1: Add package-boundary tests**

Add tests that import:

```python
from core.registry import NamedRegistry
from downloads.batch import BatchCsvWriter, OutputFile
from marketdata.catalog import source_registry as marketdata_source_registry
from marketdata.consensus import sources as consensus_sources
from marketdata.fundamentals import sources as fundamental_sources
from marketdata.indexes import sources as index_sources
from marketdata.prices import sources as price_sources
from universes.us.service import US
from universes.factset.service import Universe
from derivatives.options.service import Options
```

Assert that marketdata categories include CRSP prices, IBES consensus, Compustat fundamentals, and CRSP indexes.

- [ ] **Step 2: Add shared batch writer test**

Add a test that writes two named DataFrames with `BatchCsvWriter`:

```python
writer = BatchCsvWriter()
results = writer.write(
    tmp_path,
    [
        OutputFile("names", "names.csv", pd.DataFrame({"permno": [1]})),
        OutputFile("universe", "universe.csv", pd.DataFrame({"permno": [1]})),
    ],
)
assert [result.name for result in results] == ["names", "universe"]
assert (tmp_path / "names.csv").exists()
assert (tmp_path / "universe.csv").exists()
```

- [ ] **Step 3: Verify RED**

Run: `uv run pytest wrds/tests/test_wrds.py::test_wrds_package_structure_exposes_data_domains wrds/tests/test_wrds.py::test_batch_writer_saves_named_dataframes -q`
Expected: FAIL because the new packages do not exist yet.

### Task 3: Add Core And Downloads Packages

**Files:**
- Create: `wrds/core/__init__.py`
- Create: `wrds/core/registry.py`
- Create: `wrds/core/io.py`
- Create: `wrds/core/sql.py`
- Create: `wrds/core/workflow.py`
- Create: `wrds/downloads/__init__.py`
- Create: `wrds/downloads/batch.py`
- Create: `wrds/downloads/service.py`
- Modify: `wrds/download.py`

- [ ] **Step 1: Implement shared registry**

Create `NamedRegistry` that accepts items with a `name` attribute and resolves them by name with the same error behavior as current registries.

- [ ] **Step 2: Implement batch CSV writer**

Create `OutputFile`, `SavedFile`, and `BatchCsvWriter.write(root, files)` so domain workflows can share CSV persistence.

- [ ] **Step 3: Move limit helper into core SQL**

Add `limit(sql, value)` and use it from root modules later.

- [ ] **Step 4: Keep existing downloader behavior**

Refactor `wrds/download.py` into a compatibility wrapper around `downloads.service.Downloader`.

- [ ] **Step 5: Verify GREEN**

Run: `uv run pytest wrds/tests/test_wrds.py::test_batch_writer_saves_named_dataframes -q`
Expected: PASS.

### Task 4: Extract Marketdata Catalog

**Files:**
- Create: `wrds/marketdata/__init__.py`
- Create: `wrds/marketdata/prices.py`
- Create: `wrds/marketdata/consensus.py`
- Create: `wrds/marketdata/fundamentals.py`
- Create: `wrds/marketdata/indexes.py`
- Create: `wrds/marketdata/catalog.py`
- Create: `wrds/marketdata/workflow.py`
- Modify: `wrds/provider.py`

- [ ] **Step 1: Move WRDS source definitions by domain**

Move CRSP tables into `prices.py`, IBES tables into `consensus.py`, Compustat tables into `fundamentals.py`, and CRSP index/S&P tables into `indexes.py`.

- [ ] **Step 2: Assemble source registry in `marketdata.catalog`**

Expose `source_registry()` with the same selections as current `provider.source_registry()`.

- [ ] **Step 3: Move `wrds data` workflow**

Create `marketdata.workflow.DataWorkflow` that uses the existing `backtesting.data.Pipeline`.

- [ ] **Step 4: Keep provider compatibility**

Make `provider.source_registry()` delegate to `marketdata.catalog.source_registry()`.

- [ ] **Step 5: Verify**

Run: `uv run pytest wrds/tests/test_wrds.py::test_data_catalog_resolves_rank_numbers_to_libraries wrds/tests/test_wrds.py::test_data_registry_defaults_to_2015_through_current_year wrds/tests/test_wrds.py::test_wrds_package_structure_exposes_data_domains -q`
Expected: PASS.

### Task 5: Extract Universe, Options, And US Domains

**Files:**
- Create: `wrds/universes/factset/*.py`
- Create: `wrds/universes/us/*.py`
- Create: `wrds/derivatives/options/*.py`
- Modify: `wrds/universe.py`
- Modify: `wrds/us.py`
- Modify: `wrds/options.py`
- Modify: `wrds/provider.py`

- [ ] **Step 1: Extract FactSet universe**

Move source, strategy, registry, service, and workflow responsibilities into `universes/factset/`. Keep `wrds/universe.py` as a compatibility shim.

- [ ] **Step 2: Extract Options**

Move OptionMetrics Protocols, sources, registry, and service into `derivatives/options/`. Replace direct save loops with shared `BatchCsvWriter`.

- [ ] **Step 3: Extract US universe**

Move stock/factset sources, builder strategy, registry, service, and workflow into `universes/us/`. Replace direct save loops with shared `BatchCsvWriter`.

- [ ] **Step 4: Verify domain compatibility**

Run: `uv run pytest wrds/tests/test_wrds.py::test_us_registry_composes_sources_and_builder wrds/tests/test_wrds.py::test_universe_uses_injected_source_and_strategy wrds/tests/test_wrds.py::test_options_raw_downloads_table_named_files -q`
Expected: PASS.

### Task 6: Thin Provider And Run

**Files:**
- Modify: `wrds/provider.py`
- Modify: `wrds/run.py`

- [ ] **Step 1: Make `provider.py` assembly-only**

`provider.flow_registry()` should compose workflow objects from `marketdata`, `universes`, and `derivatives`.

- [ ] **Step 2: Keep `run.py` dispatch-only**

Keep parser defaults and command names stable. Use provider registries for workflows and marketdata source selection.

- [ ] **Step 3: Verify CLI handler coverage**

Run: `uv run pytest wrds/tests/test_wrds.py::test_flow_registry_dispatches_us_and_universe_workflows wrds/tests/test_wrds.py::test_run_command_handlers_include_data_workflows -q`
Expected: PASS.

### Task 7: Full Verification

**Files:**
- All changed WRDS files

- [ ] **Step 1: Run WRDS unit tests**

Run: `uv run pytest wrds/tests/test_wrds.py -q`
Expected: PASS.

- [ ] **Step 2: Run relevant shared data tests**

Run: `uv run pytest tests/data tests/ingest -q`
Expected: PASS.

- [ ] **Step 3: Optional WRDS login smoke**

If final live WRDS validation is needed, write local ignored `wrds/config.json` and run `uv run --with wrds python wrds/run.py check`. This may still require interactive MFA and should not be treated as required for the offline refactor tests.

- [ ] **Step 4: Final static checks**

Run targeted import/search checks to ensure root-level ad hoc save loops were removed from `US`, `Options`, and universe workflows.
