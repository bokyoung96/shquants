# Remove Cross-Repo Test References Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `shquants` tests that depend on `1w1a`-owned modules so the repo test suite reflects only code that belongs to `shquants`.

**Architecture:** Keep the change minimal by deleting the crypto dashboard test file, trimming the smoke test so it only validates `backtesting` exports, and removing the KIS-only tests. Use the existing failing collection errors as the red phase, then rerun targeted and full pytest verification from the root uv environment.

**Tech Stack:** Python, pytest, uv

---

### Task 1: Remove `kis` from smoke coverage

**Files:**
- Modify: `tests/test_smoke.py`
- Test: `uv run python -m pytest -q tests/test_smoke.py`

- [ ] **Step 1: Use the existing failing test as the red phase**

Run: `uv run python -m pytest -q tests/test_smoke.py`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'kis'`.

- [ ] **Step 2: Apply the minimal implementation**

```python
import backtesting as bt


def test_public_package_exports_import_cleanly() -> None:
    export_names = bt.__all__

    assert isinstance(export_names, tuple)
    assert export_names == tuple(dict.fromkeys(export_names))
    assert {"DataCatalog", "BacktestEngine", "ValidationSession"}.issubset(export_names)

    namespace: dict[str, object] = {}
    exec("from backtesting import *", namespace)

    for name in export_names:
        assert name in namespace


def test_reporting_exports_import_cleanly() -> None:
    export_names = set(bt.__all__)

    assert "RunReader" in export_names
    assert "RunWriter" in export_names
    assert "ReportSpec" in export_names
    assert "ReportBundle" in export_names
    assert "ReportBuilder" in export_names
```

- [ ] **Step 3: Verify green**

Run: `uv run python -m pytest -q tests/test_smoke.py`

Expected: PASS.

### Task 2: Remove the crypto dashboard test

**Files:**
- Delete: `tests/crypto_dashboard/test_app.py`
- Test: `uv run python -m pytest -q`

- [ ] **Step 1: Use the existing failing test as the red phase**

Run: `uv run python -m pytest -q tests/crypto_dashboard/test_app.py`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'crypto'`.

- [ ] **Step 2: Apply the minimal implementation**

```bash
rm tests/crypto_dashboard/test_app.py
```

- [ ] **Step 3: Verify the repo suite no longer fails on cross-repo imports**

Run: `uv run python -m pytest -q`

Expected: no collection failure from `kis` or `crypto.dashboard`.

### Task 3: Remove the KIS-only tests

**Files:**
- Delete: `tests/kis/test_root.py`
- Delete: `tests/kis/test_tr_id_imports.py`
- Test: `uv run python -m pytest -q`

- [ ] **Step 1: Use the existing failing tests as the red phase**

Run: `uv run python -m pytest -q tests/kis`

Expected: FAIL because the repo no longer contains the `kis` package.

- [ ] **Step 2: Apply the minimal implementation**

```bash
rm tests/kis/test_root.py tests/kis/test_tr_id_imports.py
```

- [ ] **Step 3: Verify the repo suite no longer fails on KIS imports**

Run: `uv run python -m pytest -q`

Expected: no collection failure from `kis`.
### Task 4: Commit and push

**Files:**
- Create: `docs/superpowers/specs/2026-04-22-remove-cross-repo-tests-design.md`
- Create: `docs/superpowers/plans/2026-04-22-remove-cross-repo-tests.md`
- Modify: `tests/test_smoke.py`
- Delete: `tests/crypto_dashboard/test_app.py`
- Delete: `tests/kis/test_root.py`
- Delete: `tests/kis/test_tr_id_imports.py`

- [ ] **Step 1: Commit**

```bash
git add docs/superpowers/specs/2026-04-22-remove-cross-repo-tests-design.md docs/superpowers/plans/2026-04-22-remove-cross-repo-tests.md tests/test_smoke.py tests/crypto_dashboard/test_app.py tests/kis/test_root.py tests/kis/test_tr_id_imports.py
git commit -m "Remove cross-repo test references"
```

- [ ] **Step 2: Push**

Run: `git push origin main`

Expected: push succeeds to `origin/main`.
