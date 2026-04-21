# shquants Root UV Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a single repo-root uv environment for all checked-in Python code in `shquants`.

**Architecture:** Add one root `pyproject.toml` that declares the shared dependencies and package discovery rules for `backtesting`, `dashboard`, and `analysts`. Then generate a root lockfile and verify the environment from the repo root with import checks and a full pytest run to surface any remaining non-environment blockers.

**Tech Stack:** uv, setuptools, pytest, pandas, numpy, fastapi, plotly, telethon, pymupdf, playwright

---

### Task 1: Add repo-root uv project metadata

**Files:**
- Create: `pyproject.toml`
- Test: `uv lock`

- [ ] **Step 1: Write the configuration file**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "shquants"
version = "0.1.0"
description = "Unified shquants workspace for backtesting, dashboard, and analysts tooling."
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "fastapi==0.115.12",
  "httpx>=0.28,<1",
  "jinja2>=3.1,<4",
  "matplotlib>=3.8,<4",
  "numpy>=1.26,<3",
  "pandas>=2.2,<3",
  "plotly>=6,<7",
  "playwright>=1.54,<2",
  "pyarrow>=15,<21",
  "pydantic==2.11.3",
  "pymupdf>=1.24,<2",
  "pytest>=8.3,<9",
  "telethon>=1.36,<2",
  "tqdm>=4.66,<5",
  "uvicorn==0.34.0",
]

[tool.setuptools]
include-package-data = true
py-modules = ["report", "root", "run"]

[tool.setuptools.packages.find]
where = [".", "analysts/src"]
include = ["analysts*", "backtesting*", "dashboard*"]
exclude = ["analysts.tests*", "tests*"]

[tool.setuptools.package-data]
backtesting = [
  "reporting/styles.css",
  "reporting/templates/*.j2",
  "reporting/templates/partials/*.j2",
]

[tool.pytest.ini_options]
testpaths = ["tests", "analysts/tests"]
addopts = "-q"
```

- [ ] **Step 2: Run lock generation**

Run: `uv lock`

Expected: a new repo-root `uv.lock` is written without configuration errors.

### Task 2: Verify the unified environment

**Files:**
- Modify: `uv.lock`
- Test: repo-root uv commands

- [ ] **Step 1: Sync the environment**

Run: `uv sync`

Expected: the root environment installs the project and its dependencies successfully.

- [ ] **Step 2: Verify key imports**

Run:

```bash
uv run python - <<'PY'
import analysts
import backtesting
import dashboard
import fastapi
import httpx
import jinja2
import matplotlib
import numpy
import pandas
import plotly
import pyarrow
import pydantic
import pytest
import telethon
import tqdm
print("imports-ok")
PY
```

Expected: prints `imports-ok`.

- [ ] **Step 3: Run the full test suite to capture remaining state**

Run: `uv run python -m pytest`

Expected: pytest starts with the root uv environment; any remaining failures should now reflect missing local modules/assets or true project issues rather than missing third-party libraries.

### Task 3: Save documentation and version control state

**Files:**
- Create: `docs/superpowers/specs/2026-04-22-shquants-root-uv-design.md`
- Create: `docs/superpowers/plans/2026-04-22-shquants-root-uv.md`

- [ ] **Step 1: Commit the uv environment setup**

```bash
git add pyproject.toml uv.lock docs/superpowers/specs/2026-04-22-shquants-root-uv-design.md docs/superpowers/plans/2026-04-22-shquants-root-uv.md
git commit -m "Add repo-wide uv environment for shquants"
```

- [ ] **Step 2: Push to main**

Run: `git push origin main`

Expected: the root uv configuration is available on `origin/main`.
