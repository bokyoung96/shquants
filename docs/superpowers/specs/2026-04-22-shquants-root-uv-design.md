# shquants Root UV Environment Design

## Goal

Create a single repo-root `uv` project for all of `shquants` so one environment can install and run the shared Python code in:

- `backtesting/`
- `dashboard/`
- `analysts/src/analysts/`
- root helper modules such as `run.py`, `report.py`, and `root.py`

## Context

The repository currently has only `analysts/pyproject.toml`, which means:

- `uv` is not configured from the repo root
- the main `tests/` suite cannot be run from a single repo-level environment
- `analysts` uses a separate dependency declaration from the rest of the repo

The user wants one `shquants` uv environment for the entire repository.

## Chosen Approach

Create one root `pyproject.toml` that:

1. defines a single `shquants` project
2. includes external dependencies needed across the repo
3. configures setuptools to discover:
   - root packages in `.`
   - the `analysts` package from `analysts/src`
4. allows `uv lock` / `uv sync` / `uv run` from the repo root

`analysts/pyproject.toml` remains in place for now to avoid unrelated churn, but the root project becomes the primary environment the user runs.

## Dependency Strategy

Put the shared runtime and test libraries directly in the root project dependencies so the single uv environment works without requiring extra group flags. This includes:

- data/runtime: `pandas`, `numpy`, `pyarrow`, `jinja2`, `matplotlib`, `plotly`
- API/dashboard: `fastapi`, `uvicorn`, `pydantic`, `httpx`
- analysts tooling: `telethon`, `pymupdf`, `playwright`, `tqdm`
- test runner: `pytest`

## Packaging Strategy

Use setuptools package discovery with multiple search roots:

- `.` for `backtesting` and `dashboard`
- `analysts/src` for `analysts`

Also expose repo-root modules:

- `run`
- `report`
- `root`

## Verification Plan

Verify the configuration by:

1. generating a repo-root `uv.lock`
2. syncing the environment from the repo root
3. importing the main third-party libraries and repo packages with `uv run python -c ...`
4. running the repo test suite to see the remaining state explicitly

## Known Risk

The existing repository appears to reference local modules and assets that are not currently present in the checkout, including:

- `kis`
- `crypto.dashboard`
- `graphify-out/generate_graphify.py`

This design does not invent or replace those missing project components. It only creates the single repo-wide uv environment and installs the external libraries needed by the checked-in Python code.
