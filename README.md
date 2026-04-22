# shquants

`shquants` is a unified research workspace for quantitative backtesting, reporting, dashboard experiments, and analyst-oriented document ingestion.

The repository is organized around three main areas:

- **Backtesting** — factor, strategy, portfolio construction, execution, and reporting workflows
- **Dashboard** — backend/frontend components for presenting research outputs
- **Analysts** — local ingestion and summarization tools for research documents from Telegram and Gmail

This repo is designed to be used with a single root-level **`uv`** environment.

---

## What’s inside

### Backtesting
The `backtesting/` package contains the core research and simulation workflow:

- raw-data ingestion
- catalog and data access layers
- strategy and construction logic
- execution and policy modeling
- analytics and reporting output generation

The current setup supports using local CSV/XLSX source data and producing parquet artifacts locally when helpful for faster iteration.

### Dashboard
The `dashboard/` package contains the application code for surfacing outputs and research views.

It is intended as the presentation layer for results generated elsewhere in the workspace rather than a separate standalone environment.

### Analysts
The `analysts/` area contains tools for local research-document collection and summarization:

- **Telegram ingestion** via Telethon
- **Gmail ingestion** via Gmail API + local OAuth flow
- PDF extraction and summary generation
- local artifact persistence under `analysts/data/`

This area is intended for local operator use. Secrets, sessions, tokens, crawled files, and processed outputs are kept out of git.

---

## Repository layout

```text
shquants/
├── analysts/        # Telegram/Gmail ingestion and local summary tooling
├── backtesting/     # Core quant research and reporting packages
├── dashboard/       # UI/backend app code
├── docs/            # Specs, plans, and project notes
├── tests/           # Root test suite
├── raw/             # Local raw research data (untracked/local artifact)
├── parquet/         # Local parquet artifacts (ignored/local artifact)
├── pyproject.toml   # Root project definition for uv
└── uv.lock          # Locked environment
```

---

## Quick start

### 1) Sync the root environment

```bash
uv sync
```

### 2) Run the test suite

```bash
uv run python -m pytest
```

### 3) Run focused backtesting/reporting validation

```bash
uv run python -m pytest \
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

---

## Analysts workflow notes

The analysts tooling is available from the same root `uv` environment, but it depends on **local configuration and secrets**.

Examples of local-only runtime artifacts:

- `analysts/config.local.json`
- `analysts/data/state/*.session`
- `analysts/data/state/*token*.json`
- `analysts/data/raw/`
- `analysts/data/processed/`

These are intentionally gitignored.

### Example commands

Show analysts config:

```bash
PYTHONPATH=analysts/src uv run python -m analysts.cli show-config --base-dir analysts
```

Telegram auth:

```bash
PYTHONPATH=analysts/src uv run python -m analysts.cli auth-login --base-dir analysts
```

Telegram one-shot run:

```bash
PYTHONPATH=analysts/src uv run python -m analysts.cli run-once --channel DOC_POOL --base-dir analysts
```

Gmail auth:

```bash
PYTHONPATH=analysts/src uv run python -m analysts.cli gmail-auth-login --base-dir analysts
```

If you use the Gmail web-capture lane, you may also need:

```bash
uv run python -m playwright install chromium
```

---

## Data and artifact policy

This repository distinguishes between **code/configuration** and **local data artifacts**.

### Tracked

- source code
- tests
- templates
- docs/specs/plans
- environment definition (`pyproject.toml`, `uv.lock`)

### Local only

- raw source datasets
- generated parquet outputs
- Telegram/Gmail crawled files
- local session/token files
- SQLite state databases
- processed summary outputs

This keeps the public repository clean while preserving local operator workflows.

---

## Current status

The workspace currently includes:

- a repo-wide root `uv` environment
- stabilized backtesting/reporting test coverage
- local parquet generation support from raw CSV/XLSX inputs
- local Telegram analyst ingestion verified through Telethon
- Gmail ingestion support with local OAuth credentials

The repository is still best understood as an actively used research workspace rather than a polished end-user product.

---

## Notes

- Python version: **3.11+**
- Primary package manager / runner: **`uv`**
- Some workflows rely on local tools such as `codex` for summary generation

If you are exploring the codebase for the first time, start with:

- `backtesting/` for research and reporting logic
- `analysts/src/analysts/cli.py` for ingestion entrypoints
- `pyproject.toml` for environment and package structure
