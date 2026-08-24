# EMP008 Research Handoff

This directory is an additive handoff package. The original `shquants` EMP008
implementation remains the reference implementation and is not modified by
this package.

The intended flow is:

```text
data/raw -> data/ingest/convert.py -> data/parquet
data/parquet -> scripts/generate_weights.py -> results/<run>/weights
target_weights.csv + data/parquet -> scripts/run_backtest.py -> results/<run>/backtest
```

Run commands from this directory (it is intentionally self-contained at the
Python import level):

```powershell
uv sync
uv run python scripts/convert_data.py --raw-dir data/raw --parquet-dir data/parquet
uv run python scripts/generate_weights.py --parquet-dir data/parquet --output-root results
uv run python scripts/run_backtest.py --data-dir data/parquet `
  --weights-csv results/mfbt_emp008/weights/target_weights.csv `
  --output-dir results/mfbt_emp008/backtest
uv run python -m pytest tests -q
```

`data/raw` is the portable source snapshot and `data/parquet` is the derived
cache. All commands accept explicit paths; no import from the parent research
repository is required. `experiments/` contains research notes and reproducible
experiment entry points, while `tests/` contains handoff regression tests.
See `VERIFICATION.md` for measured parity evidence.
