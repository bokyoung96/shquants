# EMP008 Research Handoff

This directory is an additive handoff package. The original `shquants` EMP008
implementation remains the reference implementation and is not modified by
this package.

The intended flow is:

```text
data/raw -> data/convert.py -> data/parquet
data/parquet -> scripts/generate_weights.py -> results/<run>/weights
target_weights.csv + data/parquet -> scripts/run_backtest.py -> results/<run>/backtest
```

## One-command run (recommended)

For handoff use, edit only the `SETTINGS` block in `run.py`:

```python
SETTINGS = RunSettings(
    start="2020-01-31",
    end="2024-12-31",
    factor_set="production_core",
    convert_raw_to_parquet=False,
    run_backtest=True,
    fee=0.0,
    sell_tax=0.0,
    slippage=0.0,
    run_name="my_run",
)
```

Then run:

```powershell
uv run python run.py
```

Set `convert_raw_to_parquet=True` when the raw files were updated. Set
`run_backtest=False` when only target weights are needed. The completed run is
written to `results/<run_name>/`, including `run_summary.json`.

### Factor-set naming

The operator-facing production set is `production_core`. Its factors are
price-to-252-day-high, earnings momentum, trailing dividend yield, retail flow,
FCF/TEV value, and log market cap (size). Other combinations are deliberately
prefixed so their role is clear:

| Prefix | Meaning |
| --- | --- |
| `production_` | The supported production methodology |
| `research_` | Variants used for factor research |
| `reference_` | Reproductions of the original/reference formulations |
| `diagnostic_` | Full-factor diagnostic runs, not production candidates |

The old short names such as `mfbt` remain accepted only as compatibility
aliases; new runs and saved settings should use the canonical names.
The full catalog is in [experiments/factor_sets.md](experiments/factor_sets.md).

Run commands from this directory (it is intentionally self-contained at the
Python import level):

```powershell
uv sync
uv run python scripts/convert_data.py --raw-dir data/raw --parquet-dir data/parquet
uv run python scripts/generate_weights.py --parquet-dir data/parquet --output-root results
uv run python scripts/run_backtest.py --data-dir data/parquet `
  --weights-csv results/production_core/weights/target_weights.csv `
  --output-dir results/production_core/backtest
uv run python -m pytest tests -q
```

`data/raw` is the portable source snapshot and `data/parquet` is the derived
cache. All commands accept explicit paths; no import from the parent research
repository is required. `experiments/` contains research notes and reproducible
experiment entry points, while `tests/` contains handoff regression tests.
See `VERIFICATION.md` for measured parity evidence.
