# WRDS Historical Data Pipeline

This document summarizes the backtest data downloader in `wrds/data/`.

## Scope

The catalog maps the selected research-library ranks to WRDS schemas:

| Rank | Library | Output | Purpose |
| ---: | --- | --- | --- |
| 1 | `crsp` | `wrds/output/datas/crsp/` | Current CRSP CIZ equity returns, prices, delistings, and identifiers |
| 2 | `comp` | `wrds/output/datas/comp/` | Compustat annual/quarterly fundamentals and issuer metadata |
| 4 | `ibes` | `wrds/output/datas/ibes/` | Analyst estimates, actuals, summaries, and IBES identifiers |
| 12 | `crsp_a_indexes` | `wrds/output/datas/crsp_a_indexes/` | CRSP index and S&P 500 series |

## Usage

Sample/test download:

```bash
uv run --with wrds --with pandas --with tqdm python3 wrds/run.py data 1 2 4 12 \
  --limit 1 \
  --output wrds/output/datas/sample \
  --overwrite
```

Backtest download, 2015 through the latest available year:

```bash
uv run --with wrds --with pandas --with tqdm python3 wrds/run.py data 1 2 4 12 \
  --output wrds/output/datas \
  --chunksize 100000 \
  --retries 4
```

Download one table:

```bash
uv run --with wrds --with pandas --with tqdm python3 wrds/run.py data 1 \
  --tables stkdlysecuritydata \
  --output wrds/output/datas \
  --chunksize 100000 \
  --retries 4
```

## Output Layout

Small metadata tables are saved as one CSV:

```text
wrds/output/datas/crsp/stksecurityinfohist.csv
wrds/output/datas/comp/company.csv
wrds/output/datas/ibes/id.csv
```

Large dated tables are saved by year so interrupted downloads can resume:

```text
wrds/output/datas/crsp/stkdlysecuritydata/year=2015.csv
wrds/output/datas/crsp/stkdlysecuritydata/year=2016.csv
wrds/output/datas/comp/funda/year=2015.csv
wrds/output/datas/ibes/det_epsus/year=2015.csv
```

Existing files with data rows are skipped by default. Header-only or empty partitions are retried,
and empty latest-year partitions are removed instead of kept as misleading data.

Each run writes `manifest.csv` at the output root with rank, library, table, relative file path,
row count, and save status.
