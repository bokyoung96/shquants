# WRDS OptionMetrics Raw Data

This document summarizes the OptionMetrics raw-data downloader under `wrds/options.py`.

OptionMetrics uses `secid` as its main security key. The US stock universe uses CRSP `permno`, so the bridge is:

```text
permno -> wrdsapps.opcrsphist -> secid -> optionm raw tables
```

## Folder Layout

Raw downloads are organized by date:

```text
wrds/output/datas/options/raw/<date>/
```

Example:

```text
wrds/output/datas/options/raw/2025-08-29/
```

`wrds/output/` is ignored by git.

## Command

```bash
uv run --with wrds python3 wrds/run.py options raw 2025-08-29 \
  --output wrds/output/datas/options/raw/2025-08-29 \
  --limit 1000
```

`--limit` keeps large OptionMetrics tables bounded while developing. Removing or raising it should be done deliberately because option quote and volatility tables can be very large.

## Raw Files

| File | Source Table | Meaning |
| --- | --- | --- |
| `opcrsphist.csv` | `wrdsapps.opcrsphist` | Point-in-time CRSP `permno` to OptionMetrics `secid` link rows for the requested date. |
| `securd.csv` | `optionm.securd` | OptionMetrics security master. |
| `secnmd.csv` | `optionm.secnmd` | OptionMetrics security name history. |
| `secprdYYYY.csv` | `optionm.secprdYYYY` | Underlying/security prices for the requested date. |
| `opprcdYYYY.csv` | `optionm.opprcdYYYY` | Raw option quotes, IV, and greeks for the requested date. |
| `stdopdYYYY.csv` | `optionm.stdopdYYYY` | Standardized option metrics for the requested date. |

## Core Columns

| File | Key Columns |
| --- | --- |
| `opcrsphist.csv` | `permno`, `secid`, `sdate`, `edate`, `score` |
| `securd.csv` | `secid`, `ticker`, `cusip`, `sic`, `index_flag`, `exchange_d` |
| `secnmd.csv` | `secid`, `effect_date`, `ticker`, `cusip`, `issuer`, `issue` |
| `secprdYYYY.csv` | `secid`, `date`, `open`, `close`, `volume`, `return`, `shrout` |
| `opprcdYYYY.csv` | `secid`, `date`, `optionid`, `symbol`, `exdate`, `cp_flag`, `strike_price`, `best_bid`, `best_offer`, `impl_volatility`, `delta`, `gamma`, `vega`, `theta` |
| `stdopdYYYY.csv` | `secid`, `date`, `days`, `strike_price`, `premium`, `impl_volatility`, `delta`, `gamma`, `theta`, `vega`, `cp_flag` |

## Coverage

The checked annual OptionMetrics table families exist from `1996` through `2025`:

| Prefix | Meaning |
| --- | --- |
| `opprcdYYYY` | Raw option quotes |
| `secprdYYYY` | Underlying/security prices |
| `stdopdYYYY` | Standardized options |
| `vsurfdYYYY` | Volatility surface |
| `hvoldYYYY` | Historical volatility |
| `fwdprdYYYY` | Forward prices |
| `borrateYYYY` | Borrow rates |

The latest checked date for 2025 price-style tables is `2025-08-29`.
