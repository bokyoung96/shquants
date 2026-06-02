# WRDS Column Dictionary

This note documents the columns used by the latest WRDS US universe outputs.

For file-level interpretation, row counts, coverage dates, and universe rules, see `wrds/datas/universe.md`.

## Core Identifiers

| Column | Meaning |
| --- | --- |
| `permno` | CRSP permanent security identifier. Main security-level key. |
| `permco` | CRSP permanent company identifier. Multiple securities can map to one company. |
| `hdrcusip` | CRSP header CUSIP. Useful for broad identifier lookup. |
| `hdrcusip9` | 9-character header CUSIP. |
| `cusip` | Row-level CRSP CUSIP for the name interval. |
| `cusip9` | 9-character row-level CUSIP. |
| `fsym_regional_id` | FactSet regional/listing-level identifier. |
| `fsym_security_id` | FactSet security-level identifier. |
| `factset_entity_id` | FactSet entity/company identifier. |

## Tickers And Names

| Column | Meaning |
| --- | --- |
| `crsp_ticker` | CRSP ticker for the name interval. |
| `trade_ticker` | Trading ticker. In the latest source this follows the CRSP ticker field. |
| `factset_ticker` | FactSet ticker from the CRSP-FactSet link table. |
| `ticker_exchange` | FactSet ticker with exchange-style suffix, when available. |
| `company` | CRSP company/security name for the interval. |

## Market, Type, And Status

| Column | Meaning |
| --- | --- |
| `market` | Generated exchange label: `NYSE`, `AMEX`, or `NASDAQ`. |
| `exchange` | Exchange label derived from `primaryexch`. |
| `primaryexch` | CRSP primary exchange indicator: `N`, `A`, or `Q` for the current US universe. |
| `shareclass` | CRSP share class, when populated. |
| `sharetype` | CRSP share type. Current universe keeps `NS`. |
| `securitytype` | CRSP security type. Current universe keeps `EQTY`. |
| `securitysubtype` | CRSP security subtype. Current universe keeps `COM`. |
| `usincflg` | US incorporation flag. Current universe keeps `Y`. |
| `issuertype` | CRSP issuer type, such as `CORP` or `REIT`. |
| `siccd` | CRSP SIC industry code. |
| `tradingstatusflg` | CRSP trading status flag. Tradable membership keeps `A`. |
| `conditionaltype` | CRSP trading condition/type flag. Tradable membership keeps `RW`. |

## Dates

| Column | Meaning |
| --- | --- |
| `namedt` | CRSP name/listing row start date. |
| `nameendt` | CRSP name/listing row end date. |
| `securitybegdt` | CRSP security-level start date. |
| `securityenddt` | CRSP security-level end date. |
| `link_bdate` | FactSet link start date. |
| `link_edate` | FactSet link end date. |
| `start_date` | Effective joined mapping start date. |
| `end_date` | Effective joined mapping end date. |

## File-Specific Columns

| File | Columns |
| --- | --- |
| `names.csv` | CRSP identifiers, ticker/name fields, market/type/status fields, and CRSP date fields. |
| `factset_links.csv` | CRSP identifiers, FactSet identifiers, FactSet ticker fields, and link dates. |
| `history.csv` | `names.csv` joined with `factset_links.csv`, plus `start_date` and `end_date`. |
| `latest.csv` | One latest representative row per `permno` from `history.csv`. |
| `current/universe.csv` | Latest active tradable universe at the common latest date. |
| `at_YYYYMMDD.csv` | Point-in-time tradable universe for one requested date. |
