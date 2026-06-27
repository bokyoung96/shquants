# KR_STOCK_5m

`parquet/KR_STOCK_5m` stores Korean individual-stock 5-minute OHLCV data as monthly wide parquet files:

```text
parquet/KR_STOCK_5m/
  year=2024/
    month=01/
      o.parquet
      h.parquet
      l.parquet
      c.parquet
      v.parquet
```

Each file uses `ts` as the index and ticker columns such as `A005930`. The readable adapter combines those split field files into:

```text
ts,ticker,open,high,low,close,volume
```

Quick checks:

```bash
uv run python scripts/inspect_kr_stock_5m.py --rows 10
uv run python scripts/inspect_kr_stock_5m.py --ticker 005930 --start "2024-01-03" --end "2024-01-04"
uv run python scripts/inspect_kr_stock_5m.py --ticker 005930 --start "2024-01-03" --end "2024-01-04" --output results/kr_stock_5m_005930_sample.csv
```

Python use:

```python
from pathlib import Path

from backtesting.data.kr_stock_5m import KrStock5mDataset, read_ticker_bars

dataset = KrStock5mDataset(Path("parquet/KR_STOCK_5m"))
bars = read_ticker_bars(dataset, "005930", start="2024-01-03", end="2024-01-04")
```
