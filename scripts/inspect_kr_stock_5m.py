from __future__ import annotations

import argparse
from pathlib import Path

from root import ROOT
from backtesting.data.kr_stock_5m import KrStock5mDataset, read_ticker_bars, summarize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect KR_STOCK_5m split OHLCV parquet data.")
    parser.add_argument("--root", type=Path, default=ROOT.parquet_path / "KR_STOCK_5m")
    parser.add_argument("--ticker", help="Ticker such as 005930 or A005930.")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-12-31 23:59:59")
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = KrStock5mDataset(args.root)

    if args.ticker:
        frame = read_ticker_bars(dataset, args.ticker, start=args.start, end=args.end).head(args.rows)
    else:
        frame = summarize_dataset(dataset).tail(args.rows)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(args.output)
        return

    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
