from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from backtesting.catalog import DatasetId
from backtesting.data import DataLoader, LoadRequest, ParquetStore
from backtesting.run import BacktestRunner
from backtesting.strategies import build_strategy


OUTPUT = Path("reports") / "mfbt_signal_matrices.xlsx"
START = "2000-01-01"
END = "2026-05-27"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    factors = build_mfbt_factor_signals(start=START, end=END)

    write_signal_matrix_workbook(OUTPUT, factors)
    print(OUTPUT.resolve())
    for name, frame in factors.items():
        output = signal_rows_only(frame)
        print(f"{name}: signal_rows={output.shape[0]} tickers={output.shape[1]}")


def build_mfbt_factor_signals(*, start: str, end: str) -> dict[str, pd.DataFrame]:
    strategy = build_strategy("mfbt")
    datasets = list(strategy.datasets)
    if DatasetId.QW_K200_YN not in datasets:
        datasets.append(DatasetId.QW_K200_YN)

    loader = DataLoader(BacktestRunner().catalog, ParquetStore(Path("parquet")))
    market = loader.load(LoadRequest(datasets=datasets, start=start, end=end))
    market.universe = market.frames["k200_yn"].fillna(0).astype(bool)
    return strategy.signal_producer.build(market).meta


def signal_rows_only(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(how="all")


def write_signal_matrix_workbook(path: Path, factors: dict[str, pd.DataFrame]) -> None:
    wb = Workbook(write_only=True)
    for name, frame in factors.items():
        output = signal_rows_only(frame)
        ws = wb.create_sheet(name)
        ws.append(["date", *[str(column) for column in output.columns]])
        values = output.to_numpy(dtype=object, copy=True)
        values[pd.isna(values)] = None
        for idx, row_values in zip(output.index, values):
            ws.append([idx.to_pydatetime(), *row_values.tolist()])
    wb.save(path)


if __name__ == "__main__":
    main()
