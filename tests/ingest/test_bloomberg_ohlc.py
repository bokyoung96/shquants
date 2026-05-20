from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from assetallocation.data.loader import BloombergOHLCExcelConverter


def _write_bloomberg_workbook(path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(["Start Date", pd.Timestamp("2024-01-01").to_pydatetime(), None, None, None])
    sheet.append(["End Date", None, None, None, None])
    sheet.append([None, None, None, None, None])
    sheet.append([None, "AAA Index", None, None, None, "BBB Curncy", None, None, None])
    sheet.append(
        [
            None,
            "Open Price",
            "High Price",
            "Low Price",
            "Last Price",
            "Open Price",
            "High Price",
            "Low Price",
            "Last Price",
        ]
    )
    sheet.append(["Dates", "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST"])
    sheet.append([pd.Timestamp("2024-01-03").to_pydatetime(), 10, 12, 9, 11, 20, 23, 18, 22])
    sheet.append([pd.Timestamp("2024-01-02").to_pydatetime(), 8, 10, 7, 9, 19, 21, 17, 20])
    workbook.save(path)


def test_converter_splits_bloomberg_excel_into_ohlc_parquets(tmp_path: Path) -> None:
    source = tmp_path / "data_bb.xlsx"
    output_dir = tmp_path / "data"
    _write_bloomberg_workbook(source)

    result = BloombergOHLCExcelConverter(source, output_dir).convert()

    assert result.symbols == ["AAA Index", "BBB Curncy"]
    assert [path.name for path in result.parquet_paths] == [
        "open.parquet",
        "high.parquet",
        "low.parquet",
        "close.parquet",
    ]
    assert (output_dir / "data_bb.xlsx").exists()

    opened = pd.read_parquet(output_dir / "open.parquet", engine="pyarrow")
    assert opened.index.tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert opened.index.name == "date"
    assert opened.columns.tolist() == ["AAA Index", "BBB Curncy"]
    assert opened.loc[pd.Timestamp("2024-01-02"), "AAA Index"] == 8
    assert opened.loc[pd.Timestamp("2024-01-03"), "BBB Curncy"] == 20

    closed = pd.read_parquet(output_dir / "close.parquet", engine="pyarrow")
    assert closed.loc[pd.Timestamp("2024-01-02"), "AAA Index"] == 9
    assert closed.loc[pd.Timestamp("2024-01-03"), "BBB Curncy"] == 22


def test_converter_rejects_duplicate_dates(tmp_path: Path) -> None:
    source = tmp_path / "data_bb.xlsx"
    output_dir = tmp_path / "data"
    _write_bloomberg_workbook(source)

    frame = pd.read_excel(source, header=None)
    frame.iloc[7, 0] = frame.iloc[6, 0]
    frame.to_excel(source, index=False, header=False)

    with pytest.raises(ValueError, match="duplicate date"):
        BloombergOHLCExcelConverter(source, output_dir).convert()
