from pathlib import Path

import pandas as pd

from backtesting.data.kr_stock_5m import (
    KrStock5mDataset,
    available_months,
    normalize_ticker,
    read_ticker_bars,
    read_tickers_bars,
)


def _write_field(root: Path, field: str, frame: pd.DataFrame) -> None:
    path = root / "year=2024" / "month=01" / f"{field}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow")


def test_read_ticker_bars_returns_tidy_ohlcv_when_field_files_are_split(
    tmp_path: Path,
) -> None:
    # Given: monthly OHLCV files split by field, matching parquet/KR_STOCK_5m.
    index = pd.to_datetime(["2024-01-02 09:00", "2024-01-02 09:05"])
    field_values = {
        "o": [100.0, 101.0],
        "h": [105.0, 106.0],
        "l": [99.0, 100.0],
        "c": [104.0, 102.0],
        "v": [1_000.0, 1_500.0],
    }
    for field, values in field_values.items():
        _write_field(
            tmp_path,
            field,
            pd.DataFrame({"A005930": values, "A000660": [1.0, 2.0]}, index=index).rename_axis("ts"),
        )

    # When: one ticker is read through the adapter.
    bars = read_ticker_bars(
        KrStock5mDataset(tmp_path),
        "005930",
        start="2024-01-02 09:00",
        end="2024-01-02 09:05",
    )

    # Then: the result is a readable long OHLCV table.
    assert list(bars.columns) == ["ts", "ticker", "open", "high", "low", "close", "volume"]
    assert bars["ticker"].tolist() == ["A005930", "A005930"]
    assert bars["close"].tolist() == [104.0, 102.0]
    assert bars["volume"].tolist() == [1_000.0, 1_500.0]


def test_read_tickers_bars_returns_tidy_ohlcv_for_multiple_tickers(tmp_path: Path) -> None:
    # Given: monthly OHLCV files with two requested tickers.
    index = pd.to_datetime(["2024-01-02 09:00", "2024-01-02 09:05"])
    field_values = {
        "o": ([100.0, 101.0], [200.0, 201.0]),
        "h": ([105.0, 106.0], [205.0, 206.0]),
        "l": ([99.0, 100.0], [199.0, 200.0]),
        "c": ([104.0, 102.0], [204.0, 202.0]),
        "v": ([1_000.0, 1_500.0], [2_000.0, 2_500.0]),
    }
    for field, (samsung, hynix) in field_values.items():
        _write_field(
            tmp_path,
            field,
            pd.DataFrame({"A005930": samsung, "A000660": hynix, "A999999": [1.0, 2.0]}, index=index).rename_axis("ts"),
        )

    # When: both tickers are read through the adapter.
    bars = read_tickers_bars(
        KrStock5mDataset(tmp_path),
        ("005930", "000660"),
        start="2024-01-02 09:00",
        end="2024-01-02 09:05",
    )

    # Then: the result is a readable long OHLCV table.
    assert bars.groupby("ticker")["close"].count().to_dict() == {"A000660": 2, "A005930": 2}
    assert bars.loc[bars["ticker"].eq("A000660"), "close"].tolist() == [204.0, 202.0]


def test_available_months_reads_partition_folders_when_data_exists(tmp_path: Path) -> None:
    # Given: one complete month and one metadata file.
    _write_field(
        tmp_path,
        "c",
        pd.DataFrame(
            {"A005930": [100.0]},
            index=pd.to_datetime(["2024-01-02 09:00"]),
        ).rename_axis("ts"),
    )
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    # When: months are discovered.
    months = available_months(KrStock5mDataset(tmp_path))

    # Then: only partition folders are reported.
    assert months == ["2024-01"]


def test_normalize_ticker_accepts_plain_six_digit_codes() -> None:
    # Given: a plain Korean stock code.
    raw = "005930"

    # When: the code is normalized.
    normalized = normalize_ticker(raw)

    # Then: it matches the parquet column convention.
    assert normalized == "A005930"
