from __future__ import annotations

import pandas as pd

from scripts.run_tech_gamma_long_only import TechGammaConfig
from scripts.tech_gamma_universe import filter_kospi200_historical_members, kospi200_tickers


def test_kospi200_tickers_reads_latest_positive_members(tmp_path) -> None:
    membership = pd.DataFrame(
        {
            "A005930": [1, 1],
            "A000660": [0, 1],
            "A999999": [1, 0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    ).rename_axis("date")
    membership.to_parquet(tmp_path / "qw_k200_yn.parquet", engine="pyarrow")

    tickers = kospi200_tickers(tmp_path, TechGammaConfig(universe="kospi200_latest", start="2024-01-01", end="2024-01-31"))

    assert tickers == ("A000660", "A005930")


def test_filter_kospi200_historical_members_keeps_only_active_rows(tmp_path) -> None:
    membership = pd.DataFrame(
        {
            "A005930": [1, 1],
            "A000660": [0, 1],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    ).rename_axis("date")
    membership.to_parquet(tmp_path / "qw_k200_yn.parquet", engine="pyarrow")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "ticker": ["A005930", "A000660", "A000660"],
            "close": [100.0, 90.0, 91.0],
        }
    )

    filtered = filter_kospi200_historical_members(frame, tmp_path)

    assert filtered[["date", "ticker"]].to_dict("records") == [
        {"date": pd.Timestamp("2024-01-02"), "ticker": "A005930"},
        {"date": pd.Timestamp("2024-01-03"), "ticker": "A000660"},
    ]
