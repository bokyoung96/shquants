from __future__ import annotations

import pandas as pd
import pytest

from rrg.data import build_sector_return_index, required_kospi200_wics_datasets


def test_build_sector_return_index_uses_market_cap_weights_by_sector() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    close = pd.DataFrame({"A": [100.0, 110.0], "B": [100.0, 120.0], "C": [100.0, 90.0]}, index=index)
    membership = pd.DataFrame(True, index=index, columns=close.columns)
    sector = pd.DataFrame({"A": ["Tech", "Tech"], "B": ["Tech", "Tech"], "C": ["Finance", "Finance"]}, index=index)
    market_cap = pd.DataFrame({"A": [75.0, 75.0], "B": [25.0, 25.0], "C": [100.0, 100.0]}, index=index)

    result = build_sector_return_index(
        close=close,
        membership=membership,
        sector=sector,
        market_cap=market_cap,
    )

    assert result.sector_returns.loc[index[1], "Tech"] == pytest.approx(0.125)
    assert result.sector_index.loc[index[1], "Tech"] == pytest.approx(1.125)
    assert result.sector_returns.loc[index[1], "Finance"] == pytest.approx(-0.10)


def test_build_sector_return_index_falls_back_to_equal_weight_and_excludes_missing_sector() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    close = pd.DataFrame({"A": [100.0, 110.0], "B": [100.0, 120.0], "C": [100.0, 130.0]}, index=index)
    membership = pd.DataFrame(True, index=index, columns=close.columns)
    sector = pd.DataFrame({"A": ["Tech", "Tech"], "B": ["Tech", "Tech"], "C": [None, None]}, index=index)
    market_cap = pd.DataFrame(0.0, index=index, columns=close.columns)

    result = build_sector_return_index(
        close=close,
        membership=membership,
        sector=sector,
        market_cap=market_cap,
    )

    assert result.sector_returns.loc[index[1], "Tech"] == pytest.approx(0.15)
    assert "C" not in result.sector_returns.columns
    assert list(required_kospi200_wics_datasets()) == [
        "qw_adj_c",
        "qw_BM",
        "qw_k200_yn",
        "qw_wics_sec_big",
        "qw_mktcap",
    ]
