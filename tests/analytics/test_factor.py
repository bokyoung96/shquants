import pandas as pd
import pytest

from backtesting.analytics.factor import quantile_returns, rank_ic


def test_quantile_returns_uses_overlap_and_keeps_sparse_rows() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    signal = pd.DataFrame(
        {
            "A": [3.0, 1.0],
            "B": [2.0, 2.0],
            "C": [1.0, 3.0],
        },
        index=idx,
    )
    fwd = pd.DataFrame(
        {
            "A": [0.03, 0.01],
            "B": [0.02, 0.02],
            "C": [0.01, 0.03],
        },
        index=idx,
    )

    out = quantile_returns(signal, fwd, q=2)

    assert out.loc[idx[0], "q1"] == pytest.approx(0.015)
    assert out.loc[idx[0], "q2"] == pytest.approx(0.03)
    assert out.loc[idx[1], "q1"] == pytest.approx(0.015)
    assert out.loc[idx[1], "q2"] == pytest.approx(0.03)


def test_quantile_returns_drops_duplicate_signals_without_error() -> None:
    idx = pd.to_datetime(["2024-01-02"])
    signal = pd.DataFrame({"A": [1.0], "B": [1.0], "C": [1.0]}, index=idx)
    fwd = pd.DataFrame({"A": [0.01], "B": [0.02], "C": [0.03]}, index=idx)

    out = quantile_returns(signal, fwd, q=5)

    assert out.loc[idx[0], "q1"] == pytest.approx(0.02)
    assert out.loc[idx[0], "q2":].isna().all()


def test_quantile_returns_returns_empty_frame_without_overlap() -> None:
    signal = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-02"]))
    fwd = pd.DataFrame({"A": [0.01]}, index=pd.to_datetime(["2024-01-03"]))

    out = quantile_returns(signal, fwd, q=3)

    assert out.empty
    assert out.columns.tolist() == ["q1", "q2", "q3"]


def test_quantile_returns_returns_empty_frame_without_shared_columns() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    signal = pd.DataFrame({"A": [1.0, 2.0]}, index=idx)
    fwd = pd.DataFrame({"B": [0.01, 0.02]}, index=idx)

    out = quantile_returns(signal, fwd, q=3)

    assert out.empty
    assert out.columns.tolist() == ["q1", "q2", "q3"]


def test_rank_ic_uses_common_overlap_and_returns_nan_when_empty() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    signal = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 2.0]}, index=idx)
    fwd = pd.DataFrame({"A": [0.1, 0.3], "B": [0.2, 0.2]}, index=idx)

    ic = rank_ic(signal, fwd)

    assert ic.iloc[0] == pytest.approx(1.0)
    assert pd.isna(ic.iloc[1])


def test_rank_ic_returns_empty_series_without_overlap() -> None:
    signal = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-02"]))
    fwd = pd.DataFrame({"A": [0.01]}, index=pd.to_datetime(["2024-01-03"]))

    ic = rank_ic(signal, fwd)

    assert ic.empty


def test_rank_ic_returns_empty_series_without_shared_columns() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    signal = pd.DataFrame({"A": [1.0, 2.0]}, index=idx)
    fwd = pd.DataFrame({"B": [0.01, 0.02]}, index=idx)

    ic = rank_ic(signal, fwd)

    assert ic.empty
