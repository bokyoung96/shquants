import pandas as pd
import pytest

from backtesting.analytics.perf import summarize_perf


def test_summarize_perf_reports_core_metrics() -> None:
    returns = pd.Series(
        [0.0, -0.1, 0.1],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    perf = summarize_perf(returns)

    assert perf["cagr"] < 0.0
    assert perf["mdd"] == pytest.approx(-0.1)
    assert perf["sharpe"] == pytest.approx(0.0)


def test_summarize_perf_handles_constant_returns() -> None:
    returns = pd.Series([0.01] * 252, index=pd.RangeIndex(252))

    perf = summarize_perf(returns)

    assert perf["sharpe"] == 0.0
    assert perf["cagr"] == pytest.approx((1.01 ** 252) - 1.0)


def test_summarize_perf_marks_short_samples_undefined() -> None:
    returns = pd.Series([0.05], index=pd.RangeIndex(1))

    perf = summarize_perf(returns)

    assert pd.isna(perf["cagr"])
    assert perf["mdd"] == pytest.approx(0.0)
    assert pd.isna(perf["sharpe"])
