from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from backtesting.reporting.analytics import annualized_sharpe
from backtesting.reporting.benchmarks import BenchmarkRepository, SectorRepository
from backtesting.reporting.models import BenchmarkConfig, SavedRun
from backtesting.reporting.snapshots import PerformanceSnapshotFactory


_DEFAULT = object()


def test_performance_snapshot_factory_builds_analytics_snapshot() -> None:
    run = _toy_run()
    factory = PerformanceSnapshotFactory(
        benchmark_repo=BenchmarkRepository.from_frame(_benchmark_prices()),
        sector_repo=SectorRepository.from_frame(_sector_map(), prices=_asset_prices()),
    )

    snapshot = factory.build(run, BenchmarkConfig.default_kospi200())

    assert snapshot.run_id == "toy-run"
    assert snapshot.display_name == "Toy Strategy"
    assert snapshot.metrics.cagr > 0.0
    assert snapshot.metrics.beta is not None
    assert "rolling_sharpe" in snapshot.rolling.series
    assert "rolling_beta" in snapshot.rolling.series
    assert_series_equal(snapshot.rolling.series["rolling_sharpe"].index.to_series(), run.returns.index.to_series())
    assert_series_equal(snapshot.rolling.series["rolling_beta"].index.to_series(), run.returns.index.to_series())
    assert snapshot.strategy_name == "trend_rank"
    assert snapshot.benchmark == BenchmarkConfig.default_kospi200()
    assert snapshot.exposure.holdings_count.iloc[-1] == 2
    assert len(snapshot.exposure.latest_holdings) == 2
    assert snapshot.sectors.latest_weighted.to_dict() == {"Tech": 0.6, "Utilities": 0.4}
    assert snapshot.sectors.latest_count.to_dict() == {"Tech": 1.0, "Utilities": 1.0}
    assert snapshot.drawdowns.episodes["drawdown"].lt(0.0).any()
    assert {"peak", "duration_days", "recovery_days", "recovered"} <= set(snapshot.drawdowns.episodes.columns)
    assert snapshot.drawdowns.episodes["recovered"].all()
    assert not snapshot.research.monthly_heatmap.empty
    assert snapshot.research.monthly_heatmap.loc[2024, 1] == pytest.approx(float((1.0 + run.returns).prod() - 1.0))
    assert not snapshot.research.return_distribution.empty
    assert int(snapshot.research.return_distribution["count"].sum()) == len(run.returns)
    assert list(snapshot.research.yearly_excess_returns.index.year) == [2024]
    assert snapshot.research.sector_contribution_method == "weighted-asset-return-attribution"
    assert snapshot.research.sector_weights.loc[run.returns.index[-1], ["Tech", "Utilities"]].to_dict() == {
        "Tech": 0.6,
        "Utilities": 0.4,
    }
    assert not snapshot.research.sector_contribution.empty
    assert set(snapshot.research.sector_contribution.columns) >= {"Tech", "Utilities"}
    expected_sortino = _expected_sortino(run.returns)
    assert snapshot.metrics.sortino >= 0.0
    assert snapshot.metrics.sortino == expected_sortino


def test_performance_snapshot_factory_exposes_benchmark_ohlc_for_research() -> None:
    run = _toy_run()
    factory = PerformanceSnapshotFactory(
        benchmark_repo=BenchmarkRepository.from_frame(_benchmark_ohlc_prices()),
        sector_repo=SectorRepository.from_frame(_sector_map(), prices=_asset_prices()),
    )

    snapshot = factory.build(run, BenchmarkConfig.default_kospi200())

    expected = pd.DataFrame(
        {
            "open": [199.0, 200.0, 200.0, 201.0, 200.5, 201.0, 202.0, 203.0],
            "high": [201.0, 202.0, 201.0, 203.0, 202.0, 203.0, 204.0, 205.0],
            "low": [198.0, 199.5, 199.0, 200.0, 200.0, 200.5, 201.5, 202.0],
            "close": [200.0, 201.0, 200.5, 202.0, 201.0, 202.5, 203.0, 204.0],
        },
        index=run.returns.index.copy(),
    )
    expected.index.name = "date"

    assert_frame_equal(snapshot.research.benchmark_ohlc, expected)
    expected_returns = expected["close"].pct_change().fillna(0.0).rename("benchmark_returns")
    expected_returns.index.name = run.returns.index.name
    assert_series_equal(snapshot.benchmark_returns, expected_returns)


def test_performance_snapshot_factory_derives_latest_holdings_when_optional_table_missing() -> None:
    run = _toy_run(latest_weights=None)
    factory = PerformanceSnapshotFactory(
        benchmark_repo=BenchmarkRepository.from_frame(_benchmark_prices()),
        sector_repo=SectorRepository.from_frame(_sector_map(), prices=_asset_prices()),
    )

    snapshot = factory.build(run, BenchmarkConfig.default_kospi200())

    expected = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "target_weight": [0.6, 0.4],
            "abs_weight": [0.6, 0.4],
        }
    )
    assert_frame_equal(snapshot.exposure.latest_holdings.reset_index(drop=True), expected)


def test_performance_snapshot_factory_applies_korean_sector_and_stock_display_names() -> None:
    run = _toy_run()
    factory = PerformanceSnapshotFactory(
        benchmark_repo=BenchmarkRepository.from_frame(_benchmark_prices()),
        sector_repo=SectorRepository.from_frame(
            _sector_map().rename(columns={"A": "A000001", "B": "A000002", "C": "A000003"}),
            prices=_asset_prices(),
            sector_name_map={"Tech": "Technology", "Utilities": "Utilities KR", "Health Care": "Health Care KR"},
            stock_name_map={"A000001": "Alpha", "A000002": "Beta", "A000003": "Charlie"},
        ),
    )
    mapped_run = SavedRun(
        run_id=run.run_id,
        path=run.path,
        config=run.config,
        summary=run.summary,
        equity=run.equity,
        returns=run.returns,
        turnover=run.turnover,
        weights=run.weights.rename(columns={"A": "A000001", "B": "A000002", "C": "A000003"}),
        qty=run.qty.rename(columns={"A": "A000001", "B": "A000002", "C": "A000003"}),
        latest_weights=run.latest_weights.assign(symbol=["A000001", "A000002"]),
    )

    snapshot = factory.build(mapped_run, BenchmarkConfig.default_kospi200())

    assert snapshot.sectors.latest_weighted.to_dict() == {"Technology": 0.6, "Utilities KR": 0.4}
    assert snapshot.exposure.latest_holdings["symbol"].tolist() == ["Alpha (000001)", "Beta (000002)"]


def test_performance_snapshot_factory_uses_fixed_252_day_rolling_window() -> None:
    run = _long_run()
    factory = PerformanceSnapshotFactory(
        benchmark_repo=BenchmarkRepository.from_frame(_long_benchmark_prices(run.equity.index)),
        sector_repo=SectorRepository.from_frame(_long_sector_map(run.equity.index.max()), prices=_long_asset_prices(run.equity.index)),
    )

    snapshot = factory.build(run, BenchmarkConfig.default_kospi200())

    rolling_sharpe = snapshot.rolling.series["rolling_sharpe"]
    rolling_beta = snapshot.rolling.series["rolling_beta"]
    expected_sharpe = annualized_sharpe(run.returns.iloc[-252:])

    assert_series_equal(rolling_sharpe.index.to_series(), run.returns.index.to_series())
    assert_series_equal(rolling_beta.index.to_series(), run.returns.index.to_series())
    assert rolling_sharpe.iloc[:251].isna().all()
    assert rolling_beta.iloc[:251].isna().all()
    assert pd.notna(rolling_sharpe.iloc[251])
    assert pd.notna(rolling_beta.iloc[251])
    assert pd.notna(rolling_sharpe.iloc[-1])
    assert pd.notna(rolling_beta.iloc[-1])
    assert rolling_sharpe.iloc[-1] == expected_sharpe


def _toy_run(latest_weights: pd.DataFrame | None | object = _DEFAULT) -> SavedRun:
    index = pd.date_range("2024-01-02", periods=8, freq="D")
    equity = pd.Series([100.0, 102.0, 101.0, 105.0, 103.0, 106.0, 108.0, 110.0], index=index, name="equity")
    returns = equity.pct_change().fillna(0.0).rename("returns")
    turnover = pd.Series([0.0, 0.1, 0.05, 0.08, 0.03, 0.07, 0.02, 0.01], index=index, name="turnover")
    weights = pd.DataFrame(
        {
            "A": [0.5, 0.5, 0.0, 0.6, 0.6, 0.6, 0.6, 0.6],
            "B": [0.5, 0.4, 0.7, 0.4, 0.4, 0.0, 0.0, 0.4],
            "C": [0.0, 0.1, 0.3, 0.0, 0.0, 0.4, 0.4, 0.0],
        },
        index=index,
    )
    if latest_weights is _DEFAULT:
        latest_weights = pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "target_weight": [0.6, 0.4],
                "abs_weight": [0.6, 0.4],
            }
        )
    qty = pd.DataFrame(
        {
            "A": [5, 5, 0, 6, 6, 6, 6, 6],
            "B": [5, 4, 7, 4, 4, 0, 0, 4],
            "C": [0, 1, 3, 0, 0, 4, 4, 0],
        },
        index=index,
    ).astype(float)

    return SavedRun(
        run_id="toy-run",
        path=Path("/tmp/toy-run"),
        config={"name": "Toy Strategy", "strategy": "trend_rank"},
        summary={},
        equity=equity,
        returns=returns,
        turnover=turnover,
        weights=weights,
        qty=qty,
        latest_weights=latest_weights,
    )


def _benchmark_prices() -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=8, freq="D")
    return pd.DataFrame(
        {
            "IKS200": [200.0, 201.0, 200.5, 202.0, 201.0, 202.5, 203.0, 204.0],
        },
        index=index,
    )


def _benchmark_ohlc_prices() -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=8, freq="D")
    columns = pd.MultiIndex.from_tuples(
        [
            ("IKS200", "open"),
            ("IKS200", "high"),
            ("IKS200", "low"),
            ("IKS200", "close"),
        ],
        names=["code", "field"],
    )
    return pd.DataFrame(
        [
            [199.0, 201.0, 198.0, 200.0],
            [200.0, 202.0, 199.5, 201.0],
            [200.0, 201.0, 199.0, 200.5],
            [201.0, 203.0, 200.0, 202.0],
            [200.5, 202.0, 200.0, 201.0],
            [201.0, 203.0, 200.5, 202.5],
            [202.0, 204.0, 201.5, 203.0],
            [203.0, 205.0, 202.0, 204.0],
        ],
        index=index,
        columns=columns,
    )


def _sector_map() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": ["Tech"],
            "B": ["Utilities"],
            "C": ["Health Care"],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )


def _asset_prices() -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=8, freq="D")
    return pd.DataFrame(
        {
            "A": [100.0, 102.0, 101.0, 104.0, 105.0, 106.0, 108.0, 110.0],
            "B": [100.0, 101.0, 100.0, 102.0, 103.0, 102.0, 103.0, 104.0],
            "C": [100.0, 100.0, 99.0, 98.0, 99.0, 100.0, 101.0, 100.0],
        },
        index=index,
    )


def _long_run() -> SavedRun:
    index = pd.date_range("2024-01-02", periods=260, freq="D")
    cycle = ([0.0, 0.01, -0.02, 0.015, -0.005] * 52)[:260]
    returns = pd.Series(cycle, index=index, name="returns")
    equity = (1.0 + returns).cumprod().mul(100.0).rename("equity")
    turnover = pd.Series(0.05, index=index, name="turnover")
    weights = pd.DataFrame({"A": 0.6, "B": 0.4, "C": 0.0}, index=index)
    qty = pd.DataFrame({"A": 6.0, "B": 4.0, "C": 0.0}, index=index)

    return SavedRun(
        run_id="long-run",
        path=Path("/tmp/long-run"),
        config={"name": "Long Strategy", "strategy": "trend_rank"},
        summary={},
        equity=equity,
        returns=returns,
        turnover=turnover,
        weights=weights,
        qty=qty,
        latest_weights=None,
    )


def _long_benchmark_prices(index: pd.DatetimeIndex) -> pd.DataFrame:
    returns = pd.Series([0.0] + [0.004] * (len(index) - 1), index=index)
    prices = (1.0 + returns).cumprod().mul(200.0)
    return pd.DataFrame({"IKS200": prices}, index=index)


def _long_sector_map(latest_date: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {"A": ["Tech"], "B": ["Utilities"], "C": ["Health Care"]},
        index=pd.DatetimeIndex([latest_date - pd.Timedelta(days=259)]),
    )


def _long_asset_prices(index: pd.DatetimeIndex) -> pd.DataFrame:
    asset_a = pd.Series([100.0 + step * 0.4 for step in range(len(index))], index=index)
    asset_b = pd.Series([100.0 + step * 0.2 for step in range(len(index))], index=index)
    asset_c = pd.Series([100.0 - step * 0.1 for step in range(len(index))], index=index)
    return pd.DataFrame({"A": asset_a, "B": asset_b, "C": asset_c}, index=index)


def _expected_sortino(returns: pd.Series) -> float:
    downside = returns.clip(upper=0.0)
    downside_deviation = float((downside.pow(2).mean() ** 0.5) * (252.0**0.5))
    if downside_deviation == 0.0:
        return 0.0
    return float(returns.mean() * 252.0 / downside_deviation)
