import warnings

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from backtesting.reporting.benchmarks import (
    BenchmarkRepository,
    SectorRepository,
    _load_display_name_maps,
    _read_historical_sector_frame,
    _read_quantwise_benchmark_frame,
    _read_static_sector_frame,
)
from backtesting.reporting.models import BenchmarkConfig
from root import RootPaths


def test_benchmark_repository_load_returns_uses_kospi200_price_path() -> None:
    index = pd.to_datetime(["2024-01-04", "2024-01-02", "2024-01-03"])
    frame = pd.DataFrame(
        {
            "IKS200": [102.0, 100.0, 101.0],
            "IKS001": [12.0, 10.0, 11.0],
        },
        index=index,
    )

    returns = BenchmarkRepository.from_frame(frame).load_returns(
        BenchmarkConfig.default_kospi200(),
        start="2024-01-02",
        end="2024-01-04",
    )

    expected_index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    expected = pd.Series(
        [0.0, (101.0 / 100.0) - 1.0, (102.0 / 101.0) - 1.0],
        index=expected_index,
        name="KOSPI200",
    )
    expected.index.name = "date"

    assert returns.name == "KOSPI200"
    assert list(returns.index) == list(expected_index)
    assert returns.iloc[-1] == expected.iloc[-1]
    assert_series_equal(returns, expected)


def test_benchmark_repository_load_returns_uses_close_from_ohlc_frame() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    columns = pd.MultiIndex.from_tuples(
        [
            ("IKS200", "open"),
            ("IKS200", "high"),
            ("IKS200", "low"),
            ("IKS200", "close"),
        ],
        names=["code", "field"],
    )
    frame = pd.DataFrame(
        [
            [99.0, 101.0, 98.0, 100.0],
            [100.0, 102.0, 99.0, 101.0],
            [101.0, 103.0, 100.0, 102.0],
        ],
        index=index,
        columns=columns,
    )

    series = BenchmarkRepository.from_frame(frame).load_series(
        BenchmarkConfig.default_kospi200(),
        start="2024-01-02",
        end="2024-01-04",
    )

    expected_prices = pd.Series([100.0, 101.0, 102.0], index=index, name="KOSPI200")
    expected_prices.index.name = "date"
    expected_returns = expected_prices.pct_change().fillna(0.0)

    assert_series_equal(series.prices, expected_prices)
    assert_series_equal(series.returns, expected_returns)


def test_sector_repository_latest_sector_weights_maps_latest_date() -> None:
    sector_index = pd.to_datetime(["2024-02-29", "2024-01-31"])
    sector_frame = pd.DataFrame(
        {
            "A": ["G20", "G10"],
            "B": ["G30", "G10"],
            "C": ["G40", "G15"],
        },
        index=sector_index,
    )
    weights = pd.DataFrame(
        {
            "A": [0.25],
            "B": [0.35],
            "C": [0.40],
        },
        index=pd.to_datetime(["2024-02-15"]),
    )

    exposure = SectorRepository.from_frame(sector_frame).latest_sector_weights(weights)

    expected = pd.Series({"G10": 0.6, "G15": 0.4})
    expected.index.name = None

    assert_series_equal(exposure, expected)


def test_sector_repository_exposes_latest_row_and_counts_without_internal_access() -> None:
    sector_frame = pd.DataFrame(
        {
            "A": ["Tech", "Energy"],
            "B": ["Utilities", "Utilities"],
            "C": ["Health Care", "Energy"],
        },
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    weights = pd.DataFrame(
        {
            "A": [0.6],
            "B": [0.4],
            "C": [0.0],
        },
        index=pd.to_datetime(["2024-02-15"]),
    )
    repo = SectorRepository.from_frame(sector_frame)

    latest_row = repo.latest_sector_row(pd.Timestamp("2024-02-15"))
    latest_count = repo.latest_sector_counts(weights)

    expected_row = pd.Series({"A": "Tech", "B": "Utilities", "C": "Health Care"}, name=pd.Timestamp("2024-01-31"))
    expected_count = pd.Series({"Tech": 1.0, "Utilities": 1.0}, name="count")
    expected_count.index.name = "sector"

    assert_series_equal(latest_row, expected_row)
    assert_series_equal(latest_count.sort_index(), expected_count.sort_index())


def test_sector_contribution_timeseries_avoids_pct_change_future_warning() -> None:
    sector_frame = pd.DataFrame(
        {
            "A": ["Tech"],
            "B": ["Utilities"],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )
    prices = pd.DataFrame(
        {
            "A": [100.0, None, 101.0],
            "B": [50.0, 50.5, 50.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    qty = pd.DataFrame(
        {
            "A": [6.0, 5.5, 5.0],
            "B": [4.0, 4.5, 5.0],
        },
        index=prices.index,
    )
    equity = pd.Series([100.0, 100.2, 100.4], index=prices.index)
    repo = SectorRepository.from_frame(sector_frame, prices=prices)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        contributions = repo.sector_contribution_timeseries(qty, equity)

    assert not [warning for warning in recorded if issubclass(warning.category, FutureWarning)]
    assert float(contributions.loc[pd.Timestamp("2024-01-04"), "Tech"]) <= 0.0
    assert_frame_equal(contributions.fillna(0.0), contributions.fillna(0.0).replace([float("inf"), float("-inf")], 0.0))


def test_sector_contribution_timeseries_matches_net_portfolio_return() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    sector_frame = pd.DataFrame(
        {
            "A": ["Tech"],
            "B": ["Utilities"],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )
    prices = pd.DataFrame(
        {
            "A": [100.0, 110.0, 108.0],
            "B": [100.0, 90.0, 92.0],
        },
        index=index,
    )
    qty = pd.DataFrame(
        {
            "A": [0.0, 0.5, 0.4],
            "B": [0.0, 0.5, 0.6],
        },
        index=index,
    )
    equity = pd.Series([100.0, 99.0, 98.5], index=index)

    contributions = SectorRepository.from_frame(sector_frame, prices=prices).sector_contribution_timeseries(qty, equity)
    portfolio_cumulative = equity.pct_change().fillna(0.0).cumsum()

    assert contributions.sum(axis=1).tolist() == pytest.approx(portfolio_cumulative.tolist())


def test_read_quantwise_benchmark_frame_extracts_codes_and_dates(tmp_path) -> None:
    raw = pd.DataFrame(
        [
            ["Refresh", "Last Update", None],
            ["Code", "IKS200", "IKS001"],
            ["Name", "KOSPI200", "KOSPI"],
            ["Item Code", "I100100", "I100100"],
            ["Unit", "P", "P"],
            ["D A T E", "종가지수", "종가지수"],
            [pd.Timestamp("2024-01-02"), 100.0, 200.0],
            [pd.Timestamp("2024-01-03"), 101.0, 201.0],
        ]
    )
    path = tmp_path / "qw_BM.xlsx"
    raw.to_excel(path, index=False, header=False)

    frame = _read_quantwise_benchmark_frame(path)

    expected = pd.DataFrame(
        {
            "IKS200": [100.0, 101.0],
            "IKS001": [200.0, 201.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    expected.index.name = "date"

    assert_frame_equal(frame, expected, check_dtype=False)


def test_read_quantwise_benchmark_frame_extracts_ohlc_multiindex(tmp_path) -> None:
    raw = pd.DataFrame(
        [
            ["Refresh", "Last Update", None, None, None],
            ["Code", "IKS200", "IKS200", "IKS200", "IKS200"],
            ["Name", "KOSPI200", "KOSPI200", "KOSPI200", "KOSPI200"],
            ["Item Code", "I100110", "I100120", "I100130", "I100100"],
            ["Unit", "P", "P", "P", "P"],
            ["D A T E", "시가지수", "고가지수", "저가지수", "종가지수"],
            [pd.Timestamp("2024-01-02"), 99.0, 101.0, 98.0, 100.0],
            [pd.Timestamp("2024-01-03"), 100.0, 102.0, 99.0, 101.0],
        ]
    )
    path = tmp_path / "qw_BM.xlsx"
    raw.to_excel(path, index=False, header=False)

    frame = _read_quantwise_benchmark_frame(path)

    expected_columns = pd.MultiIndex.from_tuples(
        [
            ("IKS200", "open"),
            ("IKS200", "high"),
            ("IKS200", "low"),
            ("IKS200", "close"),
        ],
        names=["code", "field"],
    )
    expected = pd.DataFrame(
        [[99.0, 101.0, 98.0, 100.0], [100.0, 102.0, 99.0, 101.0]],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        columns=expected_columns,
    )
    expected.index.name = "date"

    assert_frame_equal(frame, expected, check_dtype=False)


def test_load_display_name_maps_reads_sector_and_stock_sheets(tmp_path) -> None:
    path = tmp_path / "map.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Code": ["G10"], "Name": ["에너지"]}).to_excel(writer, sheet_name="sector_map", index=False)
        pd.DataFrame({"Ticker": ["A005930"], "Name": ["삼성전자"]}).to_excel(writer, sheet_name="Sheet3", index=False)

    sector_name_map, stock_name_map = _load_display_name_maps(path)

    assert sector_name_map == {"G10": "에너지"}
    assert stock_name_map == {"A005930": "삼성전자"}


def test_read_static_sector_frame_reads_etf_rows_from_map(tmp_path) -> None:
    path = tmp_path / "map.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(
            {
                "TICKER": ["A091160", "A005930"],
                "NAME": ["KODEX Semiconductor", "Samsung Electronics"],
                "GICS_SECTOR_NAME": ["ETF", "Information Technology"],
            }
        ).to_excel(writer, sheet_name="ticker_name_gics_sector_map", index=False)

    frame = _read_static_sector_frame(path, sector_value="ETF")

    expected = pd.DataFrame(
        {"A091160": ["ETF"]},
        index=pd.to_datetime(["1900-01-01"]),
    )
    expected.index.name = "date"

    assert_frame_equal(frame, expected)


def test_read_historical_sector_frame_pivots_long_excel(tmp_path) -> None:
    path = tmp_path / "snp_ksdq_gics_sector_big.xlsx"
    pd.DataFrame(
        {
            "DATE": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
            "TICKER": ["A091990", "091990", "A091990"],
            "GICS_SECTOR_LV1_NAME": ["Health Care", "Health Care", "Information Technology"],
        }
    ).to_excel(path, index=False)

    frame = _read_historical_sector_frame(path)

    expected = pd.DataFrame(
        {
            "A091990": ["Health Care", "Information Technology"],
        },
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    expected.index.name = "date"

    assert_frame_equal(frame, expected)


def test_default_sector_repository_for_kosdaq150_uses_kosdaq_family(monkeypatch) -> None:
    import backtesting.reporting.benchmarks as benchmarks
    from backtesting.catalog import DatasetId

    requested: list[DatasetId] = []

    def _fake_load_default_frame(dataset_id: DatasetId) -> pd.DataFrame:
        requested.append(dataset_id)
        index = pd.to_datetime(["2024-01-02"])
        if dataset_id is DatasetId.QW_KSDQ_WICS_SEC_BIG:
            return pd.DataFrame({"A": ["G45"]}, index=index)
        if dataset_id is DatasetId.QW_KSDQ_ADJ_C:
            return pd.DataFrame({"A": [100.0]}, index=index)
        if dataset_id is DatasetId.QW_BM:
            return pd.DataFrame({"IKS200": [200.0]}, index=index)
        raise AssertionError(f"unexpected dataset_id: {dataset_id}")

    monkeypatch.setattr(benchmarks, "_load_default_frame", _fake_load_default_frame)

    benchmark_repo, sector_repo = benchmarks.default_repositories_for_universe("kosdaq150")

    assert benchmark_repo is not None
    assert sector_repo is not None
    assert requested[:2] == [DatasetId.QW_KSDQ_WICS_SEC_BIG, DatasetId.QW_KSDQ_ADJ_C]


def test_default_sector_repository_for_kosdaq150_can_use_gics_source(monkeypatch, tmp_path) -> None:
    import backtesting.reporting.benchmarks as benchmarks
    from backtesting.catalog import DatasetId

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with pd.ExcelWriter(raw_dir / "map.xlsx") as writer:
        pd.DataFrame({"Ticker": ["A091990"], "Name": ["셀트리온헬스케어"]}).to_excel(writer, sheet_name="Sheet3", index=False)
    pd.DataFrame(
        {
            "DATE": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
            "TICKER": ["A091990", "A091990"],
            "GICS_SECTOR_LV1_NAME": ["Health Care", "Information Technology"],
            "GICS_SECTOR_LV2_NAME": ["Biotechnology", "Health Care Equipment & Services"],
        }
    ).to_excel(raw_dir / "snp_ksdq_gics_sector_big.xlsx", index=False)

    requested: list[DatasetId] = []

    def _fake_load_default_frame(dataset_id: DatasetId) -> pd.DataFrame:
        requested.append(dataset_id)
        index = pd.to_datetime(["2024-01-31", "2024-02-29"])
        if dataset_id is DatasetId.QW_KSDQ_ADJ_C:
            return pd.DataFrame({"A091990": [100.0, 101.0]}, index=index)
        if dataset_id is DatasetId.QW_BM:
            return pd.DataFrame({"IKS200": [200.0, 201.0]}, index=index)
        raise AssertionError(f"unexpected dataset_id: {dataset_id}")

    monkeypatch.setattr(benchmarks, "ROOT", RootPaths(tmp_path))
    monkeypatch.setattr(benchmarks, "_load_default_frame", _fake_load_default_frame)

    benchmark_repo, sector_repo = benchmarks.default_repositories_for_universe("kosdaq150", sector_source="gics")

    assert benchmark_repo is not None
    assert sector_repo.display_symbol("091990") == "셀트리온헬스케어 (091990)"
    assert sector_repo.latest_sector_row(pd.Timestamp("2024-02-29")).to_dict() == {"A091990": "Information Technology"}
    assert requested == [DatasetId.QW_KSDQ_ADJ_C, DatasetId.QW_BM]


def test_default_sector_repository_for_kosdaq150_prefers_lv1_column_from_gics_workbook(monkeypatch, tmp_path) -> None:
    import backtesting.reporting.benchmarks as benchmarks
    from backtesting.catalog import DatasetId

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    with pd.ExcelWriter(raw_dir / "map.xlsx") as writer:
        pd.DataFrame({"Ticker": ["A091990"], "Name": ["셀트리온헬스케어"]}).to_excel(writer, sheet_name="Sheet3", index=False)
    pd.DataFrame(
        {
            "DATE": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
            "TICKER": ["A091990", "A091990"],
            "GICS_SECTOR_LV1_NAME": ["Health Care", "Information Technology"],
            "GICS_SECTOR_LV2_NAME": ["Biotechnology", "Healthcare Equipment"],
        }
    ).to_excel(raw_dir / "snp_ksdq_gics_sector_big.xlsx", index=False)

    requested: list[DatasetId] = []

    def _fake_load_default_frame(dataset_id: DatasetId) -> pd.DataFrame:
        requested.append(dataset_id)
        index = pd.to_datetime(["2024-01-31", "2024-02-29"])
        if dataset_id is DatasetId.QW_KSDQ_ADJ_C:
            return pd.DataFrame({"A091990": [100.0, 101.0]}, index=index)
        if dataset_id is DatasetId.QW_BM:
            return pd.DataFrame({"IKS200": [200.0, 201.0]}, index=index)
        raise AssertionError(f"unexpected dataset_id: {dataset_id}")

    monkeypatch.setattr(benchmarks, "ROOT", RootPaths(tmp_path))
    monkeypatch.setattr(benchmarks, "_load_default_frame", _fake_load_default_frame)

    benchmark_repo, sector_repo = benchmarks.default_repositories_for_universe("kosdaq150", sector_source="gics")

    assert benchmark_repo is not None
    assert sector_repo.latest_sector_row(pd.Timestamp("2024-02-29")).to_dict() == {"A091990": "Information Technology"}
    assert requested == [DatasetId.QW_KSDQ_ADJ_C, DatasetId.QW_BM]


def test_default_sector_repository_for_etf_uses_etf_prices_and_map(monkeypatch, tmp_path) -> None:
    import backtesting.reporting.benchmarks as benchmarks
    from backtesting.catalog import DatasetId

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with pd.ExcelWriter(raw_dir / "map.xlsx") as writer:
        pd.DataFrame(
            {
                "TICKER": ["A091160", "A005930"],
                "NAME": ["KODEX Semiconductor", "Samsung Electronics"],
                "GICS_SECTOR_NAME": ["ETF", "Information Technology"],
            }
        ).to_excel(writer, sheet_name="ticker_name_gics_sector_map", index=False)

    requested: list[DatasetId] = []

    def _fake_load_default_frame(dataset_id: DatasetId) -> pd.DataFrame:
        requested.append(dataset_id)
        index = pd.to_datetime(["2024-01-02"])
        if dataset_id is DatasetId.QW_ETF_ADJ_C:
            return pd.DataFrame({"A091160": [100.0]}, index=index)
        if dataset_id is DatasetId.QW_BM:
            return pd.DataFrame({"IKS200": [200.0]}, index=index)
        raise AssertionError(f"unexpected dataset_id: {dataset_id}")

    monkeypatch.setattr(benchmarks, "ROOT", RootPaths(tmp_path))
    monkeypatch.setattr(benchmarks, "_load_default_frame", _fake_load_default_frame)

    benchmark_repo, sector_repo = benchmarks.default_repositories_for_universe("etf")

    assert benchmark_repo is not None
    assert sector_repo.display_symbol("091160") == "KODEX Semiconductor (091160)"
    assert sector_repo.latest_sector_row(pd.Timestamp("2024-01-02")).to_dict() == {"A091160": "ETF"}
    assert requested == [DatasetId.QW_ETF_ADJ_C, DatasetId.QW_BM]
