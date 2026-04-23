from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backtesting.reporting.benchmarks import BenchmarkRepository, SectorRepository
from backtesting.reporting.snapshots import PerformanceSnapshotFactory
from dashboard.backend.api import get_dashboard_payload_service
from dashboard.backend.main import app, create_app
from dashboard.backend import main as dashboard_main
from dashboard.backend.services.dashboard_payload import DashboardPayloadService
from dashboard.backend.services.run_index import RunIndexService
from dashboard.strategies import DEFAULT_LAUNCH_CONFIG


def test_dashboard_app_serves_frontend_index_when_dist_exists(tmp_path: Path, monkeypatch) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "index.html").write_text("<html><body>dashboard</body></html>", encoding="utf-8")

    monkeypatch.setattr(dashboard_main, "get_frontend_dist_dir", lambda: dist_dir)
    client = TestClient(dashboard_main.create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "dashboard" in response.text


def test_session_endpoint_returns_default_selected_run_ids() -> None:
    client = TestClient(create_app(default_selected_run_ids=["momentum_20260405_100000"]))

    response = client.get("/api/session")

    assert response.status_code == 200
    assert response.json() == {"defaultSelectedRunIds": ["momentum_20260405_100000"]}


def test_create_app_rejects_requests_outside_frontend_dist(tmp_path: Path) -> None:
    frontend_dist = tmp_path / "dist"
    frontend_dist.mkdir(parents=True)
    (frontend_dist / "index.html").write_text("<html><body>dashboard</body></html>", encoding="utf-8")
    secret_path = tmp_path / "secret.txt"
    secret_path.write_text("secret", encoding="utf-8")

    client = TestClient(create_app(frontend_dist=frontend_dist))

    response = client.get("/..%2Fsecret.txt")

    assert response.status_code == 404


def test_create_app_does_not_route_unknown_api_paths_to_spa(tmp_path: Path) -> None:
    frontend_dist = tmp_path / "dist"
    frontend_dist.mkdir(parents=True)
    (frontend_dist / "index.html").write_text("<html><body>dashboard</body></html>", encoding="utf-8")

    client = TestClient(create_app(frontend_dist=frontend_dist))

    response = client.get("/api/unknown")

    assert response.status_code == 404


def test_create_app_fails_fast_when_index_html_is_missing(tmp_path: Path) -> None:
    frontend_dist = tmp_path / "dist"
    frontend_dist.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        create_app(frontend_dist=frontend_dist)


def test_dashboard_endpoint_includes_exposure_payload(tmp_path: Path) -> None:
    _write_saved_run(
        tmp_path,
        "alpha_20260405_100000",
        name="Alpha Strategy",
        final_equity=110.0,
        avg_turnover=0.03,
        weights=[[0.6, 0.4, 0.0], [0.55, 0.45, 0.0]],
    )

    client = TestClient(app)
    app.dependency_overrides[get_dashboard_payload_service] = lambda: _build_payload_service(tmp_path)

    response = client.get("/api/dashboard", params=[("run_ids", "alpha_20260405_100000")])

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["exposure"]["holdingsCount"] == [
        {
            "runId": "alpha_20260405_100000",
            "label": "Alpha Strategy",
            "points": [
                {"date": "2024-01-02", "value": 2.0},
                {"date": "2024-01-03", "value": 2.0},
            ],
        }
    ]
    assert payload["exposure"]["latestHoldings"] == {
        "alpha_20260405_100000": [
            {"symbol": "A", "targetWeight": 0.55, "absWeight": 0.55},
            {"symbol": "B", "targetWeight": 0.45, "absWeight": 0.45},
        ]
    }
    assert payload["exposure"]["latestHoldings"]["alpha_20260405_100000"][0]["symbol"] == "A"
    assert payload["exposure"]["sectorWeights"] == {
        "alpha_20260405_100000": [
            {"name": "Tech", "value": 0.55},
            {"name": "Utilities", "value": 0.45},
        ]
    }


def test_dashboard_payload_includes_launch_metadata(tmp_path: Path) -> None:
    _write_saved_run(
        tmp_path,
        "alpha_20260405_100000",
        name="Alpha Strategy",
        final_equity=110.0,
        avg_turnover=0.03,
        weights=[[0.6, 0.4, 0.0], [0.55, 0.45, 0.0]],
    )

    client = TestClient(app)
    app.dependency_overrides[get_dashboard_payload_service] = lambda: _build_payload_service(tmp_path)

    response = client.get("/api/dashboard", params=[("run_ids", "alpha_20260405_100000")])

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["launch"] == {
        "configuredStartDate": DEFAULT_LAUNCH_CONFIG.global_config.start,
        "configuredEndDate": DEFAULT_LAUNCH_CONFIG.global_config.end,
        "capital": DEFAULT_LAUNCH_CONFIG.global_config.capital,
        "schedule": DEFAULT_LAUNCH_CONFIG.global_config.schedule,
        "fillMode": DEFAULT_LAUNCH_CONFIG.global_config.fill_mode,
        "fee": DEFAULT_LAUNCH_CONFIG.global_config.fee,
        "sellTax": DEFAULT_LAUNCH_CONFIG.global_config.sell_tax,
        "slippage": DEFAULT_LAUNCH_CONFIG.global_config.slippage,
        "benchmark": {
            "kind": "shared",
            "shared": {"code": "IKS200", "name": "KOSPI200"},
            "strategies": [
                {
                    "strategy": "momentum",
                    "label": "Momentum",
                    "benchmark": {"code": "IKS200", "name": "KOSPI200"},
                },
            ],
        },
        "asOfDate": "2024-01-03",
    }


def test_dashboard_payload_launch_benchmark_uses_shared_dashboard_default(tmp_path: Path) -> None:
    _write_saved_run(
        tmp_path,
        "macro_20260405_120000",
        name="Macro Strategy",
        strategy="macro",
        final_equity=102.0,
        avg_turnover=0.02,
        weights=[[0.6, 0.4, 0.0], [0.6, 0.4, 0.0]],
        benchmark={"code": "SPX", "name": "S&P 500"},
    )

    client = TestClient(app)
    app.dependency_overrides[get_dashboard_payload_service] = lambda: _build_payload_service(tmp_path)

    response = client.get("/api/dashboard", params=[("run_ids", "macro_20260405_120000")])

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["launch"]["benchmark"] == {
        "kind": "shared",
        "shared": {"code": "IKS200", "name": "KOSPI200"},
        "strategies": [
            {
                "strategy": "momentum",
                "label": "Momentum",
                "benchmark": {"code": "IKS200", "name": "KOSPI200"},
            },
        ],
    }
    assert payload["context"]["macro_20260405_120000"]["benchmark"] == {"code": "SPX", "name": "S&P 500"}


def test_dashboard_payload_uses_kosdaq150_defaults_when_run_is_kosdaq150(tmp_path: Path) -> None:
    _write_saved_run(
        tmp_path,
        "alpha_20260405_100000",
        name="Alpha Strategy",
        final_equity=110.0,
        avg_turnover=0.03,
        weights=[[0.6, 0.4, 0.0], [0.55, 0.45, 0.0]],
    )
    config_path = tmp_path / "alpha_20260405_100000" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["universe_id"] = "kosdaq150"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    client = TestClient(app)
    app.dependency_overrides[get_dashboard_payload_service] = lambda: _build_payload_service(
        tmp_path,
        benchmark_frame=pd.DataFrame(
            {"IKS200": [200.0, 201.0], "IKQ150": [300.0, 303.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
        sector_frame=pd.DataFrame(
            {"A": ["KQ Tech", "KQ Tech"], "B": ["KQ Utilities", "KQ Utilities"], "C": ["KQ Consumer", "KQ Consumer"]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
    )

    response = client.get("/api/dashboard", params=[("run_ids", "alpha_20260405_100000")])

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["context"]["alpha_20260405_100000"]["benchmark"] == {"code": "IKQ150", "name": "KOSDAQ150"}
    assert payload["performance"]["benchmarks"] == [
        {
            "runId": "alpha_20260405_100000",
            "label": "KOSDAQ150",
            "points": [
                {"date": "2024-01-02", "value": 100.0},
                {"date": "2024-01-03", "value": 101.0},
            ],
        }
    ]
    assert payload["exposure"]["sectorWeights"] == {
        "alpha_20260405_100000": [
            {"name": "KQ Tech", "value": 0.55},
            {"name": "KQ Utilities", "value": 0.45},
        ]
    }


def test_dashboard_payload_includes_rolling_correlation(tmp_path: Path) -> None:
    dates = [date.date().isoformat() for date in pd.bdate_range("2024-01-02", periods=260)]
    benchmark_frame = pd.DataFrame(
        {
            "IKS200": [200.0 + (index * 0.12) for index in range(len(dates))],
            "SPX": [500.0 + (index * 0.05) for index in range(len(dates))],
        },
        index=pd.to_datetime(dates),
    )
    strategy_equity = pd.Series(
        [100.0 + (index * 0.1) + ((index % 7) * 0.03) for index in range(len(dates))],
        index=pd.to_datetime(dates),
        name="equity",
    )
    expected_correlation = strategy_equity.pct_change().fillna(0.0).rolling(252, min_periods=252).corr(
        benchmark_frame["IKS200"].pct_change().fillna(0.0)
    )
    _write_saved_run(
        tmp_path,
        "alpha_20240430_100000",
        name="Alpha Strategy",
        final_equity=126.0,
        avg_turnover=0.03,
        dates=dates,
        equity_values=strategy_equity.tolist(),
        weights=[[0.6, 0.4, 0.0] for _ in dates],
    )

    client = TestClient(app)
    app.dependency_overrides[get_dashboard_payload_service] = lambda: _build_payload_service(tmp_path, benchmark_frame=benchmark_frame)

    response = client.get("/api/dashboard", params=[("run_ids", "alpha_20240430_100000")])

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    correlation = payload["rolling"]["rollingCorrelation"][0]
    assert correlation["runId"] == "alpha_20240430_100000"
    assert correlation["label"] == "Alpha Strategy"
    assert correlation["benchmark"] == {"code": "IKS200", "name": "KOSPI200"}
    assert correlation["window"] == 252
    assert len(correlation["points"]) == 9
    assert correlation["points"][0]["date"] == dates[251]
    assert correlation["points"][-1]["date"] == dates[-1]
    assert correlation["points"][0]["value"] == pytest.approx(expected_correlation.iloc[251])
    assert correlation["points"][-1]["value"] == pytest.approx(expected_correlation.iloc[-1])


def test_dashboard_payload_includes_monthly_return_distribution(tmp_path: Path) -> None:
    monthly_dates = ["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30", "2024-05-31", "2024-06-28"]
    _write_saved_run(
        tmp_path,
        "alpha_20240430_100000",
        name="Alpha Strategy",
        final_equity=112.0,
        avg_turnover=0.03,
        dates=monthly_dates,
        equity_values=[100.0, 106.0, 103.0, 109.0, 107.0, 112.0],
        monthly_returns_values=[0.06, -0.028301886792452824, 0.05825242718446602, -0.01834862385321101, 0.04672897196261672, 0.0],
        weights=[[0.6, 0.4, 0.0] for _ in monthly_dates],
    )

    client = TestClient(app)
    app.dependency_overrides[get_dashboard_payload_service] = lambda: _build_payload_service(
        tmp_path,
        benchmark_frame=pd.DataFrame(
            {"IKS200": [200.0, 204.0, 202.0, 208.0, 207.0, 210.0], "SPX": [500.0, 505.0, 503.0, 510.0, 508.0, 512.0]},
            index=pd.to_datetime(monthly_dates),
        ),
    )

    response = client.get("/api/dashboard", params=[("run_ids", "alpha_20240430_100000")])

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    daily_distribution = payload["research"]["returnDistribution"]["alpha_20240430_100000"]
    assert sum(bin_entry["count"] for bin_entry in daily_distribution) == len(monthly_dates)
    monthly_distribution = payload["research"]["monthlyReturnDistribution"]["alpha_20240430_100000"]
    assert len(monthly_distribution) == 6
    assert monthly_distribution[0]["count"] == 1
    assert monthly_distribution[-1]["count"] == 1
    assert sum(bin_entry["count"] for bin_entry in monthly_distribution) == 6
    assert monthly_distribution[0]["start"] < 0.0
    assert monthly_distribution[-1]["end"] > 0.0


def test_dashboard_payload_includes_latest_holdings_winners_and_losers(tmp_path: Path) -> None:
    _write_saved_run(
        tmp_path,
        "alpha_20260405_100000",
        name="Alpha Strategy",
        final_equity=110.0,
        avg_turnover=0.03,
        weights=[[0.6, 0.4, 0.0], [0.55, 0.45, 0.0]],
        latest_weights_rows=[
            {"symbol": "AAPL", "target_weight": 0.40, "abs_weight": 0.40},
            {"symbol": "MSFT", "target_weight": 0.25, "abs_weight": 0.25},
            {"symbol": "NVDA", "target_weight": 0.15, "abs_weight": 0.15},
            {"symbol": "AMZN", "target_weight": 0.10, "abs_weight": 0.10},
            {"symbol": "GOOG", "target_weight": 0.06, "abs_weight": 0.06},
            {"symbol": "TSLA", "target_weight": 0.04, "abs_weight": 0.04},
        ],
        latest_holdings_return_rows=[
            {"symbol": "AAPL", "return_since_latest_rebalance": -0.04},
            {"symbol": "MSFT", "return_since_latest_rebalance": 0.01},
            {"symbol": "NVDA", "return_since_latest_rebalance": 0.05},
            {"symbol": "AMZN", "return_since_latest_rebalance": 0.10},
            {"symbol": "GOOG", "return_since_latest_rebalance": 0.15},
            {"symbol": "TSLA", "return_since_latest_rebalance": 0.30},
            {"symbol": "META", "return_since_latest_rebalance": 0.99},
        ],
    )

    client = TestClient(app)
    app.dependency_overrides[get_dashboard_payload_service] = lambda: _build_payload_service(tmp_path)

    response = client.get("/api/dashboard", params=[("run_ids", "alpha_20260405_100000")])

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    winners = payload["exposure"]["latestHoldingsWinners"]["alpha_20260405_100000"]
    losers = payload["exposure"]["latestHoldingsLosers"]["alpha_20260405_100000"]
    assert len(winners) == 5
    assert len(losers) == 5
    assert [entry["symbol"] for entry in winners] == ["TSLA", "GOOG", "AMZN", "NVDA", "MSFT"]
    assert [entry["symbol"] for entry in losers] == ["AAPL", "MSFT", "NVDA", "AMZN", "GOOG"]
    assert "META" not in {entry["symbol"] for entry in winners + losers}
    assert winners[0]["returnSinceLatestRebalance"] > winners[-1]["returnSinceLatestRebalance"]
    assert losers[0]["returnSinceLatestRebalance"] < losers[-1]["returnSinceLatestRebalance"]
    assert winners[0]["targetWeight"] == pytest.approx(0.04)
    assert losers[0]["targetWeight"] == pytest.approx(0.40)


def test_dashboard_latest_holdings_returns_use_latest_rebalance_weights_not_just_symbol_cohort(tmp_path: Path) -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    _write_saved_run(
        tmp_path,
        "alpha_20260405_100000",
        name="Alpha Strategy",
        final_equity=102.0,
        avg_turnover=0.03,
        dates=dates,
        equity_values=[100.0, 101.0, 101.5, 102.0],
        weights=[
            [0.60, 0.40, 0.0],
            [0.55, 0.45, 0.0],
            [0.50, 0.50, 0.0],
            [0.50, 0.50, 0.0],
        ],
    )

    prices_frame = pd.DataFrame(
        {
            "A": [50.0, 100.0, 100.0, 102.0],
            "B": [100.0, 100.0, 100.0, 105.0],
            "C": [100.0, 100.0, 100.0, 100.0],
        },
        index=pd.to_datetime(dates),
    )

    client = TestClient(app)
    app.dependency_overrides[get_dashboard_payload_service] = lambda: _build_payload_service(
        tmp_path,
        benchmark_frame=pd.DataFrame(
            {"IKS200": [200.0, 201.0, 202.0, 203.0], "SPX": [500.0, 501.0, 502.0, 503.0]},
            index=pd.to_datetime(dates),
        ),
        prices_frame=prices_frame,
    )

    response = client.get("/api/dashboard", params=[("run_ids", "alpha_20260405_100000")])

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    winners = payload["exposure"]["latestHoldingsWinners"]["alpha_20260405_100000"]
    losers = payload["exposure"]["latestHoldingsLosers"]["alpha_20260405_100000"]
    assert winners[0]["symbol"] == "B"
    assert winners[0]["returnSinceLatestRebalance"] == pytest.approx(0.05)
    assert losers[0]["symbol"] == "A"
    assert losers[0]["returnSinceLatestRebalance"] == pytest.approx(0.02)


def test_dashboard_latest_holdings_returns_tolerate_small_float_residue_in_final_weights(tmp_path: Path) -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    _write_saved_run(
        tmp_path,
        "alpha_20260405_100000",
        name="Alpha Strategy",
        final_equity=102.0,
        avg_turnover=0.03,
        dates=dates,
        equity_values=[100.0, 101.0, 101.5, 102.0],
        weights=[
            [0.60, 0.40, 0.0],
            [0.55, 0.45, 0.0],
            [0.50000000001, 0.49999999999, 0.0],
            [0.50, 0.50, 0.0],
        ],
    )

    prices_frame = pd.DataFrame(
        {
            "A": [50.0, 100.0, 100.0, 102.0],
            "B": [100.0, 100.0, 100.0, 105.0],
            "C": [100.0, 100.0, 100.0, 100.0],
        },
        index=pd.to_datetime(dates),
    )

    client = TestClient(app)
    app.dependency_overrides[get_dashboard_payload_service] = lambda: _build_payload_service(
        tmp_path,
        benchmark_frame=pd.DataFrame(
            {"IKS200": [200.0, 201.0, 202.0, 203.0], "SPX": [500.0, 501.0, 502.0, 503.0]},
            index=pd.to_datetime(dates),
        ),
        prices_frame=prices_frame,
    )

    response = client.get("/api/dashboard", params=[("run_ids", "alpha_20260405_100000")])

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    winners = payload["exposure"]["latestHoldingsWinners"]["alpha_20260405_100000"]
    losers = payload["exposure"]["latestHoldingsLosers"]["alpha_20260405_100000"]
    assert winners[0]["symbol"] == "B"
    assert winners[0]["returnSinceLatestRebalance"] == pytest.approx(0.05)
    assert losers[0]["symbol"] == "A"
    assert losers[0]["returnSinceLatestRebalance"] == pytest.approx(0.02)


def test_dashboard_returns_single_mode_payload(tmp_path: Path) -> None:
    _write_saved_run(
        tmp_path,
        "alpha_20260405_100000",
        name="Alpha Strategy",
        final_equity=110.0,
        avg_turnover=0.03,
        weights=[[0.6, 0.4, 0.0], [0.55, 0.45, 0.0]],
    )
    _write_saved_run(
        tmp_path,
        "omega_20260405_110000",
        name="Omega Strategy",
        strategy="momentum",
        final_equity=120.0,
        avg_turnover=0.04,
        weights=[[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]],
        top_n=25,
    )

    client = TestClient(app)
    app.dependency_overrides[get_dashboard_payload_service] = lambda: _build_payload_service(tmp_path)

    response = client.get("/api/dashboard", params=[("run_ids", "alpha_20260405_100000")])

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "single"
    assert payload["selectedRunIds"] == ["alpha_20260405_100000"]
    assert [run["runId"] for run in payload["availableRuns"]] == [
        "omega_20260405_110000",
        "alpha_20260405_100000",
    ]
    assert payload["metrics"] == {
        "alpha_20260405_100000": {
            "label": "Alpha Strategy",
            "cumulativeReturn": pytest.approx(0.1),
            "cagr": pytest.approx(164_238.77066398552),
            "annualVolatility": pytest.approx(0.7937253933193773),
            "sharpe": pytest.approx(15.874507866387544),
            "sortino": 0.0,
            "calmar": 0.0,
            "maxDrawdown": 0.0,
            "finalEquity": 110.0,
            "avgTurnover": pytest.approx(0.03),
            "alpha": pytest.approx(0.0),
            "beta": pytest.approx(20.0),
            "trackingError": pytest.approx(0.7540391236534092),
            "informationRatio": pytest.approx(15.874507866387544),
        }
    }
    assert payload["context"] == {
        "alpha_20260405_100000": {
            "label": "Alpha Strategy",
            "strategy": "momentum",
            "startDate": "2024-01-02",
            "endDate": "2024-01-03",
            "asOfDate": "2024-01-03",
            "benchmark": {"code": "IKS200", "name": "KOSPI200"},
        }
    }
    assert payload["performance"]["series"] == [
        {
            "runId": "alpha_20260405_100000",
            "label": "Alpha Strategy",
            "points": [
                {"date": "2024-01-02", "value": 100.0},
                {"date": "2024-01-03", "value": 110.0},
            ],
        }
    ]
    assert payload["performance"]["benchmark"] == [
        {"date": "2024-01-02", "value": 100.0},
        {"date": "2024-01-03", "value": 100.49999999999999},
    ]
    assert payload["performance"]["benchmarks"] == [
        {
            "runId": "alpha_20260405_100000",
            "label": "KOSPI200",
            "points": [
                {"date": "2024-01-02", "value": 100.0},
                {"date": "2024-01-03", "value": 100.49999999999999},
            ],
        }
    ]
    assert payload["performance"]["drawdowns"] == [
        {
            "runId": "alpha_20260405_100000",
            "label": "Alpha Strategy",
            "points": [
                {"date": "2024-01-02", "value": 0.0},
                {"date": "2024-01-03", "value": 0.0},
            ],
        }
    ]
    assert payload["rolling"] == {"rollingSharpe": [], "rollingBeta": [], "rollingCorrelation": []}
    assert payload["exposure"]["holdingsCount"] == [
        {
            "runId": "alpha_20260405_100000",
            "label": "Alpha Strategy",
            "points": [
                {"date": "2024-01-02", "value": 2.0},
                {"date": "2024-01-03", "value": 2.0},
            ],
        }
    ]
    assert payload["exposure"]["latestHoldings"] == {
        "alpha_20260405_100000": [
            {"symbol": "A", "targetWeight": 0.55, "absWeight": 0.55},
            {"symbol": "B", "targetWeight": 0.45, "absWeight": 0.45},
        ]
    }
    assert payload["exposure"]["sectorWeights"] == {
        "alpha_20260405_100000": [
            {"name": "Tech", "value": 0.55},
            {"name": "Utilities", "value": 0.45},
        ]
    }
    assert payload["research"]["focus"] == {"kind": "all-selected", "label": "All Selected", "value": None}
    assert payload["research"]["sectorContributionMethod"] == "weighted-asset-return-attribution"
    assert payload["research"]["monthlyHeatmap"] == {
        "alpha_20260405_100000": [
            {"year": 2024, "month": 1, "value": pytest.approx(0.1)},
        ]
    }
    assert payload["research"]["returnDistribution"]["alpha_20260405_100000"]
    assert payload["research"]["yearlyExcessReturns"] == {
        "alpha_20260405_100000": [
            {"date": "2024-12-31", "value": pytest.approx(0.095)},
        ]
    }
    assert payload["research"]["sectorWeightSeries"] == {
        "alpha_20260405_100000": [
            {
                "name": "Tech",
                "points": [
                    {"date": "2024-01-02", "value": 0.6},
                    {"date": "2024-01-03", "value": 0.55},
                ],
            },
            {
                "name": "Utilities",
                "points": [
                    {"date": "2024-01-02", "value": 0.4},
                    {"date": "2024-01-03", "value": 0.45},
                ],
            },
            {
                "name": "Health Care",
                "points": [
                    {"date": "2024-01-02", "value": 0.0},
                    {"date": "2024-01-03", "value": 0.0},
                ],
            },
        ]
    }
    assert payload["research"]["sectorContributionSeries"] == {
        "alpha_20260405_100000": [
            {
                "name": "Tech",
                "points": [
                    {"date": "2024-01-02", "value": 0.0},
                    {"date": "2024-01-03", "value": pytest.approx(0.055)},
                ],
            },
            {
                "name": "Utilities",
                "points": [
                    {"date": "2024-01-02", "value": 0.0},
                    {"date": "2024-01-03", "value": pytest.approx(0.045)},
                ],
            },
            {
                "name": "Health Care",
                "points": [
                    {"date": "2024-01-02", "value": 0.0},
                    {"date": "2024-01-03", "value": 0.0},
                ],
            },
        ]
    }
    assert payload["research"]["drawdownEpisodes"] == {"alpha_20260405_100000": []}


def test_dashboard_returns_multi_mode_payload_for_repeated_run_ids(tmp_path: Path) -> None:
    _write_saved_run(
        tmp_path,
        "alpha_20260405_100000",
        name="Alpha Strategy",
        final_equity=110.0,
        avg_turnover=0.03,
        weights=[[0.6, 0.4, 0.0], [0.55, 0.45, 0.0]],
    )
    _write_saved_run(
        tmp_path,
        "omega_20260405_110000",
        name="Omega Strategy",
        final_equity=120.0,
        avg_turnover=0.04,
        weights=[[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]],
    )

    client = TestClient(app)
    app.dependency_overrides[get_dashboard_payload_service] = lambda: _build_payload_service(tmp_path)

    response = client.get(
        "/api/dashboard",
        params=[("run_ids", "omega_20260405_110000"), ("run_ids", "omega_20260405_110000")],
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "single"
    assert payload["selectedRunIds"] == ["omega_20260405_110000"]
    assert set(payload["metrics"]) == {"omega_20260405_110000"}
    assert set(payload["context"]) == {"omega_20260405_110000"}
    assert payload["performance"]["benchmark"] == [
        {"date": "2024-01-02", "value": 100.0},
        {"date": "2024-01-03", "value": 100.49999999999999},
    ]
    assert {entry["runId"] for entry in payload["performance"]["benchmarks"]} == {"omega_20260405_110000"}
    assert {entry["runId"] for entry in payload["performance"]["series"]} == {"omega_20260405_110000"}
    assert {entry["runId"] for entry in payload["performance"]["drawdowns"]} == {"omega_20260405_110000"}
    assert {entry["runId"] for entry in payload["rolling"]["rollingSharpe"]} == set()
    assert {entry["runId"] for entry in payload["exposure"]["holdingsCount"]} == {"omega_20260405_110000"}
    assert set(payload["exposure"]["latestHoldings"]) == {"omega_20260405_110000"}
    assert set(payload["exposure"]["sectorWeights"]) == {"omega_20260405_110000"}
    assert payload["exposure"]["latestHoldings"]["omega_20260405_110000"] == [
        {"symbol": "B", "targetWeight": 0.5, "absWeight": 0.5},
        {"symbol": "C", "targetWeight": 0.3, "absWeight": 0.3},
        {"symbol": "A", "targetWeight": 0.2, "absWeight": 0.2},
    ]
    assert set(payload["research"]["monthlyHeatmap"]) == {"omega_20260405_110000"}
    assert set(payload["research"]["returnDistribution"]) == {"omega_20260405_110000"}
    assert set(payload["research"]["yearlyExcessReturns"]) == {"omega_20260405_110000"}
    assert set(payload["research"]["sectorWeightSeries"]) == {"omega_20260405_110000"}
    assert set(payload["research"]["sectorContributionSeries"]) == {"omega_20260405_110000"}
    assert set(payload["research"]["drawdownEpisodes"]) == {"omega_20260405_110000"}


def test_dashboard_returns_per_run_benchmark_series_for_multi_run_selection(tmp_path: Path) -> None:
    _write_saved_run(
        tmp_path,
        "alpha_20260405_100000",
        name="Alpha Strategy",
        final_equity=110.0,
        avg_turnover=0.03,
        weights=[[0.6, 0.4, 0.0], [0.55, 0.45, 0.0]],
    )
    _write_saved_run(
        tmp_path,
        "macro_20260405_120000",
        name="Macro Strategy",
        final_equity=95.0,
        avg_turnover=0.02,
        weights=[[0.2, 0.5, 0.3], [0.1, 0.4, 0.5]],
        benchmark={"code": "SPX", "name": "S&P 500"},
    )

    client = TestClient(app)
    app.dependency_overrides[get_dashboard_payload_service] = lambda: _build_payload_service(tmp_path)

    response = client.get(
        "/api/dashboard",
        params=[("run_ids", "alpha_20260405_100000"), ("run_ids", "macro_20260405_120000")],
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "multi"
    assert payload["context"]["alpha_20260405_100000"]["benchmark"] == {"code": "IKS200", "name": "KOSPI200"}
    assert payload["context"]["macro_20260405_120000"]["benchmark"] == {"code": "SPX", "name": "S&P 500"}
    assert payload["performance"]["benchmark"] is None
    assert payload["performance"]["benchmarks"] == [
        {
            "runId": "alpha_20260405_100000",
            "label": "KOSPI200",
            "points": [
                {"date": "2024-01-02", "value": 100.0},
                {"date": "2024-01-03", "value": 100.49999999999999},
            ],
        },
        {
            "runId": "macro_20260405_120000",
            "label": "S&P 500",
            "points": [
                {"date": "2024-01-02", "value": 100.0},
                {"date": "2024-01-03", "value": 101.0},
            ],
        },
    ]


def test_dashboard_skips_non_finite_latest_holdings_values(tmp_path: Path) -> None:
    _write_saved_run(
        tmp_path,
        "alpha_20260405_100000",
        name="Alpha Strategy",
        final_equity=110.0,
        avg_turnover=0.03,
        weights=[[0.6, 0.4, 0.0], [0.55, 0.45, 0.0]],
        latest_weights_rows=[
            {"symbol": "A", "target_weight": 0.55, "abs_weight": 0.55},
            {"symbol": "B", "target_weight": float("nan"), "abs_weight": float("nan")},
        ],
    )

    client = TestClient(app)
    app.dependency_overrides[get_dashboard_payload_service] = lambda: _build_payload_service(tmp_path)

    response = client.get("/api/dashboard", params=[("run_ids", "alpha_20260405_100000")])

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["exposure"]["latestHoldings"] == {
        "alpha_20260405_100000": [
            {"symbol": "A", "targetWeight": 0.55, "absWeight": 0.55},
        ]
    }


def test_dashboard_skips_non_finite_sector_weights_values(tmp_path: Path) -> None:
    _write_saved_run(
        tmp_path,
        "alpha_20260405_100000",
        name="Alpha Strategy",
        final_equity=110.0,
        avg_turnover=0.03,
        weights=[[0.6, 0.4, 0.0], [0.55, 0.45, 0.0]],
    )

    client = TestClient(app)
    app.dependency_overrides[get_dashboard_payload_service] = lambda: _build_payload_service(
        tmp_path,
        sector_frame=pd.DataFrame(
            {"A": ["Tech"], "B": ["Utilities"], "C": [float("nan")]},
            index=pd.to_datetime(["2024-01-03"]),
        ),
    )

    response = client.get("/api/dashboard", params=[("run_ids", "alpha_20260405_100000")])

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["exposure"]["sectorWeights"] == {
        "alpha_20260405_100000": [
            {"name": "Tech", "value": 0.55},
            {"name": "Utilities", "value": 0.45},
        ]
    }
    assert payload["research"]["sectorWeightSeries"]["alpha_20260405_100000"] == [
        {
            "name": "Tech",
            "points": [
                {"date": "2024-01-02", "value": 0.0},
                {"date": "2024-01-03", "value": 0.55},
            ],
        },
        {
            "name": "Utilities",
            "points": [
                {"date": "2024-01-02", "value": 0.0},
                {"date": "2024-01-03", "value": 0.45},
            ],
        },
    ]


def test_dashboard_payload_preserves_korean_sector_and_stock_names(tmp_path: Path) -> None:
    _write_saved_run(
        tmp_path,
        "alpha_20260405_100000",
        name="Alpha Strategy",
        final_equity=110.0,
        avg_turnover=0.03,
        weights=[[0.6, 0.4, 0.0], [0.55, 0.45, 0.0]],
        latest_weights_rows=[
            {"symbol": "A005930", "target_weight": 0.55, "abs_weight": 0.55},
            {"symbol": "A000660", "target_weight": 0.45, "abs_weight": 0.45},
        ],
    )

    client = TestClient(app)
    app.dependency_overrides[get_dashboard_payload_service] = lambda: _build_payload_service(
        tmp_path,
        sector_frame=pd.DataFrame(
            {"A": ["G45", "G45"], "B": ["G40", "G40"], "C": ["G10", "G10"]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
        sector_name_map={"G45": "IT", "G40": "금융", "G10": "에너지"},
        stock_name_map={"A005930": "삼성전자", "A000660": "SK하이닉스"},
    )

    response = client.get("/api/dashboard", params=[("run_ids", "alpha_20260405_100000")])

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["exposure"]["latestHoldings"] == {
        "alpha_20260405_100000": [
            {"symbol": "삼성전자 (005930)", "targetWeight": 0.55, "absWeight": 0.55},
            {"symbol": "SK하이닉스 (000660)", "targetWeight": 0.45, "absWeight": 0.45},
        ]
    }
    assert payload["exposure"]["sectorWeights"]["alpha_20260405_100000"][0]["name"] == "IT"


def test_dashboard_returns_controlled_error_for_non_directory_run_entry(tmp_path: Path) -> None:
    (tmp_path / "broken_20260405_120000").write_text("not a directory", encoding="utf-8")

    client = TestClient(app)
    app.dependency_overrides[get_dashboard_payload_service] = lambda: _build_payload_service(tmp_path)

    response = client.get("/api/dashboard", params=[("run_ids", "broken_20260405_120000")])

    app.dependency_overrides.clear()
    assert response.status_code == 404
    assert response.json() == {"detail": "unknown run_id: broken_20260405_120000"}


def test_dashboard_returns_controlled_error_for_unreadable_run_directory(tmp_path: Path) -> None:
    _write_incomplete_run(tmp_path, "broken_20260405_130000")

    client = TestClient(app)
    app.dependency_overrides[get_dashboard_payload_service] = lambda: _build_payload_service(tmp_path)

    response = client.get("/api/dashboard", params=[("run_ids", "broken_20260405_130000")])

    app.dependency_overrides.clear()
    assert response.status_code == 404
    assert response.json()["detail"].startswith("unable to read run_id: broken_20260405_130000")


def _build_payload_service(
    runs_root: Path,
    sector_frame: pd.DataFrame | None = None,
    benchmark_frame: pd.DataFrame | None = None,
    prices_frame: pd.DataFrame | None = None,
    sector_name_map: dict[str, str] | None = None,
    stock_name_map: dict[str, str] | None = None,
) -> DashboardPayloadService:
    if benchmark_frame is None:
        benchmark_frame = pd.DataFrame(
            {"IKS200": [200.0, 201.0], "SPX": [500.0, 505.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )
    if sector_frame is None:
        sector_frame = pd.DataFrame(
            {
                "A": ["Tech"] * len(benchmark_frame.index),
                "B": ["Utilities"] * len(benchmark_frame.index),
                "C": ["Health Care"] * len(benchmark_frame.index),
            },
            index=pd.to_datetime(benchmark_frame.index),
        )
    if prices_frame is None:
        prices_frame = pd.DataFrame(
            {
                "A": [100.0 + 10.0 * index for index in range(len(benchmark_frame.index))],
                "B": [100.0 + 10.0 * index for index in range(len(benchmark_frame.index))],
                "C": [100.0 for _ in benchmark_frame.index],
            },
            index=pd.to_datetime(benchmark_frame.index),
        )
    return DashboardPayloadService(
        runs_root=runs_root,
        run_index_service=RunIndexService(runs_root),
        snapshot_factory=PerformanceSnapshotFactory(
            benchmark_repo=BenchmarkRepository.from_frame(
                benchmark_frame
            ),
            sector_repo=SectorRepository.from_frame(
                sector_frame,
                prices=prices_frame,
                sector_name_map=sector_name_map,
                stock_name_map=stock_name_map,
            ),
        ),
    )


def _write_saved_run(
    root: Path,
    run_id: str,
    *,
    name: str,
    strategy: str = "momentum",
    final_equity: float,
    avg_turnover: float,
    weights: list[list[float]],
    dates: list[str] | None = None,
    equity_values: list[float] | None = None,
    turnover_values: list[float] | None = None,
    monthly_returns_values: list[float] | None = None,
    benchmark: dict[str, object] | None = None,
    latest_weights_rows: list[dict[str, object]] | None = None,
    latest_holdings_return_rows: list[dict[str, object]] | None = None,
    top_n: int | None = None,
) -> None:
    run_dir = root / run_id
    series_dir = run_dir / "series"
    positions_dir = run_dir / "positions"
    series_dir.mkdir(parents=True)
    positions_dir.mkdir()

    config: dict[str, object] = {
        "name": name,
        "strategy": strategy,
        "start": "2024-01-02",
        "end": "2024-01-03",
    }
    if benchmark is not None:
        config["benchmark"] = benchmark
    if top_n is not None:
        config["top_n"] = top_n
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps({"final_equity": final_equity, "avg_turnover": avg_turnover}),
        encoding="utf-8",
    )

    if dates is None:
        dates = ["2024-01-02", "2024-01-03"]
    date_index = pd.to_datetime(dates)
    if equity_values is None:
        equity_values = [100.0, final_equity]
    equity = pd.Series(equity_values, index=date_index, name="equity")
    returns = equity.pct_change().fillna(0.0).rename("returns")
    if turnover_values is None:
        turnover_values = [avg_turnover for _ in dates]
    turnover = pd.Series(turnover_values, index=date_index, name="turnover")
    weights_frame = pd.DataFrame(weights, columns=["A", "B", "C"], index=date_index)
    price_frame = pd.DataFrame(
        {
            "A": [100.0 + 10.0 * index for index in range(len(date_index))],
            "B": [100.0 + 10.0 * index for index in range(len(date_index))],
            "C": [100.0 for _ in date_index],
        },
        index=date_index,
    )
    qty_frame = weights_frame.mul(equity, axis=0).div(price_frame).fillna(0.0)
    if monthly_returns_values is not None:
        monthly_dates = pd.to_datetime(dates[: len(monthly_returns_values)])
        monthly_returns = pd.Series(monthly_returns_values, index=monthly_dates, name="monthly_returns")
    else:
        monthly_returns = None

    equity.to_csv(series_dir / "equity.csv", index_label="date")
    returns.to_csv(series_dir / "returns.csv", index_label="date")
    turnover.to_csv(series_dir / "turnover.csv", index_label="date")
    if monthly_returns is not None:
        monthly_returns.to_csv(series_dir / "monthly_returns.csv", index_label="date")
    weights_frame.to_parquet(positions_dir / "weights.parquet")
    qty_frame.to_parquet(positions_dir / "qty.parquet")
    if latest_weights_rows is not None:
        pd.DataFrame(latest_weights_rows).to_csv(positions_dir / "latest_weights.csv", index=False)
    if latest_holdings_return_rows is not None:
        pd.DataFrame(latest_holdings_return_rows).to_csv(positions_dir / "latest_holdings_returns.csv", index=False)


def _write_incomplete_run(root: Path, run_id: str) -> None:
    run_dir = root / run_id
    series_dir = run_dir / "series"
    positions_dir = run_dir / "positions"
    series_dir.mkdir(parents=True)
    positions_dir.mkdir()

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "name": "Broken Strategy",
                "strategy": "momentum",
                "start": "2024-01-02",
                "end": "2024-01-03",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"final_equity": 100.0, "avg_turnover": 0.01}),
        encoding="utf-8",
    )
