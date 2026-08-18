from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtesting.data import MarketData
from backtesting.strategies.emp008.reports.attribution import FactorAttributionResult, build_emp008_factor_attribution
from backtesting.strategies.emp008.strategy import run_emp008
from backtesting.strategies.emp008.data import Emp008Config
from backtesting.strategies.emp008.factor_registry import get_factor_set_definition
from backtesting.strategies.emp008.factor_pipeline import (
    PreparedEmp008Factors,
    load_and_prepare_emp008_factors,
    prepare_emp008_factors,
)
from backtesting.strategies.emp008.optimize import OptimizationResult


def _sample_market() -> MarketData:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    columns = ["A", "B"]
    return MarketData(
        frames={
            "close": pd.DataFrame([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]], index=dates, columns=columns),
            "market_cap": pd.DataFrame([[100.0, 300.0], [120.0, 280.0], [130.0, 270.0]], index=dates, columns=columns),
            "float_market_cap": pd.DataFrame(
                [[100.0, 300.0], [120.0, 280.0], [130.0, 270.0]],
                index=dates,
                columns=columns,
            ),
            "k200_yn": pd.DataFrame(True, index=dates, columns=columns),
            "sector_neutral_big": pd.DataFrame("Tech", index=dates, columns=columns),
            "bm_weights": pd.DataFrame(
                [[None, None], [0.11, 0.89], [0.20, 0.80]],
                index=dates,
                columns=columns,
            ),
        },
        universe=None,
        benchmark=None,
    )


def _sample_prepared(config: Emp008Config | None = None) -> PreparedEmp008Factors:
    active_config = config or Emp008Config()
    market = _sample_market()
    dates = market.frames["close"].index
    columns = market.frames["close"].columns
    alpha_factors = {
        "price_to_252d_high": pd.DataFrame([[0.05, -0.05], [0.06, -0.06], [0.07, -0.07]], index=dates, columns=columns),
        "earnings_momentum": pd.DataFrame([[0.08, -0.08], [0.09, -0.09], [0.10, -0.10]], index=dates, columns=columns),
        "dividend_yield_ttm": pd.DataFrame([[0.03, -0.03], [0.04, -0.04], [0.05, -0.05]], index=dates, columns=columns),
        "retail_flow": pd.DataFrame([[0.02, -0.02], [0.03, -0.03], [0.04, -0.04]], index=dates, columns=columns),
        "value": pd.DataFrame([[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]], index=dates, columns=columns),
        "ln_market_cap": pd.DataFrame([[1.0, -1.0], [1.5, -1.5], [2.0, -2.0]], index=dates, columns=columns),
    }
    sector_factors = {
        "Tech": pd.DataFrame([[0.0, 0.0], [0.1, -0.1], [0.2, -0.2]], index=dates, columns=columns),
    }
    return PreparedEmp008Factors(
        config=active_config,
        market=market,
        factor_set_definition=get_factor_set_definition(active_config.factor_set),
        raw_factors=dict(alpha_factors),
        alpha_factors=alpha_factors,
        sector_factors=sector_factors,
        close=market.frames["close"].astype(float),
        market_cap=market.frames["market_cap"].astype(float),
        float_market_cap=market.frames["float_market_cap"].astype(float),
        universe=market.frames["k200_yn"].astype(bool),
        sector=market.frames["sector_neutral_big"],
        benchmark_weights=market.frames["bm_weights"].astype(float).fillna(0.0),
        monthly_dates=tuple(dates),
    )


def _replace_prepared(
    prepared: PreparedEmp008Factors,
    *,
    close: pd.DataFrame | None = None,
    benchmark_weights: pd.DataFrame | None = None,
    monthly_dates: tuple[pd.Timestamp, ...] | None = None,
) -> PreparedEmp008Factors:
    return PreparedEmp008Factors(
        config=prepared.config,
        market=prepared.market,
        factor_set_definition=prepared.factor_set_definition,
        raw_factors=prepared.raw_factors,
        alpha_factors=prepared.alpha_factors,
        sector_factors=prepared.sector_factors,
        close=close if close is not None else prepared.close,
        market_cap=prepared.market_cap,
        float_market_cap=prepared.float_market_cap,
        universe=prepared.universe,
        sector=prepared.sector,
        benchmark_weights=benchmark_weights if benchmark_weights is not None else prepared.benchmark_weights,
        monthly_dates=monthly_dates if monthly_dates is not None else prepared.monthly_dates,
    )


def test_prepare_emp008_factors_uses_registry_metadata_and_preserves_registry_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _sample_market()
    config = Emp008Config(value_raw_winsor_quantile=0.15, value_zscore_cap=3.5)
    raw_factors = {
        "price_to_252d_high": pd.DataFrame(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            index=market.frames["close"].index,
            columns=["A", "B"],
        ),
        "earnings_momentum": pd.DataFrame(
            [[1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
            index=market.frames["close"].index,
            columns=["A", "B"],
        ),
        "dividend_yield_ttm": pd.DataFrame(
            [[2.1, 2.2], [2.3, 2.4], [2.5, 2.6]],
            index=market.frames["close"].index,
            columns=["A", "B"],
        ),
        "retail_flow": pd.DataFrame(
            [[3.1, 3.2], [3.3, 3.4], [3.5, 3.6]],
            index=market.frames["close"].index,
            columns=["A", "B"],
        ),
        "value": pd.DataFrame([[5.0, 10.0], [6.0, 11.0], [7.0, 12.0]], index=market.frames["close"].index, columns=["A", "B"]),
        "ln_market_cap": pd.DataFrame(
            [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]],
            index=market.frames["close"].index,
            columns=["A", "B"],
        ),
    }
    for name, frame in raw_factors.items():
        frame.attrs["factor_name"] = name
    calls: list[tuple[str, bool, float | None, float | None]] = []

    def fake_build_raw_factors(_market: MarketData, _config: Emp008Config) -> dict[str, pd.DataFrame]:
        return raw_factors

    def fake_preprocess(
        raw: pd.DataFrame,
        float_mktcap: pd.DataFrame,
        universe: pd.DataFrame,
        *,
        rank_transform: bool = False,
        winsor_quantile: float | None = None,
        zscore_cap: float | None = None,
    ) -> pd.DataFrame:
        del float_mktcap, universe
        calls.append((str(raw.attrs["factor_name"]), rank_transform, winsor_quantile, zscore_cap))
        return raw.astype(float)

    def fake_sector_exposures(
        sector: pd.DataFrame,
        float_mktcap: pd.DataFrame,
        universe: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        del float_mktcap, universe
        return {"Tech": sector.eq("Tech").astype(float)}

    monkeypatch.setattr(
        "backtesting.strategies.emp008.factor_pipeline.build_raw_factors",
        fake_build_raw_factors,
    )
    monkeypatch.setattr(
        "backtesting.strategies.emp008.factor_pipeline.preprocess_factor_frame",
        fake_preprocess,
    )
    monkeypatch.setattr(
        "backtesting.strategies.emp008.factor_pipeline.build_sector_active_exposures",
        fake_sector_exposures,
    )

    prepared = prepare_emp008_factors(market, config)

    assert list(prepared.raw_factors) == [
        "price_to_252d_high",
        "earnings_momentum",
        "dividend_yield_ttm",
        "retail_flow",
        "value",
        "ln_market_cap",
    ]
    assert list(prepared.alpha_factors) == [
        "price_to_252d_high",
        "earnings_momentum",
        "dividend_yield_ttm",
        "retail_flow",
        "value",
        "ln_market_cap",
    ]
    assert prepared.market_cap.equals(market.frames["market_cap"].astype(float))
    assert prepared.float_market_cap.equals(market.frames["float_market_cap"].astype(float))
    assert prepared.benchmark_weights.loc["2024-01-31"].tolist() == pytest.approx([0.25, 0.75])
    assert prepared.alpha_factors["value"].loc["2024-02-29", "A"] == pytest.approx(6.0)
    assert prepared.alpha_factors["ln_market_cap"].loc["2024-02-29", "A"] == 0.0
    assert prepared.monthly_dates == tuple(market.frames["close"].index)
    assert calls == [
        ("price_to_252d_high", False, None, None),
        ("earnings_momentum", False, None, None),
        ("dividend_yield_ttm", False, None, None),
        ("retail_flow", False, None, None),
        ("value", False, 0.15, 3.5),
        ("ln_market_cap", True, None, None),
    ]


def test_prepare_origin_keeps_ln_market_cap_unranked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _sample_market()
    config = Emp008Config(factor_set="origin")
    raw_factors = {
        name: market.frames["close"].astype(float).copy()
        for name in ("ln_market_cap", "momentum_12m", "dividend_yield_fy0")
    }
    for name, frame in raw_factors.items():
        frame.attrs["factor_name"] = name
    calls: list[tuple[str, bool]] = []

    monkeypatch.setattr(
        "backtesting.strategies.emp008.factor_pipeline.build_raw_factors",
        lambda _market, _config: raw_factors,
    )

    def fake_preprocess(
        raw: pd.DataFrame,
        float_mktcap: pd.DataFrame,
        universe: pd.DataFrame,
        *,
        rank_transform: bool = False,
        winsor_quantile: float | None = None,
        zscore_cap: float | None = None,
    ) -> pd.DataFrame:
        del float_mktcap, universe, winsor_quantile, zscore_cap
        calls.append((str(raw.attrs["factor_name"]), rank_transform))
        return raw

    monkeypatch.setattr(
        "backtesting.strategies.emp008.factor_pipeline.preprocess_factor_frame",
        fake_preprocess,
    )

    prepare_emp008_factors(market, config)

    assert calls == [
        ("ln_market_cap", False),
        ("momentum_12m", False),
        ("dividend_yield_fy0", False),
    ]


def test_load_and_prepare_emp008_factors_loads_market_then_prepares(monkeypatch: pytest.MonkeyPatch) -> None:
    market = _sample_market()
    config = Emp008Config()
    prepared = _sample_prepared(config)
    calls: list[object] = []

    def fake_load_market(*, parquet_dir: Path, start: str, end: str, config: Emp008Config) -> MarketData:
        calls.append((parquet_dir, start, end, config))
        return market

    def fake_prepare(loaded_market: MarketData, prepared_config: Emp008Config) -> PreparedEmp008Factors:
        calls.append((loaded_market, prepared_config))
        return prepared

    monkeypatch.setattr(
        "backtesting.strategies.emp008.factor_pipeline.load_emp008_market",
        fake_load_market,
    )
    monkeypatch.setattr(
        "backtesting.strategies.emp008.factor_pipeline.prepare_emp008_factors",
        fake_prepare,
    )

    result = load_and_prepare_emp008_factors(Path("parquet"), "2024-01-31", "2024-03-29", config)

    assert result is prepared
    assert calls == [
        (Path("parquet"), "2024-01-31", "2024-03-29", config),
        (market, config),
    ]


def test_run_emp008_uses_supplied_prepared_bundle_without_reloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _sample_prepared()
    seen: list[tuple[object, object, object, object]] = []

    def fail_loader(*args: object, **kwargs: object) -> None:
        raise AssertionError("loader should not run when prepared bundle is supplied")

    def fake_optimize_month(**kwargs: object) -> OptimizationResult:
        seen.append(
            (
                kwargs["close"],
                kwargs["bm_weights"],
                kwargs["alpha_factors"],
                kwargs["sector_factors"],
            )
        )
        return OptimizationResult(
            success=True,
            final_weights=pd.Series({"A": 0.6, "B": 0.4}),
            active_weights=pd.Series({"A": 0.1, "B": -0.1}),
            objective_value=1.0,
            tracking_error=0.02,
            sector_active_exposure_abs_max=0.0,
        )

    monkeypatch.setattr("backtesting.strategies.emp008.strategy.load_and_prepare_emp008_factors", fail_loader)
    monkeypatch.setattr("backtesting.strategies.emp008.strategy._optimize_month", fake_optimize_month)

    result = run_emp008(
        parquet_dir=Path("parquet"),
        start="2024-02-29",
        end="2024-03-29",
        prepared=prepared,
    )

    assert seen == [
        (
            prepared.close,
            prepared.benchmark_weights,
            prepared.alpha_factors,
            prepared.sector_factors,
        ),
        (
            prepared.close,
            prepared.benchmark_weights,
            prepared.alpha_factors,
            prepared.sector_factors,
        ),
    ]
    assert result.target_weights.index.tolist() == [pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-29")]
    assert result.active_weights.loc["2024-03-29", "A"] == pytest.approx(0.1)


def test_run_emp008_rejects_mismatched_config_for_prepared_bundle() -> None:
    prepared = _sample_prepared(Emp008Config(factor_set="origin"))

    with pytest.raises(ValueError, match="prepared/config mismatch"):
        run_emp008(
            parquet_dir=Path("parquet"),
            start="2024-02-29",
            end="2024-03-29",
            config=Emp008Config(factor_set="mfbt"),
            prepared=prepared,
        )


def test_run_emp008_reports_non_factor_set_config_differences() -> None:
    prepared = _sample_prepared(Emp008Config(risk_model="factor_idio"))

    with pytest.raises(ValueError, match="risk_model"):
        run_emp008(
            parquet_dir=Path("parquet"),
            start="2024-02-29",
            end="2024-03-29",
            config=Emp008Config(risk_model="direct_covariance"),
            prepared=prepared,
        )


def test_run_emp008_rejects_prepared_bundle_with_insufficient_end_coverage() -> None:
    prepared = _sample_prepared()

    with pytest.raises(ValueError, match="prepared data range"):
        run_emp008(
            parquet_dir=Path("parquet"),
            start="2024-02-29",
            end="2024-04-30",
            prepared=prepared,
        )


def test_run_emp008_rejects_prepared_bundle_when_monthly_dates_stop_before_requested_end_month() -> None:
    prepared = _sample_prepared()
    april_index = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30"])
    extended_close = pd.DataFrame(
        [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0]],
        index=april_index,
        columns=prepared.close.columns,
    )
    extended_bm = pd.DataFrame(
        [[0.25, 0.75], [0.11, 0.89], [0.20, 0.80], [0.22, 0.78]],
        index=april_index,
        columns=prepared.benchmark_weights.columns,
    )
    extended_prepared = _replace_prepared(
        prepared,
        close=extended_close,
        benchmark_weights=extended_bm,
        monthly_dates=tuple(pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])),
    )

    with pytest.raises(ValueError, match="monthly output range"):
        run_emp008(
            parquet_dir=Path("parquet"),
            start="2024-02-29",
            end="2024-04-30",
            prepared=extended_prepared,
        )


def test_run_emp008_accepts_last_business_day_in_requested_end_month(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _sample_prepared()
    june_index = pd.to_datetime(["2024-04-30", "2024-05-31", "2024-06-28"])
    close = pd.DataFrame(
        [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]],
        index=june_index,
        columns=prepared.close.columns,
    )
    bm = pd.DataFrame(
        [[0.30, 0.70], [0.25, 0.75], [0.20, 0.80]],
        index=june_index,
        columns=prepared.benchmark_weights.columns,
    )
    valid_prepared = _replace_prepared(
        prepared,
        close=close,
        benchmark_weights=bm,
        monthly_dates=tuple(june_index),
    )
    calls: list[pd.Timestamp] = []

    def fake_optimize_month(**kwargs: object) -> OptimizationResult:
        calls.append(kwargs["return_date"])
        return OptimizationResult(
            success=True,
            final_weights=pd.Series({"A": 0.6, "B": 0.4}),
            active_weights=pd.Series({"A": 0.1, "B": -0.1}),
            objective_value=1.0,
            tracking_error=0.02,
            sector_active_exposure_abs_max=0.0,
        )

    monkeypatch.setattr("backtesting.strategies.emp008.strategy._optimize_month", fake_optimize_month)

    result = run_emp008(
        parquet_dir=Path("parquet"),
        start="2024-05-31",
        end="2024-06-30",
        prepared=valid_prepared,
    )

    assert calls == [pd.Timestamp("2024-05-31"), pd.Timestamp("2024-06-28")]
    assert result.target_weights.index.tolist() == [pd.Timestamp("2024-05-31"), pd.Timestamp("2024-06-28")]


def test_build_emp008_factor_attribution_uses_supplied_prepared_bundle_without_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _sample_prepared()
    run_root = tmp_path / "run"
    weights_dir = run_root / "weights"
    weights_dir.mkdir(parents=True)
    pd.DataFrame(
        {"A": [0.10], "B": [-0.10]},
        index=pd.to_datetime(["2024-02-29"]),
    ).to_parquet(weights_dir / "active_weights.parquet")
    captured: dict[str, object] = {}

    def fail_loader(*args: object, **kwargs: object) -> None:
        raise AssertionError("attribution should not reload when prepared bundle is supplied")

    def fake_compute_attribution(**kwargs: object) -> FactorAttributionResult:
        captured.update(kwargs)
        index = pd.to_datetime(["2024-03-29"])
        monthly = pd.DataFrame(
            {
                "value": [0.001],
                "ln_market_cap": [-0.0002],
                "sector_total": [0.0],
                "specific": [0.0001],
                "alpha_total": [0.0008],
                "model_active_return": [0.0009],
            },
            index=index,
        )
        return FactorAttributionResult(
            monthly_contribution=monthly,
            cumulative_contribution=monthly[["value", "ln_market_cap", "sector_total", "specific"]].cumsum(),
            yearly_contribution=monthly[["value", "ln_market_cap", "sector_total", "specific"]].groupby(index.year).sum(),
            factor_summary_bps=pd.DataFrame({"total_bp": [9.0]}, index=["value"]),
            active_factor_exposure=pd.DataFrame({"value": [0.1]}, index=index),
            realized_factor_return=pd.DataFrame({"value": [0.01]}, index=index),
            reconciliation=pd.DataFrame({"actual_active_return": [0.0009]}, index=index),
        )

    monkeypatch.setattr("backtesting.strategies.emp008.reports.attribution.load_and_prepare_emp008_factors", fail_loader)
    monkeypatch.setattr("backtesting.strategies.emp008.reports.attribution._compute_attribution", fake_compute_attribution)
    monkeypatch.setattr(
        "backtesting.strategies.emp008.reports.attribution.write_factor_attribution",
        lambda result, output_dir: {"excel": str(output_dir / "factor_attribution.xlsx")},
    )

    payload = build_emp008_factor_attribution(
        parquet_dir=Path("parquet"),
        run_root=run_root,
        output_dir=tmp_path / "out",
        prepared=prepared,
    )

    assert captured["alpha_factors"] is prepared.alpha_factors
    assert captured["sector_factors"] is prepared.sector_factors
    assert payload["periods"] == 1
    assert payload["date_start"] == "2024-03-29"
    assert payload["date_end"] == "2024-03-29"


def test_build_emp008_factor_attribution_rejects_mismatched_config_for_prepared_bundle(
    tmp_path: Path,
) -> None:
    prepared = _sample_prepared(Emp008Config(factor_set="origin"))
    run_root = tmp_path / "run"
    weights_dir = run_root / "weights"
    weights_dir.mkdir(parents=True)
    pd.DataFrame(
        {"A": [0.10], "B": [-0.10]},
        index=pd.to_datetime(["2024-02-29"]),
    ).to_parquet(weights_dir / "active_weights.parquet")

    with pytest.raises(ValueError, match="prepared/config mismatch"):
        build_emp008_factor_attribution(
            parquet_dir=Path("parquet"),
            run_root=run_root,
            config=Emp008Config(factor_set="mfbt"),
            prepared=prepared,
        )


def test_build_emp008_factor_attribution_rejects_missing_required_target_date(
    tmp_path: Path,
) -> None:
    prepared = _sample_prepared()
    run_root = tmp_path / "run"
    weights_dir = run_root / "weights"
    weights_dir.mkdir(parents=True)
    pd.DataFrame(
        {"A": [0.10], "B": [-0.10]},
        index=pd.to_datetime(["2024-04-30"]),
    ).to_parquet(weights_dir / "active_weights.parquet")

    with pytest.raises(ValueError, match="missing required target dates"):
        build_emp008_factor_attribution(
            parquet_dir=Path("parquet"),
            run_root=run_root,
            prepared=prepared,
        )
