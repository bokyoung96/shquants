from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import backtesting.strategies.emp008.strategy as emp008_strategy
from backtesting.strategies.emp008.data import Emp008Config
from backtesting.strategies.emp008.factor_registry import FactorDirection
from backtesting.strategies.emp008.factor_timing import (
    FactorTimingConfig,
    decide_factor_timing,
)
from backtesting.strategies.emp008.optimize import OptimizationResult
from backtesting.strategies.emp008.strategy import Emp008Result


def _returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "size": [-0.01, -0.02, -0.01, 0.99],
            "momentum": [-0.01, -0.02, -0.01, 0.99],
            "earnings": [-0.10, 0.02, 0.02, 0.99],
        },
        index=pd.to_datetime(["2023-10-31", "2023-11-30", "2023-12-29", "2024-01-31"]),
    )


def _directions() -> dict[str, FactorDirection]:
    return {
        "size": FactorDirection.LOW,
        "momentum": FactorDirection.HIGH,
        "earnings": FactorDirection.HIGH,
    }


def _weights() -> pd.Series:
    return pd.Series({"size": 0.3, "momentum": 0.4, "earnings": 0.3})


def test_disabled_timing_returns_normalized_base_weights_without_diagnostics() -> None:
    decision = decide_factor_timing(
        factor_returns=_returns(),
        factor_directions=_directions(),
        base_weights=pd.Series({"size": 3.0, "momentum": 4.0, "earnings": 3.0}),
        rebalance_date=pd.Timestamp("2024-01-31"),
        config=None,
    )

    pd.testing.assert_series_equal(decision.weights, _weights())
    assert decision.diagnostics.empty


def test_momentum_timing_classifies_directional_strong_neutral_and_weak_states() -> None:
    decision = decide_factor_timing(
        factor_returns=_returns(),
        factor_directions=_directions(),
        base_weights=_weights(),
        rebalance_date=pd.Timestamp("2024-01-31"),
        config=FactorTimingConfig(fast_lookback=2, slow_lookback=3),
    )

    diagnostics = decision.diagnostics.set_index("factor")
    assert diagnostics.loc["size", "state"] == "strong"
    assert diagnostics.loc["size", "multiplier"] == pytest.approx(1.25)
    assert diagnostics.loc["momentum", "state"] == "weak"
    assert diagnostics.loc["momentum", "multiplier"] == pytest.approx(0.75)
    assert diagnostics.loc["earnings", "state"] == "neutral"
    assert diagnostics.loc["earnings", "multiplier"] == pytest.approx(1.0)
    assert decision.weights.sum() == pytest.approx(1.0)
    assert decision.weights.ge(0.0).all()


def test_timing_excludes_factor_return_stamped_with_rebalance_date() -> None:
    config = FactorTimingConfig(fast_lookback=2, slow_lookback=3)
    original = decide_factor_timing(
        factor_returns=_returns(),
        factor_directions=_directions(),
        base_weights=_weights(),
        rebalance_date=pd.Timestamp("2024-01-31"),
        config=config,
    )
    mutated_returns = _returns()
    mutated_returns.loc[pd.Timestamp("2024-01-31")] = -999.0
    mutated = decide_factor_timing(
        factor_returns=mutated_returns,
        factor_directions=_directions(),
        base_weights=_weights(),
        rebalance_date=pd.Timestamp("2024-01-31"),
        config=config,
    )

    pd.testing.assert_series_equal(original.weights, mutated.weights)
    pd.testing.assert_frame_equal(original.diagnostics, mutated.diagnostics)
    assert original.diagnostics["last_signal_date"].max() == pd.Timestamp("2023-12-29")


def test_insufficient_history_keeps_base_weights_and_records_status() -> None:
    decision = decide_factor_timing(
        factor_returns=_returns().iloc[:2],
        factor_directions=_directions(),
        base_weights=_weights(),
        rebalance_date=pd.Timestamp("2024-01-31"),
        config=FactorTimingConfig(fast_lookback=2, slow_lookback=3),
    )

    pd.testing.assert_series_equal(decision.weights, _weights())
    assert set(decision.diagnostics["state"]) == {"insufficient_history"}
    assert set(decision.diagnostics["multiplier"]) == {1.0}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"policy": "macro"}, "policy"),
        ({"fast_lookback": 0}, "lookback"),
        ({"fast_lookback": 13, "slow_lookback": 12}, "fast_lookback"),
        ({"weak_multiplier": 0.0}, "multiplier"),
    ],
)
def test_factor_timing_config_rejects_invalid_settings(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FactorTimingConfig(**kwargs)


def test_factor_timing_rejects_missing_direction_metadata() -> None:
    directions = _directions()
    directions.pop("earnings")

    with pytest.raises(ValueError, match="missing factor directions"):
        decide_factor_timing(
            factor_returns=_returns(),
            factor_directions=directions,
            base_weights=_weights(),
            rebalance_date=pd.Timestamp("2024-01-31"),
            config=FactorTimingConfig(fast_lookback=2, slow_lookback=3),
        )


def test_emp008_config_and_result_keep_timing_disabled_by_default() -> None:
    result = Emp008Result(
        target_weights=pd.DataFrame(),
        active_weights=pd.DataFrame(),
        diagnostics=pd.DataFrame(),
    )

    assert Emp008Config().factor_timing is None
    assert result.factor_timing.empty


def test_emp008_result_writes_timing_files_only_when_diagnostics_exist(tmp_path) -> None:
    index = pd.to_datetime(["2024-01-31"])
    timing = pd.DataFrame(
        {
            "rebalance_date": index,
            "factor": ["momentum"],
            "direction": ["high"],
            "base_weight": [1.0],
            "fast_return": [0.01],
            "slow_return": [0.02],
            "state": ["strong"],
            "multiplier": [1.25],
            "timed_weight": [1.0],
            "last_signal_date": pd.to_datetime(["2023-12-29"]),
        }
    )
    enabled_dir = tmp_path / "enabled"
    Emp008Result(
        target_weights=pd.DataFrame({"A": [1.0]}, index=index),
        active_weights=pd.DataFrame({"A": [0.0]}, index=index),
        diagnostics=pd.DataFrame({"target_date": index, "success": [True]}),
        factor_timing=timing,
    ).write_outputs(enabled_dir)

    assert (enabled_dir / "factor_timing.csv").exists()
    assert (enabled_dir / "factor_timing.parquet").exists()

    disabled_dir = tmp_path / "disabled"
    Emp008Result(
        target_weights=pd.DataFrame({"A": [1.0]}, index=index),
        active_weights=pd.DataFrame({"A": [0.0]}, index=index),
        diagnostics=pd.DataFrame({"target_date": index, "success": [True]}),
    ).write_outputs(disabled_dir)

    assert not (disabled_dir / "factor_timing.csv").exists()
    assert not (disabled_dir / "factor_timing.parquet").exists()


def test_optimize_month_applies_timed_weights_to_expected_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factor_date = pd.Timestamp("2024-01-31")
    return_date = pd.Timestamp("2024-02-29")
    dates = pd.DatetimeIndex([factor_date, return_date])
    tickers = ["A", "B"]
    close = pd.DataFrame([[10.0, 20.0], [11.0, 19.0]], index=dates, columns=tickers)
    bm_weights = pd.DataFrame([[0.5, 0.5], [0.5, 0.5]], index=dates, columns=tickers)
    factor_frame = pd.DataFrame([[-1.0, 1.0], [-1.0, 1.0]], index=dates, columns=tickers)
    history_dates = list(pd.to_datetime(["2023-10-31", "2023-11-30", "2023-12-29"]))
    factor_return_rows = [
        pd.Series({"ln_market_cap": -0.01, "earnings_momentum": -0.01})
        for _ in history_dates
    ]
    residual_rows = [pd.Series({"A": 0.0, "B": 0.0}) for _ in history_dates]
    captured_alpha: list[pd.Series] = []

    monkeypatch.setattr(
        emp008_strategy,
        "fit_cross_sectional_factor_returns",
        lambda *_: SimpleNamespace(
            factor_returns=pd.Series({"ln_market_cap": 0.99, "earnings_momentum": 0.99}),
            residuals=pd.Series({"A": 0.0, "B": 0.0}),
        ),
    )
    monkeypatch.setattr(
        emp008_strategy,
        "compute_expected_alpha",
        lambda *_args, **_kwargs: pd.Series({"ln_market_cap": 1.0, "earnings_momentum": 1.0}),
    )
    monkeypatch.setattr(
        emp008_strategy,
        "factor_covariance",
        lambda *_: pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["ln_market_cap", "earnings_momentum"],
            columns=["ln_market_cap", "earnings_momentum"],
        ),
    )
    monkeypatch.setattr(
        emp008_strategy,
        "residual_variance",
        lambda *_: pd.Series({"A": 0.1, "B": 0.1}),
    )

    def fake_optimize_active_weights(**kwargs: object) -> OptimizationResult:
        captured_alpha.append(kwargs["expected_alpha"].copy())  # type: ignore[union-attr]
        return OptimizationResult(
            success=True,
            final_weights=pd.Series({"A": 0.5, "B": 0.5}),
            active_weights=pd.Series({"A": 0.0, "B": 0.0}),
            objective_value=0.0,
            tracking_error=0.0,
            sector_active_exposure_abs_max=0.0,
        )

    monkeypatch.setattr(emp008_strategy, "optimize_active_weights", fake_optimize_active_weights)
    timing_rows: list[pd.DataFrame] = []

    result = emp008_strategy._optimize_month(
        close=close,
        bm_weights=bm_weights,
        alpha_factors={"ln_market_cap": factor_frame, "earnings_momentum": factor_frame},
        sector_factors={},
        factor_date=factor_date,
        return_date=return_date,
        factor_return_rows=factor_return_rows,
        residual_rows=residual_rows,
        factor_return_dates=history_dates,
        stock_excess_return_rows=[],
        stock_excess_return_dates=[],
        alpha_factor_names=["ln_market_cap", "earnings_momentum"],
        sector_factor_names=[],
        factor_weights=pd.Series({"ln_market_cap": 1.0, "earnings_momentum": 1.0}),
        factor_timing_rows=timing_rows,
        config=Emp008Config(
            risk_window=1,
            factor_timing=FactorTimingConfig(fast_lookback=2, slow_lookback=3),
        ),
        run_optimization=True,
    )

    assert result is not None
    assert captured_alpha[0].loc["ln_market_cap"] == pytest.approx(0.625)
    assert captured_alpha[0].loc["earnings_momentum"] == pytest.approx(0.375)
    assert timing_rows[0].set_index("factor").loc["ln_market_cap", "state"] == "strong"
    assert timing_rows[0].set_index("factor").loc["earnings_momentum", "state"] == "weak"
