import pandas as pd
import pytest
from pathlib import Path

from backtesting.catalog import DatasetId
from backtesting.data import MarketData
from backtesting.strategies.emp008.experiments.active_weight_factor_plots import (
    DEFAULT_TICKERS,
    active_weight_panel,
    build_strategy_config,
    factor_driver_summary,
    factor_contribution_panel,
)
from backtesting.strategies.emp008.data import Emp008Config
from backtesting.strategies.emp008.factor_registry import get_factor_set_definition
from backtesting.strategies.emp008.factor_pipeline import PreparedEmp008Factors


def test_active_weight_panel_reports_strategy_active_weights_and_pair_gap() -> None:
    index = pd.to_datetime(["2024-01-31", "2024-02-29"])
    mfbt_active = pd.DataFrame(
        {"A005930": [0.01, -0.02], "A000660": [-0.03, 0.04]},
        index=index,
    )
    origin_active = pd.DataFrame(
        {"A005930": [0.03, -0.01], "A000660": [-0.01, 0.01]},
        index=index,
    )

    result = active_weight_panel(
        {"mfbt": mfbt_active, "origin": origin_active},
        tickers=("A005930", "A000660"),
    )

    assert result.index.names == ["date", "ticker"]
    assert result.loc[(index[0], "A005930"), "mfbt_active"] == pytest.approx(0.01)
    assert result.loc[(index[0], "A005930"), "origin_active"] == pytest.approx(0.03)
    assert result.loc[(index[0], "A005930"), "origin_minus_mfbt_active"] == pytest.approx(0.02)
    assert result.loc[(index[1], "A000660"), "origin_minus_mfbt_active"] == pytest.approx(-0.03)


def test_factor_driver_summary_sorts_by_mean_absolute_contribution() -> None:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-31", "2024-02-29"]), ["A005930"]],
        names=["date", "ticker"],
    )
    contributions = pd.DataFrame(
        {
            "strategy": ["mfbt", "mfbt"],
            "momentum": [0.001, -0.003],
            "value": [0.0001, 0.0002],
        },
        index=index,
    )

    result = factor_driver_summary(contributions, factor_columns=("momentum", "value"))

    assert result.index.tolist() == ["momentum", "value"]
    assert result.loc["momentum", "mean_abs_contribution_bp"] == pytest.approx(20.0)
    assert result.loc["value", "mean_abs_contribution_bp"] == pytest.approx(1.5)


def test_build_strategy_config_maps_mfbt_zcap5_to_value_cap_variant() -> None:
    config = build_strategy_config(
        strategy="mfbt_zcap5",
        tracking_error_annual=0.007,
        risk_model="factor_idio",
    )

    assert config.factor_set == "mfbt"
    assert config.value_zscore_cap == pytest.approx(5.0)
    assert config.value_raw_winsor_quantile is None


def test_build_strategy_config_maps_mfbt_wics_to_sector_neutral_variant() -> None:
    config = build_strategy_config(
        strategy="mfbt_wics",
        tracking_error_annual=0.007,
        risk_model="factor_idio",
    )

    assert config.factor_set == "mfbt"
    assert config.sector_neutral_dataset == DatasetId.QW_WICS_SEC_BIG


def test_factor_contribution_panel_uses_shared_prepared_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Emp008Config(factor_set="mfbt_origin_smallcap", risk_window=1)
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    close = pd.DataFrame(
        {
            "A005930": [10.0, 11.0, 12.0],
            "A000660": [20.0, 21.0, 22.0],
        },
        index=dates,
    )
    prepared = PreparedEmp008Factors(
        config=config,
        market=MarketData(
            frames={
                "close": close,
                "market_cap": close * 100.0,
                "float_market_cap": close * 100.0,
                "k200_yn": pd.DataFrame(True, index=dates, columns=close.columns),
                "sector_neutral_big": pd.DataFrame("Tech", index=dates, columns=close.columns),
                "bm_weights": pd.DataFrame(0.5, index=dates, columns=close.columns),
            },
            universe=None,
            benchmark=None,
        ),
        factor_set_definition=get_factor_set_definition(config.factor_set),
        raw_factors={
            "value": close.copy(),
            "ln_market_cap": close.copy(),
        },
        alpha_factors={
            "value": pd.DataFrame(
                [[0.2, -0.2], [0.3, -0.3], [0.4, -0.4]],
                index=dates,
                columns=close.columns,
            ),
            "ln_market_cap": pd.DataFrame(
                [[-0.1, 0.1], [-0.2, 0.2], [-0.3, 0.3]],
                index=dates,
                columns=close.columns,
            ),
        },
        sector_factors={
            "Tech": pd.DataFrame(
                [[0.0, 0.0], [0.1, -0.1], [0.2, -0.2]],
                index=dates,
                columns=close.columns,
            ),
        },
        close=close,
        market_cap=close * 100.0,
        float_market_cap=close * 100.0,
        universe=pd.DataFrame(True, index=dates, columns=close.columns),
        sector=pd.DataFrame("Tech", index=dates, columns=close.columns),
        benchmark_weights=pd.DataFrame(0.5, index=dates, columns=close.columns),
        monthly_dates=tuple(dates),
    )

    monkeypatch.setattr(
        "backtesting.strategies.emp008.experiments.active_weight_factor_plots.load_and_prepare_emp008_factors",
        lambda parquet_dir, start, end, config: prepared,
    )
    monkeypatch.setattr(
        "backtesting.strategies.emp008.experiments.active_weight_factor_plots.build_strategy_config",
        lambda strategy, tracking_error_annual, risk_model: config,
    )
    monkeypatch.setattr(
        "backtesting.strategies.emp008.experiments.active_weight_factor_plots.fit_cross_sectional_factor_returns",
        lambda exposures, excess_returns: type(
            "Regression",
            (),
            {
                "factor_returns": pd.Series({"value": 0.02, "ln_market_cap": -0.03, "Tech": 0.0}),
                "residuals": excess_returns * 0.0,
            },
        )(),
    )
    monkeypatch.setattr(
        "backtesting.strategies.emp008.experiments.active_weight_factor_plots.compute_expected_alpha",
        lambda factor_returns, alpha_factor_names, sector_factor_names, window: pd.Series(
            {"value": 0.02, "ln_market_cap": 0.03, "Tech": 0.0}
        ),
    )

    panel, factor_names = factor_contribution_panel(
        strategy="origin",
        parquet_dir=Path("parquet"),
        start="2024-02-29",
        end="2024-03-29",
        tickers=DEFAULT_TICKERS,
        tracking_error_annual=0.007,
        risk_model="factor_idio",
    )

    assert tuple(factor_names) == ("value", "ln_market_cap")
    assert panel.loc[(pd.Timestamp("2024-03-29"), "A005930"), "ln_market_cap"] == 0.0
    assert panel.loc[(pd.Timestamp("2024-03-29"), "A000660"), "value"] == pytest.approx(-0.008)
