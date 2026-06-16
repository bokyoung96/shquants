import pandas as pd
import pytest

from backtesting.catalog import DatasetId
from backtesting.strategies.emp008.mfbt_emp008_experiments.active_weight_factor_plots import (
    active_weight_panel,
    build_strategy_config,
    factor_driver_summary,
)


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
