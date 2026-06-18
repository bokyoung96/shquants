from __future__ import annotations

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.strategies import build_strategy, list_strategies


def test_signal_event_rotation_is_registered_with_kospi200_data_contract() -> None:
    strategies = list_strategies()

    assert "signal_event_rotation" in strategies
    assert "signal_event_rotation_selected" in strategies

    strategy = build_strategy("signal_event_rotation")
    dataset_values = {dataset.value for dataset in strategy.datasets}

    assert DatasetId.QW_K200_YN.value in dataset_values
    assert DatasetId.QW_BM.value in dataset_values
    assert DatasetId.QW_OP_NFY1.value in dataset_values
    assert DatasetId.QW_FOREIGN.value in dataset_values
    assert DatasetId.QW_INSTITUTION.value in dataset_values
    assert DatasetId.QW_RETAIL.value in dataset_values

    selected = build_strategy("signal_event_rotation_selected")
    assert selected.score_mode == "op12"
    assert selected.event_mode == "accel"
    assert selected.flow_gate == "retail_contra"
    assert selected.construction_mode == "k2"
    assert selected.risk_mode == "ls03"


def test_event_participation_ramps_after_new_event_and_resets() -> None:
    from backtesting.strategies.signal_event_rotation import _event_participation

    index = pd.date_range("2024-01-02", periods=6, freq="D")
    event = pd.DataFrame({"A": [False, True, False, False, False, False]}, index=index)
    hold = pd.DataFrame({"A": [False, True, True, True, False, True]}, index=index)

    ramp = _event_participation(event=event, hold=hold, steps=3)

    assert ramp["A"].tolist() == [0.0, 1 / 3, 2 / 3, 1.0, 0.0, 0.0]


def test_signal_event_rotation_rejects_unknown_modes() -> None:
    import pytest

    with pytest.raises(ValueError, match="score_mode"):
        build_strategy("signal_event_rotation", score_mode="curve_fit")

    with pytest.raises(ValueError, match="flow_gate"):
        build_strategy("signal_event_rotation", flow_gate="magic")
