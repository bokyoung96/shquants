import pandas as pd
import pytest

from backtesting.construction.sector_rotation import SectorRotationLongShort
from backtesting.signals.base import SignalBundle


def test_sector_rotation_long_short_builds_dollar_neutral_book() -> None:
    index = pd.to_datetime(["2024-01-02"])
    alpha = pd.DataFrame(
        {
            "A": [10.0],
            "B": [8.0],
            "C": [2.0],
            "D": [1.0],
            "E": [9.0],
            "F": [0.0],
        },
        index=index,
    )
    sector = pd.DataFrame(
        {
            "A": ["Tech"],
            "B": ["Tech"],
            "C": ["Finance"],
            "D": ["Finance"],
            "E": ["Energy"],
            "F": ["Energy"],
        },
        index=index,
    )
    long_sector = pd.DataFrame(
        {"Tech": [True], "Finance": [False], "Energy": [False]},
        index=index,
    )
    short_sector = pd.DataFrame(
        {"Tech": [False], "Finance": [True], "Energy": [False]},
        index=index,
    )
    sector_weight_basis = pd.DataFrame(
        {
            "A": [70.0],
            "B": [30.0],
            "C": [30.0],
            "D": [70.0],
            "E": [100.0],
            "F": [100.0],
        },
        index=index,
    )
    bundle = SignalBundle(
        alpha=alpha,
        context={
            "sector": sector,
            "long_sector": long_sector,
            "short_sector": short_sector,
            "sector_weight_basis": sector_weight_basis,
            "tradable": alpha.notna(),
        },
    )

    result = SectorRotationLongShort(long_count=2, short_count=2).build(bundle)

    weights = result.base_target_weights.loc[index[0]]
    assert weights["A"] == pytest.approx(0.5)
    assert weights["B"] == pytest.approx(0.5)
    assert weights["C"] == pytest.approx(-0.5)
    assert weights["D"] == pytest.approx(-0.5)
    assert weights["E"] == pytest.approx(0.0)
    assert weights["F"] == pytest.approx(0.0)
    assert weights.sum() == pytest.approx(0.0)
    assert weights.clip(lower=0.0).sum() == pytest.approx(1.0)
    assert (-weights.clip(upper=0.0)).sum() == pytest.approx(1.0)
    assert bool(result.meta["selected_long"].loc[index[0], "A"])
    assert bool(result.meta["selected_short"].loc[index[0], "D"])


def test_sector_rotation_budgets_multiple_sectors_by_kospi200_weight_basis() -> None:
    index = pd.to_datetime(["2024-01-02"])
    alpha = pd.DataFrame(
        {"A": [10.0], "B": [1.0], "C": [9.0], "D": [0.0], "E": [3.0], "F": [2.0]},
        index=index,
    )
    sector = pd.DataFrame(
        {
            "A": ["Tech"],
            "B": ["Tech"],
            "C": ["Energy"],
            "D": ["Energy"],
            "E": ["Finance"],
            "F": ["Finance"],
        },
        index=index,
    )
    long_sector = pd.DataFrame(
        {"Tech": [True], "Energy": [True], "Finance": [False]},
        index=index,
    )
    short_sector = pd.DataFrame(
        {"Tech": [False], "Energy": [False], "Finance": [True]},
        index=index,
    )
    sector_weight_basis = pd.DataFrame(
        {
            "A": [25.0],
            "B": [75.0],
            "C": [150.0],
            "D": [150.0],
            "E": [80.0],
            "F": [20.0],
        },
        index=index,
    )
    bundle = SignalBundle(
        alpha=alpha,
        context={
            "sector": sector,
            "long_sector": long_sector,
            "short_sector": short_sector,
            "sector_weight_basis": sector_weight_basis,
        },
    )

    result = SectorRotationLongShort(long_count=2, short_count=1).build(bundle)

    weights = result.base_target_weights.loc[index[0]]
    assert weights["A"] == pytest.approx(0.25)
    assert weights["C"] == pytest.approx(0.75)
    assert weights["F"] == pytest.approx(-1.0)
    assert weights.sum() == pytest.approx(0.0)
    assert result.group_long_budget.loc[index[0], "Tech"] == pytest.approx(0.25)
    assert result.group_long_budget.loc[index[0], "Energy"] == pytest.approx(0.75)
    assert result.group_short_budget.loc[index[0], "Finance"] == pytest.approx(1.0)


def test_sector_rotation_reduces_side_exposure_when_no_qualified_sector_exists() -> None:
    index = pd.to_datetime(["2024-01-02"])
    alpha = pd.DataFrame({"A": [5.0], "B": [1.0]}, index=index)
    sector = pd.DataFrame({"A": ["Tech"], "B": ["Tech"]}, index=index)
    long_sector = pd.DataFrame({"Tech": [True]}, index=index)
    short_sector = pd.DataFrame({"Tech": [False]}, index=index)
    bundle = SignalBundle(
        alpha=alpha,
        context={"sector": sector, "long_sector": long_sector, "short_sector": short_sector},
    )

    result = SectorRotationLongShort(long_count=1, short_count=1).build(bundle)

    weights = result.base_target_weights.loc[index[0]]
    assert weights["A"] == pytest.approx(1.0)
    assert weights["B"] == pytest.approx(0.0)
    assert weights.sum() == pytest.approx(1.0)
    assert result.meta["side_exposure"].loc[index[0], "long"] == pytest.approx(1.0)
    assert result.meta["side_exposure"].loc[index[0], "short"] == pytest.approx(0.0)
