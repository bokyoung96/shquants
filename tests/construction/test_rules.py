import pandas as pd
import pytest

from backtesting.construction.long_short import LongShortTopBottom
from backtesting.construction.sector_neutral import SectorNeutralTopBottom
from backtesting.signals.base import SignalBundle


def test_long_short_top_bottom_is_dollar_neutral() -> None:
    index = pd.to_datetime(["2024-01-02"])
    alpha = pd.DataFrame(
        {"A": [5.0], "B": [4.0], "C": [1.0], "D": [0.0]},
        index=index,
    )
    bundle = SignalBundle(alpha=alpha, context={"tradable": alpha.notna()})

    result = LongShortTopBottom(top_n=2, bottom_n=1).build(bundle)

    weights = result.base_target_weights.loc[index[0]]
    assert weights["A"] == 0.5
    assert weights["B"] == 0.5
    assert weights["D"] == -1.0
    assert round(float(weights.sum()), 8) == 0.0
    assert bool(result.selection_mask.loc[index[0], "C"]) is False
    assert bool(result.meta["selected_long"].loc[index[0], "A"])
    assert bool(result.meta["selected_long"].loc[index[0], "B"])
    assert bool(result.meta["selected_short"].loc[index[0], "D"])


def test_sector_neutral_top_bottom_balances_by_sector() -> None:
    index = pd.to_datetime(["2024-01-02"])
    alpha = pd.DataFrame(
        {"A": [9.0], "B": [1.0], "C": [8.0], "D": [0.0]},
        index=index,
    )
    sector = pd.DataFrame(
        {"A": ["Tech"], "B": ["Tech"], "C": ["Energy"], "D": ["Energy"]},
        index=index,
    )
    bundle = SignalBundle(
        alpha=alpha,
        context={"tradable": alpha.notna(), "sector": sector},
    )

    result = SectorNeutralTopBottom(top_n=1, bottom_n=1).build(bundle)

    weights = result.base_target_weights.loc[index[0]]
    assert weights["A"] == 0.5
    assert weights["B"] == -0.5
    assert weights["C"] == 0.5
    assert weights["D"] == -0.5
    assert round(float(weights.sum()), 8) == 0.0
    assert result.group_long_budget.loc[index[0], "Tech"] == 0.5
    assert result.group_long_budget.loc[index[0], "Energy"] == 0.5
    assert result.group_short_budget.loc[index[0], "Tech"] == 0.5
    assert result.group_short_budget.loc[index[0], "Energy"] == 0.5
    assert bool(result.meta["selected_long"].loc[index[0], "A"])
    assert bool(result.meta["selected_short"].loc[index[0], "B"])
    assert bool(result.meta["selected_long"].loc[index[0], "C"])
    assert bool(result.meta["selected_short"].loc[index[0], "D"])
    assert result.meta["group_id"].loc[index[0], "A"] == "Tech"
    assert result.meta["group_long_budget"].loc[index[0], "Tech"] == 0.5
    assert result.meta["group_short_budget"].loc[index[0], "Energy"] == 0.5


def test_long_short_top_bottom_shrinks_long_leg_to_preserve_small_universe_neutrality() -> None:
    index = pd.to_datetime(["2024-01-02"])
    alpha = pd.DataFrame(
        {"A": [5.0], "B": [1.0]},
        index=index,
    )
    bundle = SignalBundle(alpha=alpha, context={"tradable": alpha.notna()})

    result = LongShortTopBottom(top_n=2, bottom_n=1).build(bundle)

    weights = result.base_target_weights.loc[index[0]]
    assert weights["A"] == 1.0
    assert weights["B"] == -1.0
    assert round(float(weights.sum()), 8) == 0.0
    assert bool(result.meta["selected_long"].loc[index[0], "A"])
    assert bool(result.meta["selected_short"].loc[index[0], "B"])


def test_sector_neutral_top_bottom_skips_undersized_sector() -> None:
    index = pd.to_datetime(["2024-01-02"])
    alpha = pd.DataFrame(
        {"A": [9.0], "B": [1.0], "C": [7.0]},
        index=index,
    )
    sector = pd.DataFrame(
        {"A": ["Tech"], "B": ["Tech"], "C": ["Utilities"]},
        index=index,
    )
    bundle = SignalBundle(
        alpha=alpha,
        context={"tradable": alpha.notna(), "sector": sector},
    )

    result = SectorNeutralTopBottom(top_n=1, bottom_n=1).build(bundle)

    weights = result.base_target_weights.loc[index[0]]
    assert weights["A"] == 1.0
    assert weights["B"] == -1.0
    assert weights["C"] == 0.0
    assert round(float(weights.sum()), 8) == 0.0
    assert result.group_long_budget.loc[index[0], "Tech"] == 1.0
    assert result.group_short_budget.loc[index[0], "Tech"] == 1.0
    assert "Utilities" not in result.group_long_budget.columns
    assert "Utilities" not in result.group_short_budget.columns
    assert bool(result.meta["selected_long"].loc[index[0], "A"])
    assert bool(result.meta["selected_short"].loc[index[0], "B"])
    assert bool(result.meta["selected_long"].loc[index[0], "C"]) is False
    assert result.meta["group_id"].loc[index[0], "A"] == "Tech"
    assert result.meta["group_id"].loc[index[0], "C"] == "Utilities"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_n": 0, "bottom_n": 1}, "top_n must be positive"),
        ({"top_n": 1, "bottom_n": 0}, "bottom_n must be positive"),
        ({"top_n": -1, "bottom_n": 1}, "top_n must be positive"),
        ({"top_n": 1, "bottom_n": -1}, "bottom_n must be positive"),
    ],
)
def test_long_short_top_bottom_validates_leg_sizes(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LongShortTopBottom(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_n": 0, "bottom_n": 1}, "top_n must be positive"),
        ({"top_n": 1, "bottom_n": 0}, "bottom_n must be positive"),
        ({"top_n": -1, "bottom_n": 1}, "top_n must be positive"),
        ({"top_n": 1, "bottom_n": -1}, "bottom_n must be positive"),
    ],
)
def test_sector_neutral_top_bottom_validates_leg_sizes(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SectorNeutralTopBottom(**kwargs)
