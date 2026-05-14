import pandas as pd
import pytest

from backtesting.construction.long_only import LongOnlyTopN
from backtesting.construction.long_short import LongShortTopBottom
from backtesting.construction.sector_neutral import SectorNeutralTopBottom
from backtesting.signals.base import SignalBundle
from backtesting.strategy.cross import RankLongOnly


def test_long_only_top_n_builds_weights_without_row_by_row_ranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    alpha = pd.DataFrame(
        {
            "A": [5.0, 1.0],
            "B": [4.0, float("nan")],
            "C": [1.0, 3.0],
        },
        index=index,
    )
    bundle = SignalBundle(alpha=alpha, context={"tradable": alpha.notna()})

    def fail_row_ranker(self: RankLongOnly, signal: pd.Series) -> pd.Series:
        raise AssertionError("LongOnlyTopN should vectorize across dates")

    monkeypatch.setattr(RankLongOnly, "target_weights", fail_row_ranker)

    result = LongOnlyTopN(top_n=2).build(bundle)

    expected = pd.DataFrame(
        {
            "A": [0.5, 0.5],
            "B": [0.5, 0.0],
            "C": [0.0, 0.5],
        },
        index=index,
    )
    pd.testing.assert_frame_equal(result.base_target_weights, expected)
    pd.testing.assert_frame_equal(result.selection_mask, expected.ne(0.0))


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


def test_long_short_top_bottom_builds_weights_without_row_by_row_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    alpha = pd.DataFrame(
        {
            "A": [5.0, 1.0],
            "B": [4.0, float("nan")],
            "C": [1.0, 3.0],
            "D": [0.0, 2.0],
        },
        index=index,
    )
    bundle = SignalBundle(alpha=alpha, context={"tradable": alpha.notna()})

    def fail_series_sort(self: pd.Series, *args, **kwargs) -> pd.Series:
        raise AssertionError("LongShortTopBottom should vectorize across dates")

    monkeypatch.setattr(pd.Series, "sort_values", fail_series_sort)

    result = LongShortTopBottom(top_n=2, bottom_n=1).build(bundle)

    expected = pd.DataFrame(
        {
            "A": [0.5, -1.0],
            "B": [0.5, 0.0],
            "C": [0.0, 0.5],
            "D": [-1.0, 0.5],
        },
        index=index,
    )
    pd.testing.assert_frame_equal(result.base_target_weights, expected)


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


def test_sector_neutral_top_bottom_supports_proportional_selected_group_budget() -> None:
    index = pd.to_datetime(["2024-01-02"])
    alpha = pd.DataFrame(
        {"A": [10.0], "B": [9.0], "C": [1.0], "D": [0.0], "E": [8.0], "F": [2.0]},
        index=index,
    )
    sector = pd.DataFrame(
        {
            "A": ["Tech"],
            "B": ["Tech"],
            "C": ["Tech"],
            "D": ["Tech"],
            "E": ["Energy"],
            "F": ["Energy"],
        },
        index=index,
    )
    bundle = SignalBundle(alpha=alpha, context={"tradable": alpha.notna(), "sector": sector})

    result = SectorNeutralTopBottom(
        top_n=2,
        bottom_n=2,
        group_budget="proportional_selected",
    ).build(bundle)

    weights = result.base_target_weights.loc[index[0]]
    assert weights["A"] == pytest.approx(1.0 / 3.0)
    assert weights["B"] == pytest.approx(1.0 / 3.0)
    assert weights["C"] == pytest.approx(-1.0 / 3.0)
    assert weights["D"] == pytest.approx(-1.0 / 3.0)
    assert weights["E"] == pytest.approx(1.0 / 3.0)
    assert weights["F"] == pytest.approx(-1.0 / 3.0)
    assert result.group_long_budget.loc[index[0], "Tech"] == pytest.approx(2.0 / 3.0)
    assert result.group_long_budget.loc[index[0], "Energy"] == pytest.approx(1.0 / 3.0)
    assert result.group_short_budget.loc[index[0], "Tech"] == pytest.approx(2.0 / 3.0)
    assert result.group_short_budget.loc[index[0], "Energy"] == pytest.approx(1.0 / 3.0)


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
