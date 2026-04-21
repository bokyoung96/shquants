import pandas as pd
import pytest

from backtesting.strategy import CrossSectionalStrategy
from backtesting.strategy.cross import RankLongOnly, RankLongShort


def test_rank_long_only_selects_top_names() -> None:
    factor = pd.Series({"A": 1.0, "B": 3.0, "C": 2.0})
    strategy = RankLongOnly(top_n=2)

    weights = strategy.target_weights(factor)

    assert weights["B"] == 0.5
    assert weights["C"] == 0.5
    assert weights["A"] == 0.0
    assert weights.index.tolist() == factor.index.tolist()


def test_rank_long_short_balances_long_and_short_legs() -> None:
    factor = pd.Series({"A": 1.0, "B": 4.0, "C": 2.0, "D": 0.5})
    strategy = RankLongShort(top_n=2, bottom_n=1)

    weights = strategy.target_weights(factor)

    assert weights["B"] == 0.5
    assert weights["C"] == 0.5
    assert weights["D"] == -1.0
    assert weights["A"] == 0.0


def test_rank_long_short_avoids_overlap_in_small_universe() -> None:
    factor = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0})
    strategy = RankLongShort(top_n=2, bottom_n=2)

    weights = strategy.target_weights(factor)

    assert weights["A"] == 0.5
    assert weights["B"] == 0.5
    assert weights["C"] == -1.0
    assert weights.sum() == 0.0


def test_rank_long_only_ignores_nan_names_when_selection_exceeds_valid_count() -> None:
    factor = pd.Series({"A": 3.0, "B": 2.0, "C": float("nan")})
    strategy = RankLongOnly(top_n=2)

    weights = strategy.target_weights(factor)

    assert weights["A"] == 0.5
    assert weights["B"] == 0.5
    assert weights["C"] == 0.0


def test_rank_long_short_ignores_nan_names_when_selection_exceeds_valid_count() -> None:
    factor = pd.Series({"A": 4.0, "B": 2.0, "C": 1.0, "D": float("nan")})
    strategy = RankLongShort(top_n=3, bottom_n=2)

    weights = strategy.target_weights(factor)

    assert weights["A"] == 1 / 3
    assert weights["B"] == 1 / 3
    assert weights["C"] == 1 / 3
    assert weights["D"] == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_n": 0}, "top_n must be positive"),
        ({"top_n": -1}, "top_n must be positive"),
    ],
)
def test_rank_long_only_validates_top_n(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RankLongOnly(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_n": 0, "bottom_n": 1}, "top_n must be positive"),
        ({"top_n": 1, "bottom_n": 0}, "bottom_n must be positive"),
        ({"top_n": -1, "bottom_n": 1}, "top_n must be positive"),
        ({"top_n": 1, "bottom_n": -1}, "bottom_n must be positive"),
    ],
)
def test_rank_long_short_validates_leg_sizes(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RankLongShort(**kwargs)


def test_cross_strategies_expose_cross_sectional_extension_point() -> None:
    assert isinstance(RankLongOnly(top_n=1), CrossSectionalStrategy)
    assert isinstance(RankLongShort(top_n=1, bottom_n=1), CrossSectionalStrategy)
