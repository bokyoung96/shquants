import uuid

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from backtesting.specs.models import WeightingSpec
from backtesting.weighting import (
    build_weights,
    register_weighting_hook,
    unregister_weighting_hook,
    weighting_fields,
)


@pytest.fixture
def selection() -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=25, freq="D")
    selected = pd.DataFrame(False, index=index, columns=["A", "B", "C"], dtype=bool)
    selected.loc[index[0], ["A", "B"]] = True
    selected.loc[index[1], ["A", "C"]] = True
    selected.loc[index[20], ["A", "B", "C"]] = True
    selected.loc[index[21], ["B", "C"]] = True
    return selected


@pytest.fixture
def feature_frames(selection: pd.DataFrame) -> dict[str, pd.DataFrame]:
    index = selection.index
    return {
        "market_cap": pd.DataFrame(
            {
                "A": [100.0] * len(index),
                "B": [300.0] * len(index),
                "C": [600.0] * len(index),
            },
            index=index,
        ),
        "float_market_cap": pd.DataFrame(
            {
                "A": [50.0] * len(index),
                "B": [150.0] * len(index),
                "C": [300.0] * len(index),
            },
            index=index,
        ),
        "alpha_score": pd.DataFrame(
            {
                "A": [2.0, -1.0] + [0.5] * (len(index) - 2),
                "B": [1.0, 4.0] + [1.5] * (len(index) - 2),
                "C": [-3.0, 3.0] + [2.5] * (len(index) - 2),
            },
            index=index,
        ),
        "close": pd.DataFrame(
            {
                "A": [100.0 + i for i in range(len(index))],
                "B": [100.0 + 2 * i for i in range(len(index))],
                "C": [100.0 + (0.5 * i) for i in range(len(index))],
            },
            index=index,
        ),
    }


def test_equal_weight_normalizes_only_selected_names(selection: pd.DataFrame) -> None:
    actual = build_weights(WeightingSpec(kind="equal_weight"), selection, {})

    assert actual.dtypes.tolist() == [float, float, float]
    assert actual.loc[selection.index[0]].tolist() == [0.5, 0.5, 0.0]
    assert actual.loc[selection.index[1]].tolist() == [0.5, 0.0, 0.5]
    assert actual.loc[selection.index[2]].tolist() == [0.0, 0.0, 0.0]
    assert actual.loc[selection.index[20]].sum() == pytest.approx(1.0)


def test_market_cap_normalizes_within_selected_names(
    selection: pd.DataFrame, feature_frames: dict[str, pd.DataFrame]
) -> None:
    actual = build_weights(WeightingSpec(kind="market_cap"), selection, feature_frames)

    expected = pd.DataFrame(0.0, index=selection.index, columns=selection.columns)
    expected.loc[selection.index[0], ["A", "B"]] = [0.25, 0.75]
    expected.loc[selection.index[1], ["A", "C"]] = [100.0 / 700.0, 600.0 / 700.0]
    expected.loc[selection.index[20], ["A", "B", "C"]] = [0.1, 0.3, 0.6]
    expected.loc[selection.index[21], ["B", "C"]] = [1.0 / 3.0, 2.0 / 3.0]
    assert_frame_equal(actual, expected.astype(float))


def test_float_market_cap_uses_float_market_cap_feature(
    selection: pd.DataFrame, feature_frames: dict[str, pd.DataFrame]
) -> None:
    actual = build_weights(WeightingSpec(kind="float_market_cap"), selection, feature_frames)

    expected = pd.DataFrame(0.0, index=selection.index, columns=selection.columns)
    expected.loc[selection.index[0], ["A", "B"]] = [0.25, 0.75]
    expected.loc[selection.index[1], ["A", "C"]] = [50.0 / 350.0, 300.0 / 350.0]
    expected.loc[selection.index[20], ["A", "B", "C"]] = [0.1, 0.3, 0.6]
    expected.loc[selection.index[21], ["B", "C"]] = [1.0 / 3.0, 2.0 / 3.0]
    assert_frame_equal(actual, expected.astype(float))


def test_score_uses_only_positive_values_from_configured_field(
    selection: pd.DataFrame, feature_frames: dict[str, pd.DataFrame]
) -> None:
    actual = build_weights(WeightingSpec(kind="score", field="alpha_score"), selection, feature_frames)

    expected = pd.DataFrame(0.0, index=selection.index, columns=selection.columns)
    expected.loc[selection.index[0], ["A", "B"]] = [2.0 / 3.0, 1.0 / 3.0]
    expected.loc[selection.index[1], "C"] = 1.0
    expected.loc[selection.index[20], ["A", "B", "C"]] = [0.5 / 4.5, 1.5 / 4.5, 2.5 / 4.5]
    expected.loc[selection.index[21], ["B", "C"]] = [1.5 / 4.0, 2.5 / 4.0]
    assert_frame_equal(actual, expected.astype(float))


def test_inverse_vol_returns_finite_selected_weights(
    selection: pd.DataFrame, feature_frames: dict[str, pd.DataFrame]
) -> None:
    actual = build_weights(WeightingSpec(kind="inverse_vol"), selection, feature_frames)

    row = actual.loc[selection.index[21]]
    assert row[["B", "C"]].gt(0.0).all()
    assert row[["B", "C"]].map(pd.notna).all()
    assert row[["B", "C"]].map(lambda value: value != float("inf")).all()
    assert row.sum() == pytest.approx(1.0)
    assert actual.loc[selection.index[0]].sum() == pytest.approx(0.0)


def test_explicit_reads_csv_and_aligns_to_selection(selection: pd.DataFrame, tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "weights.csv"
    path.write_text(
        ",B,C,D\n2024-01-02,0.25,0.75,1.00\n2024-01-03,,2.0,\n2024-01-22,3.0,,\n",
        encoding="utf-8",
    )

    actual = build_weights(WeightingSpec(kind="explicit", path=str(path)), selection, {})

    expected = pd.DataFrame(0.0, index=selection.index, columns=selection.columns)
    expected.loc[pd.Timestamp("2024-01-02"), "B"] = 1.0
    expected.loc[pd.Timestamp("2024-01-03"), "C"] = 1.0
    expected.loc[pd.Timestamp("2024-01-22"), ["A", "B", "C"]] = [0.0, 1.0, 0.0]
    assert_frame_equal(actual, expected.astype(float))


def test_hook_delegates_to_registered_weighting_hook(
    selection: pd.DataFrame, feature_frames: dict[str, pd.DataFrame]
) -> None:
    hook_id = f"weight_hook_{uuid.uuid4().hex}"
    register_weighting_hook(hook_id, lambda spec, selected, frames: frames["market_cap"] * selected.astype(float))
    try:
        actual = build_weights(WeightingSpec(kind="hook", hook_id=hook_id), selection, feature_frames)
    finally:
        unregister_weighting_hook(hook_id)

    expected = build_weights(WeightingSpec(kind="market_cap"), selection, feature_frames)
    assert_frame_equal(actual, expected)


def test_weighting_fields_reports_required_inputs() -> None:
    assert weighting_fields(WeightingSpec(kind="equal_weight")) == ()
    assert weighting_fields(WeightingSpec(kind="market_cap")) == ("market_cap",)
    assert weighting_fields(WeightingSpec(kind="float_market_cap")) == ("float_market_cap",)
    assert weighting_fields(WeightingSpec(kind="score", field="alpha_score")) == ("alpha_score",)
    assert weighting_fields(WeightingSpec(kind="inverse_vol")) == ("close",)
    assert weighting_fields(WeightingSpec(kind="explicit")) == ()
    assert weighting_fields(WeightingSpec(kind="hook", hook_id="demo", params={"fields": ["x", "y"]})) == ("x", "y")


@pytest.mark.parametrize(
    ("spec", "pattern"),
    [
        (WeightingSpec(kind="mystery"), "unknown weighting kind"),
        (WeightingSpec(kind="score"), "requires field"),
        (WeightingSpec(kind="explicit"), "requires path"),
        (WeightingSpec(kind="hook"), "requires hook_id"),
    ],
)
def test_invalid_specs_raise_clear_errors(
    spec: WeightingSpec,
    pattern: str,
    selection: pd.DataFrame,
    feature_frames: dict[str, pd.DataFrame],
) -> None:
    with pytest.raises((ValueError, KeyError), match=pattern):
        build_weights(spec, selection, feature_frames)


def test_missing_feature_field_raises_clear_error(selection: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="unknown weighting field: market_cap"):
        build_weights(WeightingSpec(kind="market_cap"), selection, {})
