import uuid

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from backtesting.specs.models import ConditionSpec, SelectionSpec
from backtesting.selection import (
    build_selection,
    register_selection_hook,
    selection_fields,
    unregister_selection_hook,
)


@pytest.fixture
def feature_frames() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2024-01-02", periods=3, freq="D")
    return {
        "score": pd.DataFrame(
            {
                "A": [10.0, 30.0, 20.0],
                "B": [20.0, 10.0, 30.0],
                "C": [30.0, 20.0, 10.0],
            },
            index=index,
        ),
        "quality": pd.DataFrame(
            {
                "A": [5.0, 1.0, 3.0],
                "B": [2.0, 4.0, 2.0],
                "C": [4.0, 3.0, 1.0],
            },
            index=index,
        ),
        "event_flag": pd.DataFrame(
            {
                "A": [1.0, 0.0, 0.0],
                "B": [0.0, 1.0, 0.0],
                "C": [0.0, 0.0, 1.0],
            },
            index=index,
        ),
        "nullable": pd.DataFrame(
            {
                "A": [1.0, None, 3.0],
                "B": [None, 2.0, None],
                "C": [0.0, None, 5.0],
            },
            index=index,
        ),
    }


def test_filter_keeps_every_passing_name_without_top_n_cap(feature_frames: dict[str, pd.DataFrame]) -> None:
    spec = SelectionSpec(
        kind="filter",
        conditions=(
            ConditionSpec(field="score", op=">=", value=20.0),
            ConditionSpec(field="quality", op=">=", value=2.0),
        ),
        n=1,
    )

    actual = build_selection(spec, feature_frames)

    expected = pd.DataFrame(
        {
            "A": [False, False, True],
            "B": [True, False, True],
            "C": [True, True, False],
        },
        index=feature_frames["score"].index,
        dtype=bool,
    )
    assert_frame_equal(actual, expected)


def test_rank_top_n_selects_exact_count_per_date(feature_frames: dict[str, pd.DataFrame]) -> None:
    spec = SelectionSpec(kind="rank_top_n", field="score", n=2)

    actual = build_selection(spec, feature_frames)

    assert actual.sum(axis=1).tolist() == [2, 2, 2]
    expected = pd.DataFrame(
        {
            "A": [False, True, True],
            "B": [True, False, True],
            "C": [True, True, False],
        },
        index=feature_frames["score"].index,
        dtype=bool,
    )
    assert_frame_equal(actual, expected)


def test_rank_top_bottom_selects_long_and_short_candidates(feature_frames: dict[str, pd.DataFrame]) -> None:
    spec = SelectionSpec(kind="rank_top_bottom", field="score", top_n=1, bottom_n=1)

    actual = build_selection(spec, feature_frames)

    expected = pd.DataFrame(
        {
            "A": [True, True, False],
            "B": [False, True, True],
            "C": [True, False, True],
        },
        index=feature_frames["score"].index,
        dtype=bool,
    )
    assert_frame_equal(actual, expected)


def test_rank_top_n_rejects_non_positive_n(feature_frames: dict[str, pd.DataFrame]) -> None:
    spec = SelectionSpec(kind="rank_top_n", field="score", n=0)

    with pytest.raises(ValueError, match="requires n > 0"):
        build_selection(spec, feature_frames)


@pytest.mark.parametrize("invalid_n", ["x", "1", 1.0, True])
def test_rank_top_n_rejects_invalid_typed_n(feature_frames: dict[str, pd.DataFrame], invalid_n: object) -> None:
    spec = SelectionSpec(kind="rank_top_n", field="score", n=invalid_n)

    with pytest.raises(ValueError, match="requires integer n"):
        build_selection(spec, feature_frames)


def test_score_threshold_selects_scores_meeting_threshold(feature_frames: dict[str, pd.DataFrame]) -> None:
    spec = SelectionSpec(kind="score_threshold", field="score", threshold=20.0)

    actual = build_selection(spec, feature_frames)

    expected = feature_frames["score"].ge(20.0).astype(bool)
    assert_frame_equal(actual, expected)


def test_score_threshold_rejects_invalid_typed_threshold(feature_frames: dict[str, pd.DataFrame]) -> None:
    spec = SelectionSpec(kind="score_threshold", field="score", threshold="x")

    with pytest.raises(ValueError, match="requires numeric threshold"):
        build_selection(spec, feature_frames)


def test_event_extends_flags_by_hold_days(feature_frames: dict[str, pd.DataFrame]) -> None:
    spec = SelectionSpec(kind="event", field="event_flag", hold_days=1)

    actual = build_selection(spec, feature_frames)

    expected = pd.DataFrame(
        {
            "A": [True, True, False],
            "B": [False, True, True],
            "C": [False, False, True],
        },
        index=feature_frames["event_flag"].index,
        dtype=bool,
    )
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("invalid_hold_days", ["x", "1", 1.0, True])
def test_event_rejects_invalid_typed_hold_days(feature_frames: dict[str, pd.DataFrame], invalid_hold_days: object) -> None:
    spec = SelectionSpec(kind="event", field="event_flag", hold_days=invalid_hold_days)

    with pytest.raises(ValueError, match="requires integer hold_days"):
        build_selection(spec, feature_frames)


def test_explicit_aligns_csv_to_feature_index_and_columns(
    feature_frames: dict[str, pd.DataFrame], tmp_path: pytest.TempPathFactory
) -> None:
    explicit = pd.DataFrame(
        {
            "B": [1, 0],
            "C": [0, 1],
            "D": [1, 1],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    path = tmp_path / "explicit.csv"
    explicit.to_csv(path)
    spec = SelectionSpec(kind="explicit", path=str(path))

    actual = build_selection(spec, feature_frames)

    expected = pd.DataFrame(
        {
            "A": [False, False, False],
            "B": [True, False, False],
            "C": [False, True, False],
        },
        index=feature_frames["score"].index,
        dtype=bool,
    )
    assert_frame_equal(actual, expected)


def test_explicit_rejects_malformed_dates(feature_frames: dict[str, pd.DataFrame], tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "explicit_bad_dates.csv"
    path.write_text(""",A,B,C
not-a-date,1,0,0
2024-01-03,0,1,0
""")
    spec = SelectionSpec(kind="explicit", path=str(path))

    with pytest.raises(ValueError, match="valid unique dates"):
        build_selection(spec, feature_frames)


def test_explicit_rejects_non_iso_dates(feature_frames: dict[str, pd.DataFrame], tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "explicit_non_iso_dates.csv"
    path.write_text(""",A,B,C
01/02/2024,1,0,0
2024-01-03,0,1,0
""")
    spec = SelectionSpec(kind="explicit", path=str(path))

    with pytest.raises(ValueError, match="ISO format"):
        build_selection(spec, feature_frames)


def test_explicit_rejects_duplicate_dates(feature_frames: dict[str, pd.DataFrame], tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "explicit_duplicate_dates.csv"
    path.write_text(""",A,B,C
2024-01-02,1,0,0
2024-01-02,0,1,0
""")
    spec = SelectionSpec(kind="explicit", path=str(path))

    with pytest.raises(ValueError, match="unique dates"):
        build_selection(spec, feature_frames)


def test_explicit_rejects_blank_dates(feature_frames: dict[str, pd.DataFrame], tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "explicit_blank_dates.csv"
    path.write_text(""",A,B,C
,1,0,0
2024-01-03,0,1,0
""")
    spec = SelectionSpec(kind="explicit", path=str(path))

    with pytest.raises(ValueError, match="valid unique dates"):
        build_selection(spec, feature_frames)


def test_explicit_rejects_duplicate_columns(feature_frames: dict[str, pd.DataFrame], tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "explicit_duplicate_columns.csv"
    path.write_text(""",A,A,C
2024-01-02,1,0,0
2024-01-03,0,1,0
""")
    spec = SelectionSpec(kind="explicit", path=str(path))

    with pytest.raises(ValueError, match="unique labels"):
        build_selection(spec, feature_frames)


def test_explicit_rejects_blank_columns(feature_frames: dict[str, pd.DataFrame], tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "explicit_blank_columns.csv"
    path.write_text(""",A,,C
2024-01-02,1,0,0
2024-01-03,0,1,0
""")
    spec = SelectionSpec(kind="explicit", path=str(path))

    with pytest.raises(ValueError, match="blank labels"):
        build_selection(spec, feature_frames)


def test_explicit_rejects_invalid_cell_contents(feature_frames: dict[str, pd.DataFrame], tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "explicit_invalid_values.csv"
    path.write_text(""",A,B,C
2024-01-02,1,yes,
2024-01-03,0,1,no
""")
    spec = SelectionSpec(kind="explicit", path=str(path))

    with pytest.raises(ValueError, match=r"2024-01-02.*/B"):
        build_selection(spec, feature_frames)


@pytest.mark.parametrize("invalid_value", ["2", "-1"])
def test_explicit_rejects_non_binary_numeric_values(
    feature_frames: dict[str, pd.DataFrame], tmp_path: pytest.TempPathFactory, invalid_value: str
) -> None:
    path = tmp_path / "explicit_invalid_numeric_values.csv"
    path.write_text(
        f""",A,B,C
2024-01-02,1,{invalid_value},
2024-01-03,0,1,0
""",
        encoding="utf-8",
    )
    spec = SelectionSpec(kind="explicit", path=str(path))

    with pytest.raises(ValueError, match=r"2024-01-02.*/B"):
        build_selection(spec, feature_frames)


def test_explicit_accepts_blank_cells_as_false(feature_frames: dict[str, pd.DataFrame], tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "explicit_blank_values.csv"
    path.write_text(""",A,B,C
2024-01-02,1,,0
2024-01-03,,1,
""")
    spec = SelectionSpec(kind="explicit", path=str(path))

    actual = build_selection(spec, feature_frames)

    expected = pd.DataFrame(
        {
            "A": [True, False, False],
            "B": [False, True, False],
            "C": [False, False, False],
        },
        index=feature_frames["score"].index,
        dtype=bool,
    )
    assert_frame_equal(actual, expected)


def test_hook_delegates_to_registered_selection_hook(feature_frames: dict[str, pd.DataFrame]) -> None:
    hook_id = f"test_hook_{uuid.uuid4().hex}"
    register_selection_hook(hook_id, lambda spec, frames: frames["quality"].ge(spec.params["minimum"]))
    try:
        spec = SelectionSpec(kind="hook", hook_id=hook_id, params={"minimum": 3.0})

        actual = build_selection(spec, feature_frames)

        expected = feature_frames["quality"].ge(3.0).astype(bool)
        assert_frame_equal(actual, expected)
    finally:
        unregister_selection_hook(hook_id)


def test_unregister_selection_hook_removes_registration(feature_frames: dict[str, pd.DataFrame]) -> None:
    hook_id = f"test_hook_{uuid.uuid4().hex}"
    register_selection_hook(hook_id, lambda spec, frames: frames["score"].ge(20.0))
    unregister_selection_hook(hook_id)

    with pytest.raises(KeyError, match="unknown selection hook_id"):
        build_selection(SelectionSpec(kind="hook", hook_id=hook_id), feature_frames)


def test_hook_requires_hook_id(feature_frames: dict[str, pd.DataFrame]) -> None:
    with pytest.raises(ValueError, match="requires hook_id"):
        build_selection(SelectionSpec(kind="hook"), feature_frames)


def test_invalid_condition_operator_raises_value_error(feature_frames: dict[str, pd.DataFrame]) -> None:
    spec = SelectionSpec(kind="filter", conditions=(ConditionSpec(field="score", op="contains", value=1),))

    with pytest.raises(ValueError, match="unsupported condition operator"):
        build_selection(spec, feature_frames)


def test_selection_fields_include_condition_and_primary_fields() -> None:
    spec = SelectionSpec(
        kind="rank_top_n",
        field="score",
        conditions=(ConditionSpec(field="quality", op=">=", value=2.0),),
    )

    assert selection_fields(spec) == ("score", "quality")

@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (
            ConditionSpec(field="score", op=">", value=20.0),
            pd.DataFrame(
                {
                    "A": [False, True, False],
                    "B": [False, False, True],
                    "C": [True, False, False],
                },
                index=pd.date_range("2024-01-02", periods=3, freq="D"),
                dtype=bool,
            ),
        ),
        (
            ConditionSpec(field="score", op="<", value=20.0),
            pd.DataFrame(
                {
                    "A": [True, False, False],
                    "B": [False, True, False],
                    "C": [False, False, True],
                },
                index=pd.date_range("2024-01-02", periods=3, freq="D"),
                dtype=bool,
            ),
        ),
        (
            ConditionSpec(field="score", op="<=", value=20.0),
            pd.DataFrame(
                {
                    "A": [True, False, True],
                    "B": [True, True, False],
                    "C": [False, True, True],
                },
                index=pd.date_range("2024-01-02", periods=3, freq="D"),
                dtype=bool,
            ),
        ),
        (
            ConditionSpec(field="score", op="==", value=20.0),
            pd.DataFrame(
                {
                    "A": [False, False, True],
                    "B": [True, False, False],
                    "C": [False, True, False],
                },
                index=pd.date_range("2024-01-02", periods=3, freq="D"),
                dtype=bool,
            ),
        ),
        (
            ConditionSpec(field="score", op="!=", value=20.0),
            pd.DataFrame(
                {
                    "A": [True, True, False],
                    "B": [False, True, True],
                    "C": [True, False, True],
                },
                index=pd.date_range("2024-01-02", periods=3, freq="D"),
                dtype=bool,
            ),
        ),
        (
            ConditionSpec(field="nullable", op="notna"),
            pd.DataFrame(
                {
                    "A": [True, False, True],
                    "B": [False, True, False],
                    "C": [True, False, True],
                },
                index=pd.date_range("2024-01-02", periods=3, freq="D"),
                dtype=bool,
            ),
        ),
        (
            ConditionSpec(field="nullable", op="isna"),
            pd.DataFrame(
                {
                    "A": [False, True, False],
                    "B": [True, False, True],
                    "C": [False, True, False],
                },
                index=pd.date_range("2024-01-02", periods=3, freq="D"),
                dtype=bool,
            ),
        ),
    ],
)
def test_filter_supports_condition_operators(
    feature_frames: dict[str, pd.DataFrame], condition: ConditionSpec, expected: pd.DataFrame
) -> None:
    spec = SelectionSpec(kind="filter", conditions=(condition,))

    actual = build_selection(spec, feature_frames)

    assert_frame_equal(actual, expected)
