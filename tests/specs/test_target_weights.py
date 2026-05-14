from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from backtesting.data import MarketData
from backtesting.specs.models import ExecutionSpec, ShortingSpec, TargetWeightsSpec
from backtesting.specs.target_weights import (
    build_position_plan_from_target_weights,
    read_target_weights_csv,
)


def _market(
    *,
    close: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    shortable: pd.DataFrame | None = None,
) -> MarketData:
    if close is None:
        close = pd.DataFrame(
            {"A": [10.0], "B": [10.0]},
            index=pd.to_datetime(["2024-01-02"]),
        )
    frames: dict[str, pd.DataFrame] = {"close": close}
    if shortable is not None:
        frames["shortable"] = shortable
    return MarketData(frames=frames, universe=universe, benchmark=None)


def _spec(path: Path, *, shorting: ShortingSpec | None = None) -> ExecutionSpec:
    return ExecutionSpec(
        start="2024-01-02",
        end="2024-01-02",
        target_weights=TargetWeightsSpec(kind="file", path=str(path)),
        shorting=shorting or ShortingSpec(enabled=True),
    )


def test_target_weights_file_preserves_signed_weights(tmp_path: Path) -> None:
    path = tmp_path / "weights.csv"
    path.write_text(",A,B\n2024-01-02,0.5,-0.5\n", encoding="utf-8")

    plan, meta = build_position_plan_from_target_weights(_spec(path), _market())

    expected = pd.DataFrame(
        {"A": [0.5], "B": [-0.5]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    assert_frame_equal(plan.target_weights, expected)
    assert meta["plan_source"] == "target_weights"
    assert meta["max_gross_exposure"] == pytest.approx(1.0)
    assert meta["min_net_exposure"] == pytest.approx(0.0)


def test_target_weights_csv_accepts_blank_cells_as_zero(tmp_path: Path) -> None:
    path = tmp_path / "weights.csv"
    path.write_text(",A,B\n2024-01-02,0.5,\n", encoding="utf-8")

    actual = read_target_weights_csv(path)

    expected = pd.DataFrame(
        {"A": [0.5], "B": [0.0]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (",A\n2024/01/02,0.5\n", "ISO format"),
        (",A\n2024-01-02,0.5\n2024-01-02,0.4\n", "unique dates"),
        (",A,A\n2024-01-02,0.5,0.4\n", "unique labels"),
        (",A\n2024-01-02,nope\n", "invalid values"),
        (",A\n2024-01-02,inf\n", "invalid values"),
    ],
)
def test_target_weights_csv_rejects_invalid_files(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "weights.csv"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_target_weights_csv(path)


def test_target_weights_rejects_negative_weights_without_shorting_enabled(tmp_path: Path) -> None:
    path = tmp_path / "weights.csv"
    path.write_text(",A,B\n2024-01-02,0.5,-0.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="shorting.enabled"):
        build_position_plan_from_target_weights(
            _spec(path, shorting=ShortingSpec(enabled=False)),
            _market(),
        )


def test_target_weights_rejects_nonzero_unknown_symbols(tmp_path: Path) -> None:
    path = tmp_path / "weights.csv"
    path.write_text(",A,B,C\n2024-01-02,0.5,-0.5,0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="absent from market data"):
        build_position_plan_from_target_weights(_spec(path), _market())


def test_target_weights_rejects_nonzero_untradable_targets(tmp_path: Path) -> None:
    path = tmp_path / "weights.csv"
    path.write_text(",A,B\n2024-01-02,0.5,-0.5\n", encoding="utf-8")
    universe = pd.DataFrame(
        {"A": [True], "B": [False]},
        index=pd.to_datetime(["2024-01-02"]),
    )

    with pytest.raises(ValueError, match="untradable target weights"):
        build_position_plan_from_target_weights(_spec(path), _market(universe=universe))


def test_target_weights_rejects_unshortable_short_targets(tmp_path: Path) -> None:
    path = tmp_path / "weights.csv"
    path.write_text(",A,B\n2024-01-02,0.5,-0.5\n", encoding="utf-8")
    shortable = pd.DataFrame(
        {"A": [True], "B": [False]},
        index=pd.to_datetime(["2024-01-02"]),
    )

    with pytest.raises(ValueError, match="unshortable short targets"):
        build_position_plan_from_target_weights(
            _spec(path, shorting=ShortingSpec(enabled=True, shortable_field="shortable")),
            _market(shortable=shortable),
        )
