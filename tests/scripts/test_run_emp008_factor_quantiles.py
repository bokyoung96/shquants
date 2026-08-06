from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from backtesting.data import MarketData
from backtesting.strategies.emp008.data import Emp008Config
from backtesting.strategies.emp008.factor_pipeline import PreparedEmp008Factors
from backtesting.strategies.emp008.factor_registry import FactorSetId, get_factor_set_definition
from backtesting.strategies.emp008.run_weights import DEFAULT_START

from backtesting.strategies.emp008 import run_factor_quantiles


def _frame(
    dates: pd.DatetimeIndex,
    columns: list[str],
    rows: list[list[object]],
) -> pd.DataFrame:
    return pd.DataFrame(rows, index=dates, columns=columns)


def make_prepared_bundle() -> PreparedEmp008Factors:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    columns = ["A", "B", "C", "D"]
    close = _frame(
        dates,
        columns,
        [
            [10.0, 20.0, 30.0, 40.0],
            [11.0, 19.0, 33.0, 38.0],
            [12.0, 18.0, 36.0, 37.0],
        ],
    )
    market_cap = _frame(
        dates,
        columns,
        [
            [100.0, 300.0, 200.0, 400.0],
            [110.0, 290.0, 210.0, 390.0],
            [120.0, 280.0, 220.0, 380.0],
        ],
    )
    universe = _frame(
        dates,
        columns,
        [
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
        ],
    ).astype(bool)
    factor_names = [factor_id.value for factor_id in get_factor_set_definition("mfbt").factors]
    alpha_factors = {
        factor_name: _frame(
            dates,
            columns,
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5],
                [0.3, 0.4, 0.5, 0.6],
            ],
        )
        for factor_name in factor_names
    }
    sectors = _frame(
        dates,
        columns,
        [
            ["Tech", "Finance", "Health", "Utilities"],
            ["Tech", "Finance", "Health", "Utilities"],
            ["Tech", "Finance", "Health", "Utilities"],
        ],
    )
    benchmark_weights = _frame(
        dates,
        columns,
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.10, 0.20, 0.30, 0.40],
            [0.10, 0.20, 0.30, 0.40],
        ],
    )
    market = MarketData(
        frames={
            "close": close,
            "market_cap": market_cap,
            "float_market_cap": market_cap,
            "k200_yn": universe,
            "sector_neutral_big": sectors,
            "bm_weights": benchmark_weights,
        },
        universe=None,
        benchmark=None,
    )
    return PreparedEmp008Factors(
        config=Emp008Config(),
        market=market,
        factor_set_definition=get_factor_set_definition("mfbt"),
        raw_factors=dict(alpha_factors),
        alpha_factors=alpha_factors,
        sector_factors={},
        close=close,
        market_cap=market_cap,
        float_market_cap=market_cap,
        universe=universe,
        sector=sectors,
        benchmark_weights=benchmark_weights,
        monthly_dates=tuple(dates),
    )


class _FakeQuantileResult:
    def __init__(self, payload: dict[str, object]) -> None:
        self.write_outputs = Mock(return_value=payload)


def test_parser_exposes_registry_factor_sets_and_quantile_count() -> None:
    parser = run_factor_quantiles._parser()

    factor_action = next(action for action in parser._actions if action.dest == "factor_set")
    assert tuple(factor_action.choices) == tuple(member.value for member in FactorSetId)

    args = parser.parse_args(["--factor-set", "all_factors", "--quantiles", "4"])

    assert args.factor_set == "all_factors"
    assert args.quantiles == 4


def test_main_prepares_once_and_writes_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prepared = make_prepared_bundle()
    result = _FakeQuantileResult({"summary_csv": str(tmp_path / "summary.csv")})
    prepare = Mock(return_value=prepared)
    evaluate = Mock(return_value=result)

    monkeypatch.setattr(run_factor_quantiles, "load_and_prepare_emp008_factors", prepare)
    monkeypatch.setattr(run_factor_quantiles, "run_emp008_factor_quantiles", evaluate)

    run_factor_quantiles.main(["--end", "2024-06-30", "--output-dir", str(tmp_path)])

    prepare.assert_called_once()
    evaluate.assert_called_once_with(prepared=prepared, start=DEFAULT_START, end="2024-06-30", q=5)
    result.write_outputs.assert_called_once_with(tmp_path, factor_set=prepared.config.factor_set, q=5)


def test_main_uses_latest_common_end_when_end_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = make_prepared_bundle()
    result = _FakeQuantileResult({"summary_csv": str(tmp_path / "summary.csv"), "메시지": "완료"})
    prepare = Mock(return_value=prepared)
    evaluate = Mock(return_value=result)
    latest_end = Mock(return_value="2024-05-31")

    monkeypatch.setattr(run_factor_quantiles, "load_and_prepare_emp008_factors", prepare)
    monkeypatch.setattr(run_factor_quantiles, "run_emp008_factor_quantiles", evaluate)
    monkeypatch.setattr(run_factor_quantiles, "latest_common_end", latest_end)

    run_factor_quantiles.main(["--output-dir", str(tmp_path)])

    latest_end.assert_called_once()
    evaluate.assert_called_once_with(prepared=prepared, start=DEFAULT_START, end="2024-05-31", q=5)
    assert "\"메시지\": \"완료\"" in capsys.readouterr().out


def test_run_cli_prints_expected_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(run_factor_quantiles, "main", Mock(side_effect=ValueError("q must be at least 2")))

    exit_code = run_factor_quantiles.run_cli(["--quantiles", "1"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.strip() == "Error: q must be at least 2"


def test_main_real_quantile_run_writes_outputs_and_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = make_prepared_bundle()
    prepare = Mock(return_value=prepared)

    monkeypatch.setattr(run_factor_quantiles, "load_and_prepare_emp008_factors", prepare)

    run_factor_quantiles.main(
        [
            "--end",
            "2024-03-29",
            "--output-dir",
            str(tmp_path),
            "--quantiles",
            "2",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert prepare.call_count == 1
    assert payload["monthly_returns_rows"] > 0
    assert payload["weights_rows"] > 0
    assert payload["rank_ic_rows"] > 0
    assert payload["daily_cumulative_returns_rows"] > payload["cumulative_returns_rows"]
    assert payload["summary_rows"] > 0
    for key in (
        "monthly_returns_csv",
        "monthly_returns_parquet",
        "portfolio_weights_parquet",
        "rank_ic_csv",
        "rank_ic_parquet",
        "cumulative_returns_csv",
        "daily_cumulative_returns_csv",
        "daily_cumulative_returns_parquet",
        "cumulative_quintiles_equal_weight_png",
        "cumulative_quintiles_market_cap_weight_png",
        "summary_csv",
        "summary_json",
        "manifest_json",
    ):
        assert Path(payload[key]).is_file()
    json.dumps(payload, ensure_ascii=False, indent=2)

    manifest = json.loads(Path(payload["manifest_json"]).read_text(encoding="utf-8"))
    assert manifest["weighting_modes"] == ["equal_weight", "market_cap_weight"]
    assert manifest["rebalance_frequency"] == "monthly"
    assert manifest["nav_frequency"] == "daily"
    assert manifest["selected_factors"] == [factor_id.value for factor_id in get_factor_set_definition("mfbt").factors]
