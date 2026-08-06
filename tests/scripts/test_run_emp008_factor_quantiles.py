from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from backtesting.data import MarketData
from backtesting.strategies.emp008.mfbt_emp008_data import MfbtEmp008Config
from backtesting.strategies.emp008.mfbt_emp008_factor_pipeline import PreparedEmp008Factors
from backtesting.strategies.emp008.mfbt_emp008_factor_quantiles import Emp008FactorQuantileResult
from backtesting.strategies.emp008.mfbt_emp008_factor_registry import FactorSetId, get_factor_set_definition
from backtesting.strategies.emp008.run_weights import DEFAULT_START

from backtesting.strategies.emp008 import run_factor_quantiles


def _frame(
    dates: pd.DatetimeIndex,
    columns: list[str],
    rows: list[list[object]],
) -> pd.DataFrame:
    return pd.DataFrame(rows, index=dates, columns=columns)


def make_prepared_bundle() -> PreparedEmp008Factors:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    columns = ["A", "B"]
    close = _frame(dates, columns, [[10.0, 20.0], [11.0, 19.0]])
    market_cap = _frame(dates, columns, [[100.0, 300.0], [110.0, 290.0]])
    universe = _frame(dates, columns, [[True, True], [True, True]]).astype(bool)
    factor_names = [factor_id.value for factor_id in get_factor_set_definition("mfbt").factors]
    alpha_factors = {
        factor_name: _frame(dates, columns, [[0.2, -0.2], [0.3, -0.3]])
        for factor_name in factor_names
    }
    market = MarketData(
        frames={
            "close": close,
            "market_cap": market_cap,
            "float_market_cap": market_cap,
            "k200_yn": universe,
            "sector_neutral_big": _frame(dates, columns, [["Tech", "Tech"], ["Tech", "Tech"]]),
            "bm_weights": _frame(dates, columns, [[0.25, 0.75], [0.25, 0.75]]),
        },
        universe=None,
        benchmark=None,
    )
    return PreparedEmp008Factors(
        config=MfbtEmp008Config(),
        market=market,
        factor_set_definition=get_factor_set_definition("mfbt"),
        raw_factors=dict(alpha_factors),
        alpha_factors=alpha_factors,
        sector_factors={},
        close=close,
        market_cap=market_cap,
        float_market_cap=market_cap,
        universe=universe,
        sector=_frame(dates, columns, [["Tech", "Tech"], ["Tech", "Tech"]]),
        benchmark_weights=_frame(dates, columns, [[0.25, 0.75], [0.25, 0.75]]),
        monthly_dates=tuple(dates),
    )


def make_quantile_result() -> Emp008FactorQuantileResult:
    monthly_returns = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-31", "2024-01-31"]),
            "return_date": pd.to_datetime(["2024-02-29", "2024-02-29"]),
            "factor": ["price_momentum", "price_momentum"],
            "weighting": ["equal_weight", "equal_weight"],
            "portfolio": ["Q1", "Q5"],
            "return": [0.01, 0.03],
            "constituent_count": [1, 1],
        }
    )
    portfolio_weights = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-31", "2024-01-31"]),
            "return_date": pd.to_datetime(["2024-02-29", "2024-02-29"]),
            "factor": ["price_momentum", "price_momentum"],
            "weighting": ["equal_weight", "market_cap_weight"],
            "quantile": ["Q1", "Q1"],
            "ticker": ["A", "A"],
            "weight": [1.0, 1.0],
        }
    )
    rank_ic = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-31"]),
            "return_date": pd.to_datetime(["2024-02-29"]),
            "factor": ["price_momentum"],
            "rank_ic": [1.0],
            "directional_rank_ic": [1.0],
            "n_obs": [2],
        }
    )
    cumulative_returns = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-31", "2024-01-31"]),
            "return_date": pd.to_datetime(["2024-02-29", "2024-02-29"]),
            "factor": ["price_momentum", "price_momentum"],
            "weighting": ["equal_weight", "equal_weight"],
            "portfolio": ["Q1", "Q5"],
            "cumulative_return": [0.01, 0.03],
        }
    )
    summary = pd.DataFrame(
        {
            "factor": ["price_momentum", "price_momentum"],
            "weighting": ["equal_weight", "equal_weight"],
            "portfolio": ["Q1", "Q5"],
            "observations": [1, 1],
            "annualized_return": [0.12, 0.36],
            "annualized_volatility": [0.0, 0.0],
            "sharpe": [0.0, 0.0],
            "max_drawdown": [0.0, 0.0],
            "positive_month_rate": [1.0, 1.0],
            "mean_monthly_return": [0.01, 0.03],
            "average_constituent_count": [1.0, 1.0],
            "average_one_way_turnover": [float("nan"), float("nan")],
            "mean_rank_ic": [1.0, 1.0],
            "directional_mean_rank_ic": [1.0, 1.0],
            "ic_information_ratio": [0.0, 0.0],
            "ic_positive_rate": [1.0, 1.0],
            "quantile_monotonicity": [1.0, 1.0],
        }
    )
    return Emp008FactorQuantileResult(
        monthly_returns=monthly_returns,
        portfolio_weights=portfolio_weights,
        rank_ic=rank_ic,
        cumulative_returns=cumulative_returns,
        summary=summary,
    )


def test_parser_exposes_registry_factor_sets_and_quantile_count() -> None:
    parser = run_factor_quantiles._parser()

    factor_action = next(action for action in parser._actions if action.dest == "factor_set")
    assert tuple(factor_action.choices) == tuple(member.value for member in FactorSetId)

    args = parser.parse_args(["--factor-set", "origin", "--quantiles", "4"])

    assert args.factor_set == "origin"
    assert args.quantiles == 4


def test_main_prepares_once_and_writes_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prepared = make_prepared_bundle()
    result = make_quantile_result()
    prepare = Mock(return_value=prepared)
    evaluate = Mock(return_value=result)
    write = Mock(return_value={"summary_csv": str(tmp_path / "summary.csv")})

    monkeypatch.setattr(run_factor_quantiles, "load_and_prepare_emp008_factors", prepare)
    monkeypatch.setattr(run_factor_quantiles, "run_emp008_factor_quantiles", evaluate)
    monkeypatch.setattr(type(result), "write_outputs", write)

    run_factor_quantiles.main(["--end", "2024-06-30", "--output-dir", str(tmp_path)])

    prepare.assert_called_once()
    evaluate.assert_called_once_with(prepared=prepared, start=DEFAULT_START, end="2024-06-30", q=5)
    write.assert_called_once_with(tmp_path, factor_set=prepared.config.factor_set, q=5)


def test_main_uses_latest_common_end_when_end_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = make_prepared_bundle()
    result = make_quantile_result()
    prepare = Mock(return_value=prepared)
    evaluate = Mock(return_value=result)
    latest_end = Mock(return_value="2024-05-31")
    write = Mock(return_value={"summary_csv": str(tmp_path / "summary.csv"), "메시지": "완료"})

    monkeypatch.setattr(run_factor_quantiles, "load_and_prepare_emp008_factors", prepare)
    monkeypatch.setattr(run_factor_quantiles, "run_emp008_factor_quantiles", evaluate)
    monkeypatch.setattr(run_factor_quantiles, "latest_common_end", latest_end)
    monkeypatch.setattr(type(result), "write_outputs", write)

    run_factor_quantiles.main(["--output-dir", str(tmp_path)])

    latest_end.assert_called_once()
    evaluate.assert_called_once_with(prepared=prepared, start=DEFAULT_START, end="2024-05-31", q=5)
    assert "\"메시지\": \"완료\"" in capsys.readouterr().out
