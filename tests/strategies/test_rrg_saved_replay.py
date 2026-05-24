from pathlib import Path

import pandas as pd
from pandas.testing import assert_series_equal

from backtesting.run import BacktestRunner, RunConfig


def test_rrg_saved_strategy_replays_archived_state_equal_holdcap_weights() -> None:
    saved = pd.read_parquet("results/backtests/rrg_20260519_174931/positions/weights.parquet").fillna(0.0)
    runner = BacktestRunner(
        raw_dir=Path("raw"),
        parquet_dir=Path("parquet"),
        result_dir=Path("/tmp/shquants-rrg-test"),
        write_report_assets=False,
    )
    report = runner.run(
        RunConfig(
            start="2020-01-02",
            end="2020-04-10",
            strategy="rrg_sector_rotation",
            name="rrg-saved-replay-test",
            top_n=25,
            lookback=20,
            schedule="weekly",
            fill_mode="close",
            strategy_params={
                "gross_short": 0,
                "alpha_mode": "fwd_only",
                "use_name_cap": False,
                "sector_budget_mode": "state_equal",
                "fwd_entry_rule": "majority_horizons",
                "hold_weakening_longs": True,
                "hold_long_mode": "cap",
            },
        )
    )

    for date in ("2020-03-05", "2020-03-06", "2020-04-07", "2020-04-10"):
        actual = report.position_plan.target_weights.loc[date].reindex(saved.columns).fillna(0.0)
        expected = saved.loc[date].reindex(saved.columns).fillna(0.0)

        assert_series_equal(actual, expected, check_names=False, atol=1e-12, rtol=1e-12)
