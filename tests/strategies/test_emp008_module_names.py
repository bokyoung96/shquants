from __future__ import annotations

from importlib import import_module
from pathlib import Path


def test_emp008_package_uses_neutral_module_and_symbol_names() -> None:
    strategy = import_module("backtesting.strategies.emp008.strategy")
    data = import_module("backtesting.strategies.emp008.data")
    factors = import_module("backtesting.strategies.emp008.factors")

    assert strategy.run_emp008 is not None
    assert strategy.Emp008Result is not None
    assert strategy.apply_expected_alpha_policy is not None
    assert strategy.positive_benchmark_weights is not None
    assert data.Emp008Config is not None
    assert data.load_emp008_market is not None
    assert factors.build_raw_factors is not None


def test_emp008_package_has_no_mfbt_prefixed_implementation_modules() -> None:
    package_dir = Path(__file__).resolve().parents[2] / "backtesting" / "strategies" / "emp008"

    prefixed_entries = sorted(path.name for path in package_dir.glob("mfbt_emp008*"))

    assert prefixed_entries == []


def test_emp008_report_modules_live_in_reports_package() -> None:
    module_names = (
        "adjust_comparison",
        "attribution",
        "comparison",
        "model_comparison",
        "run_model_comparison",
    )

    for module_name in module_names:
        assert import_module(f"backtesting.strategies.emp008.reports.{module_name}") is not None

    package_dir = Path(__file__).resolve().parents[2] / "backtesting" / "strategies" / "emp008"
    old_and_simplified_names = {
        "adjust_comparison.py",
        "adjust_comparison_report.py",
        "attribution.py",
        "comparison.py",
        "model_comparison.py",
        "model_comparison_report.py",
        "run_model_comparison.py",
        "run_model_comparison_report.py",
    }
    root_report_modules = sorted(
        path.name for path in package_dir.iterdir() if path.name in old_and_simplified_names
    )

    assert root_report_modules == []
