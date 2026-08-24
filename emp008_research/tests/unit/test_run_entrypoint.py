from dataclasses import replace

import pytest

from run import RunSettings, run_settings


def test_run_settings_has_beginner_defaults():
    settings = RunSettings()
    assert settings.factor_set == "production_core"
    assert not hasattr(settings, "risk_model")
    assert not hasattr(settings, "expected_alpha_estimator")
    assert settings.convert_raw_to_parquet is False
    assert settings.run_backtest is True


def test_run_settings_rejects_invalid_date_range(tmp_path):
    settings = replace(RunSettings(), start="2024-02-01", end="2024-01-01")
    with pytest.raises(ValueError, match="end must be on or after start"):
        run_settings(settings, project_dir=tmp_path)
