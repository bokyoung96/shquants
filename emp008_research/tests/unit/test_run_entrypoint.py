from dataclasses import replace

import pytest

from run import RunSettings, run_settings
from emp008.run_weights import build_emp008_config
from data.catalog import DatasetId


def test_run_settings_has_beginner_defaults():
    settings = RunSettings()
    assert settings.factor_set == "production_core"
    assert settings.sector_neutral_dataset == "wi26"
    assert not hasattr(settings, "risk_model")
    assert not hasattr(settings, "expected_alpha_estimator")
    assert settings.convert_raw_to_parquet is False
    assert settings.run_backtest is True


def test_run_settings_rejects_invalid_date_range(tmp_path):
    settings = replace(RunSettings(), start="2024-02-01", end="2024-01-01")
    with pytest.raises(ValueError, match="end must be on or after start"):
        run_settings(settings, project_dir=tmp_path)


def test_run_settings_rejects_unknown_sector_taxonomy(tmp_path):
    settings = replace(RunSettings(), sector_neutral_dataset="unknown")
    with pytest.raises(ValueError, match="sector_neutral_dataset"):
        run_settings(settings, project_dir=tmp_path)


def test_sector_taxonomy_setting_maps_to_wics_dataset():
    config = build_emp008_config(sector_neutral_dataset="wics")
    assert config.sector_neutral_dataset is DatasetId.QW_WICS_SEC_BIG
