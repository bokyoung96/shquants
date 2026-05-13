from __future__ import annotations

import pytest

from dashboard.strategies import DEFAULT_LAUNCH_CONFIG, StrategyPreset, enabled_strategy_presets


def test_default_launch_config_enables_all_active_strategies() -> None:
    enabled = [preset.strategy_name for preset in DEFAULT_LAUNCH_CONFIG.strategies if preset.enabled]
    momentum = DEFAULT_LAUNCH_CONFIG.strategies[0]
    earnings_revision = DEFAULT_LAUNCH_CONFIG.strategies[1]

    assert enabled == ["trend_rank", "earnings_revision", "benchmark_overlay", "benchmark_tilt"]
    assert DEFAULT_LAUNCH_CONFIG.global_config.start == "2020-01-01"
    assert DEFAULT_LAUNCH_CONFIG.global_config.end == "2026-05-11"
    assert DEFAULT_LAUNCH_CONFIG.global_config.fill_mode == "next_open"
    assert DEFAULT_LAUNCH_CONFIG.global_config.fee == 0.0002
    assert DEFAULT_LAUNCH_CONFIG.global_config.sell_tax == 0.0015
    assert DEFAULT_LAUNCH_CONFIG.global_config.slippage == 0.0005
    assert momentum.benchmark.code == "IKS200"
    assert momentum.benchmark.name == "KOSPI200"
    assert momentum.warmup.extra_days > 0
    assert earnings_revision.schedule == "daily"
    assert earnings_revision.fill_mode == "close"


def test_enabled_strategy_presets_filters_disabled_entries() -> None:
    presets = (
        StrategyPreset(enabled=True, strategy_name="trend_rank", display_label="Trend Rank", params={"top_n": 20}),
        StrategyPreset(
            enabled=False,
            strategy_name="experimental_disabled",
            display_label="Experimental Disabled",
            params={"top_n": 20},
        ),
    )

    assert [preset.strategy_name for preset in enabled_strategy_presets(presets)] == ["trend_rank"]


def test_default_launch_config_strategy_params_are_read_only() -> None:
    params = DEFAULT_LAUNCH_CONFIG.strategies[0].params

    with pytest.raises(TypeError):
        params["top_n"] = 25
