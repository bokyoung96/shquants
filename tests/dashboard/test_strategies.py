from __future__ import annotations

import pytest

from dashboard.strategies import DEFAULT_LAUNCH_CONFIG, StrategyPreset, enabled_strategy_presets


def test_default_launch_config_enables_momentum_strategy() -> None:
    enabled = [preset.strategy_name for preset in DEFAULT_LAUNCH_CONFIG.strategies if preset.enabled]
    momentum = DEFAULT_LAUNCH_CONFIG.strategies[0]

    assert enabled == ["momentum"]
    assert DEFAULT_LAUNCH_CONFIG.global_config.start == "2020-01-01"
    assert DEFAULT_LAUNCH_CONFIG.global_config.fill_mode == "next_open"
    assert momentum.benchmark.code == "IKS200"
    assert momentum.benchmark.name == "KOSPI200"
    assert momentum.warmup.extra_days > 0


def test_enabled_strategy_presets_filters_disabled_entries() -> None:
    presets = (
        StrategyPreset(enabled=True, strategy_name="momentum", display_label="Momentum", params={"top_n": 20}),
        StrategyPreset(
            enabled=False,
            strategy_name="experimental_disabled",
            display_label="Experimental Disabled",
            params={"top_n": 20},
        ),
    )

    assert [preset.strategy_name for preset in enabled_strategy_presets(presets)] == ["momentum"]


def test_default_launch_config_strategy_params_are_read_only() -> None:
    params = DEFAULT_LAUNCH_CONFIG.strategies[0].params

    with pytest.raises(TypeError):
        params["top_n"] = 25
