from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType

from backtesting.reporting.models import BenchmarkConfig


@dataclass(frozen=True, slots=True)
class GlobalRunConfig:
    start: str = "2020-01-01"
    end: str = "2025-12-31"
    capital: float = 100_000_000.0
    schedule: str = "monthly"
    fill_mode: str = "next_open"
    fee: float = 0.0
    sell_tax: float = 0.0
    slippage: float = 0.0
    use_k200: bool = True
    allow_fractional: bool = True


@dataclass(frozen=True, slots=True)
class WarmupConfig:
    extra_days: int = 0


@dataclass(frozen=True, slots=True)
class StrategyPreset:
    enabled: bool
    strategy_name: str
    display_label: str
    params: Mapping[str, object] = field(default_factory=dict)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig.default_kospi200)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    universe_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True, slots=True)
class DashboardLaunchConfig:
    global_config: GlobalRunConfig
    strategies: tuple[StrategyPreset, ...]


DEFAULT_LAUNCH_CONFIG = DashboardLaunchConfig(
    global_config=GlobalRunConfig(),
    strategies=(
        StrategyPreset(
            True,
            "momentum",
            "Momentum",
            {"top_n": 20, "lookback": 20},
            warmup=WarmupConfig(extra_days=20),
        ),
    ),
)


def enabled_strategy_presets(presets: tuple[StrategyPreset, ...]) -> tuple[StrategyPreset, ...]:
    return tuple(preset for preset in presets if preset.enabled)
