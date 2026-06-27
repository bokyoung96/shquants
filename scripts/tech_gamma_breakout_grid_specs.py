from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TypeVar

from scripts.run_tech_gamma_long_only import TechGammaConfig


GridItem = TypeVar("GridItem")


@dataclass(frozen=True, slots=True)
class BreakoutStrategySpec:
    name: str
    config: TechGammaConfig


@dataclass(frozen=True, slots=True)
class FeatureKey:
    positivity_lookback_days: int
    positivity_benchmark: str
    positivity_margin: float
    factor_filter: str
    factor_lookback_days: int


def build_strategy_specs(max_strategies: int = 5000) -> list[BreakoutStrategySpec]:
    specs: list[BreakoutStrategySpec] = []
    for values in _sampled_grid(max_strategies):
        config = TechGammaConfig(
            scheme="52w_high_breakout",
            universe="kospi200_historical",
            range_end_hhmm=values["range_end_hhmm"],
            exit_hhmm=values["exit_hhmm"],
            stop_bps=values["stop_bps"],
            trailing_bps=values["trailing_bps"],
            holding_mode=values["holding_mode"],
            min_holding_days=values["min_holding_days"],
            use_positivity=True,
            positivity_lookback_days=values["positivity_lookback"],
            min_daily_positivity=0.0,
            positivity_benchmark=values["benchmark"],
            positivity_margin=values["positivity_margin"],
            factor_filter=values["factor_filter"],
            factor_lookback_days=values["factor_lookback"],
            atr_stop_multiplier=values["atr_multiplier"],
            range_buffer_bps=values["buffer_bps"],
            overnight_enabled=False,
        )
        specs.append(BreakoutStrategySpec(name=_spec_name(config), config=config))
    return specs


def feature_key(config: TechGammaConfig) -> FeatureKey:
    return FeatureKey(
        positivity_lookback_days=config.positivity_lookback_days,
        positivity_benchmark=config.positivity_benchmark,
        positivity_margin=config.positivity_margin,
        factor_filter=config.factor_filter,
        factor_lookback_days=config.factor_lookback_days,
    )


def _sampled_grid(max_strategies: int) -> list[dict[str, int | float | str]]:
    feature_dimensions = list(
        product(
            (40, 60, 90, 126, 252),
            ("sector_cap_weighted", "index_cap_weighted", "sector_equal_weight", "index_equal_weight"),
            (0.0, 0.02, 0.05),
            (
                "none",
                "op_revision_positive",
                "op_sector_rank_positive",
                "foreign_flow_positive",
                "institution_flow_positive",
                "op_or_flow_positive",
            ),
            (40, 60, 90),
        )
    )
    execution_dimensions = list(
        product(
            ("continuation",),
            (1, 2),
            (0.75, 1.0, 1.25, 1.5, 2.0),
            (0.0, 5.0, 8.0, 12.0, 20.0),
            ("0920", "0930"),
            ("1450", "1455"),
            (45.0, 55.0),
            (35.0, 45.0),
        )
    )
    feature_count = min(100, len(feature_dimensions), max(1, max_strategies))
    execution_count = max(1, max_strategies // feature_count)
    chosen_features = _even_sample(feature_dimensions, feature_count)
    chosen_executions = _even_sample(execution_dimensions, execution_count)
    chosen = list(product(chosen_features, chosen_executions))[:max_strategies]
    return [_grid_row(feature, execution) for feature, execution in chosen]


def _grid_row(feature: tuple[object, ...], execution: tuple[object, ...]) -> dict[str, int | float | str]:
    return {
        "positivity_lookback": feature[0],
        "benchmark": feature[1],
        "positivity_margin": feature[2],
        "factor_filter": feature[3],
        "factor_lookback": feature[4],
        "holding_mode": execution[0],
        "min_holding_days": execution[1],
        "atr_multiplier": execution[2],
        "buffer_bps": execution[3],
        "range_end_hhmm": execution[4],
        "exit_hhmm": execution[5],
        "stop_bps": execution[6],
        "trailing_bps": execution[7],
    }


def _even_sample(items: list[GridItem], count: int) -> list[GridItem]:
    if count >= len(items):
        return items
    if count <= 1:
        return [items[0]]
    step = (len(items) - 1) / (count - 1)
    return [items[round(index * step)] for index in range(count)]


def _spec_name(config: TechGammaConfig) -> str:
    atr = str(config.atr_stop_multiplier).replace(".", "")
    buffer = str(config.range_buffer_bps).replace(".", "")
    margin = str(config.positivity_margin).replace(".", "")
    return (
        f"brk_{config.positivity_benchmark}_pos{config.positivity_lookback_days}"
        f"_m{margin}_{config.factor_filter}_flb{config.factor_lookback_days}"
        f"_{config.holding_mode}_h{config.min_holding_days}_atr{atr}_buf{buffer}"
        f"_r{config.range_end_hhmm}_x{config.exit_hhmm}_s{int(config.stop_bps)}_t{int(config.trailing_bps)}"
    )
