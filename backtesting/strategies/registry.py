from __future__ import annotations

import inspect
from typing import Callable

from .base import RegisteredStrategy
from .breakout_staged import Breakout52WeekStaged
from .breakout_simple import Breakout52WeekSimple
from .breakout_52w_nearness import Breakout52WeekNearnessTopN
from .momentum import MomentumTopN
from .op_fwd import OpFwdYieldTopN


StrategyFactory = Callable[..., RegisteredStrategy]

_REGISTRY: dict[str, StrategyFactory] = {}


def register_strategy(name: str, factory: StrategyFactory) -> None:
    if name in _REGISTRY:
        raise ValueError(f"strategy already registered: {name}")
    _REGISTRY[name] = factory


def build_strategy(name: str, **kwargs: object) -> RegisteredStrategy:
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown strategy '{name}'. Available: {available}") from exc
    params = inspect.signature(factory).parameters
    filtered = {key: value for key, value in kwargs.items() if key in params}
    return factory(**filtered)


def list_strategies() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


register_strategy("momentum", MomentumTopN)
register_strategy("op_fwd_yield", OpFwdYieldTopN)
register_strategy("breakout_52w_simple", Breakout52WeekSimple)
register_strategy("breakout_52w_staged", Breakout52WeekStaged)
register_strategy("breakout_52w_nearness", Breakout52WeekNearnessTopN)
