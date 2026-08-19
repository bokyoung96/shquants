from __future__ import annotations

import inspect
from typing import Callable

from .base import RegisteredStrategy
from .benchmark_overlay import BenchmarkOverlay
from .benchmark_tilt import BenchmarkTilt
from .earnings_revision import EarningsRevision
from .mfbt import Mfbt
from .revision_signal import RevisionSignal
from .rrg_sector_rotation import (
    RrgSectorRotation,
    RrgSectorRotationOpRrgEx10K1,
    RrgSectorRotationOpRrgEx10K2,
    RrgSectorRotationOpRrgK1,
    RrgSectorRotationOpRrgK2,
    RrgSectorRotationOpRrgMonthly1M,
    RrgSectorRotationOpRrgQavgAccelX128,
    RrgSectorRotationPrune90,
)
from .signal_event_rotation import SignalEventRotation, SignalEventRotationSelected
from .trend_rank import TrendRank


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


register_strategy("trend_rank", TrendRank)
register_strategy("earnings_revision", EarningsRevision)
register_strategy("revision_signal", RevisionSignal)
register_strategy("mfbt", Mfbt)
register_strategy("benchmark_overlay", BenchmarkOverlay)
register_strategy("benchmark_tilt", BenchmarkTilt)
register_strategy("rrg_sector_rotation", RrgSectorRotation)
register_strategy("rrg_sector_rotation_prune90", RrgSectorRotationPrune90)
register_strategy("rrg_sector_rotation_op_rrg_k2", RrgSectorRotationOpRrgK2)
register_strategy("op_rrg_strat", RrgSectorRotationOpRrgMonthly1M)
register_strategy("rrg_sector_rotation_op_rrg_k1", RrgSectorRotationOpRrgK1)
register_strategy("rrg_sector_rotation_op_rrg_ex10_k2", RrgSectorRotationOpRrgEx10K2)
register_strategy("rrg_sector_rotation_op_rrg_ex10_k1", RrgSectorRotationOpRrgEx10K1)
register_strategy("rrg_sector_rotation_op_rrg_qavg_accel_x128", RrgSectorRotationOpRrgQavgAccelX128)
register_strategy("signal_event_rotation", SignalEventRotation)
register_strategy("signal_event_rotation_selected", SignalEventRotationSelected)
