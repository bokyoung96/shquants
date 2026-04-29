from __future__ import annotations

from dataclasses import dataclass

from backtesting.signals.foreign_ownership_jump_core_band import ForeignOwnershipJumpCoreBandSignalProducer

from .event_core_band import EventCoreBandStrategy


@dataclass(slots=True)
class ForeignOwnershipJumpCoreBandTopN(EventCoreBandStrategy):
    top_n: int = 10
    core_fraction: float = 0.85
    active_fractions: tuple[float, ...] = (0.05, 0.05, 0.05)
    change_lookback: int = 5
    flow_lookback: int = 20
    min_ratio_change: float = 0.002
    support_momentum_lookback: int = 20

    def __post_init__(self) -> None:
        self.signal_producer = ForeignOwnershipJumpCoreBandSignalProducer(
            change_lookback=self.change_lookback,
            flow_lookback=self.flow_lookback,
            min_ratio_change=self.min_ratio_change,
            support_momentum_lookback=self.support_momentum_lookback,
        )
        EventCoreBandStrategy.__post_init__(self)
