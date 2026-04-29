from __future__ import annotations

from dataclasses import dataclass

from backtesting.signals.foreign_reentry_revision_core_band import ForeignReentryRevisionCoreBandSignalProducer

from .event_core_band import EventCoreBandStrategy


@dataclass(slots=True)
class ForeignReentryRevisionCoreBandTopN(EventCoreBandStrategy):
    top_n: int = 10
    core_fraction: float = 0.85
    active_fractions: tuple[float, ...] = (0.05, 0.05, 0.05)
    ratio_lookback: int = 20
    revision_threshold: float = 0.05
    support_momentum_lookback: int = 20

    def __post_init__(self) -> None:
        self.signal_producer = ForeignReentryRevisionCoreBandSignalProducer(
            ratio_lookback=self.ratio_lookback,
            revision_threshold=self.revision_threshold,
            support_momentum_lookback=self.support_momentum_lookback,
        )
        EventCoreBandStrategy.__post_init__(self)
