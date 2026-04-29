from __future__ import annotations

from dataclasses import dataclass

from backtesting.signals.revision_event_core_band import RevisionEventCoreBandSignalProducer

from .event_core_band import EventCoreBandStrategy


@dataclass(slots=True)
class RevisionEventCoreBandTopN(EventCoreBandStrategy):
    top_n: int = 10
    core_fraction: float = 0.85
    active_fractions: tuple[float, ...] = (0.05, 0.05, 0.05)
    revision_threshold: float = 0.08
    momentum_lookback: int = 20

    def __post_init__(self) -> None:
        self.signal_producer = RevisionEventCoreBandSignalProducer(
            revision_threshold=self.revision_threshold,
            momentum_lookback=self.momentum_lookback,
        )
        EventCoreBandStrategy.__post_init__(self)
