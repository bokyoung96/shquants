from __future__ import annotations

from dataclasses import dataclass

from backtesting.signals.creative_event_asymmetry import CreativeEventAsymmetrySignalProducer

from .event_core_band import EventCoreBandStrategy


@dataclass(slots=True)
class CreativeEventAsymmetryTopN(EventCoreBandStrategy):
    top_n: int = 8
    core_fraction: float = 0.70
    active_fractions: tuple[float, ...] = (0.15, 0.10, 0.05)
    revision_threshold: float = 0.04
    flow_lookback: int = 20
    support_lookback: int = 20

    def __post_init__(self) -> None:
        self.signal_producer = CreativeEventAsymmetrySignalProducer(
            revision_threshold=self.revision_threshold,
            flow_lookback=self.flow_lookback,
            support_lookback=self.support_lookback,
        )
        EventCoreBandStrategy.__post_init__(self)
