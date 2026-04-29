from __future__ import annotations

from dataclasses import dataclass

from backtesting.signals.retail_capitulation_rebound_core_band import RetailCapitulationReboundCoreBandSignalProducer

from .event_core_band import EventCoreBandStrategy


@dataclass(slots=True)
class RetailCapitulationReboundCoreBandTopN(EventCoreBandStrategy):
    top_n: int = 10
    core_fraction: float = 0.85
    active_fractions: tuple[float, ...] = (0.05, 0.05, 0.05)
    flow_lookback: int = 10
    rebound_lookback: int = 5
    retail_threshold: float = -0.02

    def __post_init__(self) -> None:
        self.signal_producer = RetailCapitulationReboundCoreBandSignalProducer(
            flow_lookback=self.flow_lookback,
            rebound_lookback=self.rebound_lookback,
            retail_threshold=self.retail_threshold,
        )
        EventCoreBandStrategy.__post_init__(self)
