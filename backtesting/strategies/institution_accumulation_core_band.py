from __future__ import annotations

from dataclasses import dataclass

from backtesting.signals.institution_accumulation_core_band import InstitutionAccumulationCoreBandSignalProducer

from .event_core_band import EventCoreBandStrategy


@dataclass(slots=True)
class InstitutionAccumulationCoreBandTopN(EventCoreBandStrategy):
    top_n: int = 10
    core_fraction: float = 0.85
    active_fractions: tuple[float, ...] = (0.05, 0.05, 0.05)
    flow_lookback: int = 15
    support_momentum_lookback: int = 20
    min_inst_intensity: float = 0.015

    def __post_init__(self) -> None:
        self.signal_producer = InstitutionAccumulationCoreBandSignalProducer(
            flow_lookback=self.flow_lookback,
            support_momentum_lookback=self.support_momentum_lookback,
            min_inst_intensity=self.min_inst_intensity,
        )
        EventCoreBandStrategy.__post_init__(self)
