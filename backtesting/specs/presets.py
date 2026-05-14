from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pandas as pd

from .models import DataPolicySpec, ExecutionSpec, ScheduleSpec, WeightSourceSpec


PresetFactory = Callable[[], ExecutionSpec]
_PRESETS: dict[str, PresetFactory] = {}


def register_preset(preset_id: str, factory: PresetFactory) -> None:
    if preset_id in _PRESETS:
        raise ValueError(f"preset already registered: {preset_id}")
    _PRESETS[preset_id] = factory


def get_preset(preset_id: str) -> ExecutionSpec:
    try:
        return _PRESETS[preset_id]()
    except KeyError as exc:
        raise KeyError(f"unknown preset_id: {preset_id}") from exc


def _semiannual_floatcap_preset() -> ExecutionSpec:
    return ExecutionSpec(
        start="2019-01-01",
        end=pd.Timestamp.today().date().isoformat(),
        strategy="trend_rank",
        name="kospi200_semiannual_floatcap_close_v1",
        schedule=ScheduleSpec(kind="custom_dates"),
        fill_mode="close",
        use_k200=True,
        allow_fractional=True,
        weight_source=WeightSourceSpec(kind="hook", hook_id="kospi200_semiannual_floatcap"),
        data_policy=DataPolicySpec(
            requested_weight_basis="float_market_cap",
            fallback_order=("market_cap",),
        ),
        spec_source="preset",
        preset_id="kospi200_semiannual_floatcap",
        notes=("Use the second Thursday of June and December as rebalance dates.",),
    )


def _revision_signal_preset() -> ExecutionSpec:
    return ExecutionSpec(
        start="2015-01-02",
        end=pd.Timestamp.today().date().isoformat(),
        strategy="revision_signal",
        name="kospi200_revision_signal_close_v1",
        top_n=0,
        lookback=20,
        warmup_days=180,
        schedule=ScheduleSpec(kind="signal_dates", name=None),
        fill_mode="close",
        fee=0.0002,
        sell_tax=0.0015,
        slippage=0.0005,
        use_k200=True,
        allow_fractional=True,
        spec_source="preset",
        preset_id="kospi200_revision_signal",
        notes=(
            "Hold all KOSPI200 names with positive 20-day EPS and OP forward revisions.",
            "Move to cash when the KOSPI200 benchmark is below its fixed 120-day trend average.",
            "Use signal_dates so trades occur only when target weights change.",
        ),
    )


register_preset("kospi200_semiannual_floatcap", _semiannual_floatcap_preset)
register_preset("kospi200_revision_signal", _revision_signal_preset)
