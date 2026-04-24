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
        strategy="momentum",
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


register_preset("kospi200_semiannual_floatcap", _semiannual_floatcap_preset)
