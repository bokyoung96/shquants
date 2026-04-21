from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True, slots=True)
class SignalBundle:
    alpha: pd.DataFrame
    context: dict[str, pd.DataFrame]
    meta: dict[str, pd.DataFrame | pd.Series] = field(default_factory=dict)
