from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from .plot import make_rrg_2d_figure, make_rrg_3d_figure


def export_multi_horizon_rrg(
    rrg_frame: pd.DataFrame,
    *,
    output_path: Path | str,
    trail_length: int = 12,
    sector_name_map: Mapping[str, str] | None = None,
    mode: str = "quadrant_2d",
    include_plotlyjs: str | bool = "cdn",
) -> Path:
    if mode not in {"quadrant_2d", "phase_3d"}:
        raise ValueError(f"unsupported RRG export mode: {mode}")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "phase_3d":
        fig = make_rrg_3d_figure(rrg_frame, trail_length=trail_length, sector_name_map=sector_name_map)
    else:
        fig = make_rrg_2d_figure(rrg_frame, trail_length=trail_length, sector_name_map=sector_name_map)
    fig.write_html(path, include_plotlyjs=include_plotlyjs, full_html=True)
    return path
