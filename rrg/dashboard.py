from __future__ import annotations

from pathlib import Path

import pandas as pd

from .plot import make_rrg_3d_figure


def export_multi_horizon_rrg(
    rrg_frame: pd.DataFrame,
    *,
    output_path: Path | str,
    trail_length: int = 20,
    include_plotlyjs: str | bool = "cdn",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = make_rrg_3d_figure(rrg_frame, trail_length=trail_length)
    fig.write_html(path, include_plotlyjs=include_plotlyjs, full_html=True)
    return path
