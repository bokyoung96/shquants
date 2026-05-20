from __future__ import annotations

import pandas as pd

from rrg.dashboard import export_multi_horizon_rrg
from rrg.plot import make_rrg_3d_figure


def _rrg_rows() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    rows = []
    for horizon in ("short", "medium"):
        for sector in ("Tech", "Finance"):
            for i, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "sector": sector,
                        "horizon": horizon,
                        "rs": 1.0 + i * 0.01,
                        "log_rs": i * 0.01,
                        "rs_centered": i * 0.01 if sector == "Tech" else -i * 0.01,
                        "mom": i * 0.02,
                        "acc": (-1.0 if sector == "Finance" else 1.0) * i * 0.01,
                        "acc_z": (-1.0 if sector == "Finance" else 1.0) * i,
                        "state": "Leading" if sector == "Tech" else "Improving",
                        "turning_label": "Trend strengthening",
                        "persistence": i + 1,
                        "confidence": 0.5 + i * 0.1,
                    }
                )
    return pd.DataFrame(rows)


def test_make_rrg_3d_figure_contains_scatter3d_traces_and_horizon_dropdown() -> None:
    fig = make_rrg_3d_figure(_rrg_rows(), trail_length=2)

    assert fig.data
    assert {trace.type for trace in fig.data} == {"scatter3d"}
    assert fig.layout.updatemenus
    buttons = fig.layout.updatemenus[0].buttons
    assert [button.label for button in buttons] == ["short", "medium"]


def test_export_multi_horizon_rrg_writes_html(tmp_path) -> None:
    output_path = tmp_path / "rrg.html"

    written = export_multi_horizon_rrg(_rrg_rows(), output_path=output_path, trail_length=2)

    assert written == output_path
    assert output_path.exists()
    assert "plotly" in output_path.read_text(encoding="utf-8").lower()
