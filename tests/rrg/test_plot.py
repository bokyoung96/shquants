from __future__ import annotations

import pandas as pd

from rrg.dashboard import export_multi_horizon_rrg
from rrg.plot import make_rrg_2d_figure, make_rrg_3d_figure


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


def _quadrant_rows() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    states = {
        "G45": ("Leading", 0.02, 0.03, 1.0),
        "G35": ("Improving", -0.02, 0.03, 0.5),
        "G10": ("Lagging", -0.02, -0.03, -1.0),
        "G40": ("Weakening", 0.02, -0.03, -0.5),
    }
    rows = []
    for sector, (state, rs_value, mom_value, acc_value) in states.items():
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "sector": sector,
                    "horizon": "short",
                    "rs": 1.0,
                    "log_rs": rs_value,
                    "rs_centered": rs_value + i * 0.001,
                    "mom": mom_value + i * 0.001,
                    "acc": acc_value,
                    "acc_z": acc_value,
                    "state": state,
                    "turning_label": "Neutral",
                    "persistence": i + 1,
                    "confidence": 5.0,
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


def test_make_rrg_2d_figure_uses_visible_quadrants_and_sector_names() -> None:
    fig = make_rrg_2d_figure(_quadrant_rows(), trail_length=2)

    marker_traces = [trace for trace in fig.data if "markers" in trace.mode and trace.name in {"Information Technology", "Health Care", "Energy", "Financials"}]

    assert {trace.type for trace in fig.data} == {"scatter"}
    assert len(fig.layout.shapes) >= 6
    assert {"Leading", "Improving", "Lagging", "Weakening"}.issubset({annotation.text for annotation in fig.layout.annotations})
    assert {trace.name for trace in marker_traces} == {"Information Technology", "Health Care", "Energy", "Financials"}
    assert {trace.marker.color for trace in marker_traces} == {"#1f9d55", "#2563eb", "#dc2626", "#f59e0b"}
    assert {trace.marker.symbol for trace in marker_traces} == {"diamond", "circle", "x", "square"}
    assert fig.layout.xaxis.title.text == "Relative Strength (centered log RS)"
    assert fig.layout.yaxis.title.text == "Momentum"
    assert all(shape.xref == "x" and shape.yref == "y" for shape in fig.layout.shapes[:4])
    assert any(shape.x0 == 0 and shape.x1 > 0 and shape.y0 == 0 and shape.y1 > 0 for shape in fig.layout.shapes[:4])
    assert any(shape.x0 < 0 and shape.x1 == 0 and shape.y0 < 0 and shape.y1 == 0 for shape in fig.layout.shapes[:4])


def test_make_rrg_3d_figure_maps_sector_names_and_styles_quadrants() -> None:
    fig = make_rrg_3d_figure(_quadrant_rows(), trail_length=2)

    marker_traces = [trace for trace in fig.data if "marker" in trace.mode and trace.name in {"Information Technology", "Health Care", "Energy", "Financials"}]

    assert {trace.name for trace in marker_traces} == {"Information Technology", "Health Care", "Energy", "Financials"}
    assert {trace.marker.color for trace in marker_traces} == {"#1f9d55", "#2563eb", "#dc2626", "#f59e0b"}
    assert {trace.marker.symbol for trace in marker_traces} == {"diamond", "circle", "x", "square"}
    assert all(float(trace.marker.size) >= 14.0 for trace in marker_traces)


def test_export_multi_horizon_rrg_writes_html(tmp_path) -> None:
    output_path = tmp_path / "rrg.html"

    written = export_multi_horizon_rrg(_rrg_rows(), output_path=output_path, trail_length=2)

    assert written == output_path
    assert output_path.exists()
    assert "plotly" in output_path.read_text(encoding="utf-8").lower()
