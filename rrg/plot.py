from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import plotly.graph_objects as go


DEFAULT_WICS_SECTOR_NAME_MAP: dict[str, str] = {
    "G10": "Energy",
    "G15": "Materials",
    "G20": "Industrials",
    "G25": "Consumer Discretionary",
    "G30": "Consumer Staples",
    "G35": "Health Care",
    "G40": "Financials",
    "G45": "Information Technology",
    "G50": "Communication Services",
    "G55": "Utilities",
}

STATE_COLORS: dict[str, str] = {
    "Leading": "#1f9d55",
    "Improving": "#2563eb",
    "Lagging": "#dc2626",
    "Weakening": "#f59e0b",
    "Unclassified": "#6b7280",
}

STATE_SYMBOLS: dict[str, str] = {
    "Leading": "diamond",
    "Improving": "circle",
    "Lagging": "x",
    "Weakening": "square",
    "Unclassified": "circle-open",
}


def make_rrg_3d_figure(
    rrg_frame: pd.DataFrame,
    *,
    trail_length: int = 12,
    z_column: str = "acc_z",
    title: str = "Advanced 3D Relative Rotation Graph",
    sector_name_map: Mapping[str, str] | None = None,
) -> go.Figure:
    if trail_length <= 0:
        raise ValueError("trail_length must be positive")
    required = {"date", "sector", "horizon", "rs_centered", "mom", z_column, "acc_z", "state", "turning_label", "persistence", "confidence"}
    missing = required.difference(rrg_frame.columns)
    if missing:
        raise ValueError(f"missing RRG columns: {sorted(missing)}")

    horizons = [str(horizon) for horizon in rrg_frame["horizon"].dropna().drop_duplicates()]
    frame = rrg_frame.sort_values(["sector", "date"]).copy()
    display_names = {**DEFAULT_WICS_SECTOR_NAME_MAP, **{str(key): str(value) for key, value in (sector_name_map or {}).items()}}
    fig = go.Figure()
    trace_horizons: list[str] = []
    for horizon in horizons:
        horizon_frame = frame[frame["horizon"].eq(horizon)]
        visible = horizon == horizons[0]
        for sector, sector_frame in horizon_frame.groupby("sector", sort=True):
            trail = sector_frame.dropna(subset=["rs_centered", "mom", z_column]).tail(trail_length)
            if trail.empty:
                continue
            latest = trail.iloc[-1]
            sector_code = str(sector)
            sector_label = display_names.get(sector_code, sector_code)
            state = str(latest["state"])
            state_color = STATE_COLORS.get(state, STATE_COLORS["Unclassified"])
            state_symbol = STATE_SYMBOLS.get(state, STATE_SYMBOLS["Unclassified"])
            fig.add_trace(
                go.Scatter3d(
                    x=trail["rs_centered"],
                    y=trail["mom"],
                    z=trail[z_column],
                    mode="lines",
                    line={"width": 6, "color": state_color},
                    opacity=0.55,
                    name=f"{sector_label} trail",
                    legendgroup=sector_label,
                    showlegend=False,
                    visible=visible,
                    hoverinfo="skip",
                )
            )
            trace_horizons.append(horizon)
            fig.add_trace(
                go.Scatter3d(
                    x=[latest["rs_centered"]],
                    y=[latest["mom"]],
                    z=[latest[z_column]],
                    mode="markers+text",
                    text=[sector_label],
                    textposition="top center",
                    marker={
                        "size": max(14.0, min(30.0, 12.0 + float(latest["confidence"]))),
                        "color": state_color,
                        "symbol": state_symbol,
                        "opacity": 0.96,
                        "line": {"color": "#111827", "width": 2},
                    },
                    customdata=[
                        [
                            sector_code,
                            latest["date"],
                            latest["state"],
                            latest["turning_label"],
                            latest["persistence"],
                            latest["rs_centered"],
                            latest["mom"],
                            latest[z_column],
                        ]
                    ],
                    hovertemplate=(
                        "sector=%{text}<br>"
                        "code=%{customdata[0]}<br>"
                        "date=%{customdata[1]|%Y-%m-%d}<br>"
                        "state=%{customdata[2]}<br>"
                        "turning=%{customdata[3]}<br>"
                        "persistence=%{customdata[4]}<br>"
                        "RS centered=%{customdata[5]:.4f}<br>"
                        "MOM=%{customdata[6]:.4f}<br>"
                        "Z=%{customdata[7]:.4f}<extra></extra>"
                    ),
                    name=sector_label,
                    legendgroup=sector_label,
                    visible=visible,
                )
            )
            trace_horizons.append(horizon)
        for trace in _reference_traces(horizon_frame, z_column=z_column, visible=visible):
            fig.add_trace(trace)
            trace_horizons.append(horizon)
        for trace in _state_legend_traces(visible=visible):
            fig.add_trace(trace)
            trace_horizons.append(horizon)

    buttons = []
    for horizon in horizons:
        visibility = [trace_horizon == horizon for trace_horizon in trace_horizons]
        buttons.append(
            {
                "label": horizon,
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {"title": f"{title} - {horizon}"},
                ],
            }
        )

    fig.update_layout(
        title=f"{title} - {horizons[0]}" if horizons else title,
        scene={
            "xaxis_title": "Relative Strength (centered log RS)",
            "yaxis_title": "Momentum",
            "zaxis_title": z_column,
            "xaxis": {"zeroline": True, "showbackground": True, "backgroundcolor": "#f8fafc"},
            "yaxis": {"zeroline": True, "showbackground": True, "backgroundcolor": "#f8fafc"},
            "zaxis": {"zeroline": True, "showbackground": True, "backgroundcolor": "#f8fafc"},
        },
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.0, "y": 1.12}] if buttons else [],
        legend={"title": {"text": "Sector / State"}, "itemsizing": "constant"},
        margin={"l": 0, "r": 0, "b": 0, "t": 70},
    )
    return fig


def make_rrg_2d_figure(
    rrg_frame: pd.DataFrame,
    *,
    trail_length: int = 12,
    title: str = "Advanced Relative Rotation Graph",
    sector_name_map: Mapping[str, str] | None = None,
) -> go.Figure:
    if trail_length <= 0:
        raise ValueError("trail_length must be positive")
    required = {"date", "sector", "horizon", "rs_centered", "mom", "acc_z", "state", "turning_label", "persistence", "confidence"}
    missing = required.difference(rrg_frame.columns)
    if missing:
        raise ValueError(f"missing RRG columns: {sorted(missing)}")

    horizons = [str(horizon) for horizon in rrg_frame["horizon"].dropna().drop_duplicates()]
    frame = rrg_frame.sort_values(["sector", "date"]).copy()
    display_names = {**DEFAULT_WICS_SECTOR_NAME_MAP, **{str(key): str(value) for key, value in (sector_name_map or {}).items()}}
    x_range, y_range = _axis_ranges_2d(frame)
    fig = go.Figure()
    trace_horizons: list[str] = []
    for horizon in horizons:
        horizon_frame = frame[frame["horizon"].eq(horizon)]
        visible = horizon == horizons[0]
        for sector, sector_frame in horizon_frame.groupby("sector", sort=True):
            trail = sector_frame.dropna(subset=["rs_centered", "mom"]).tail(trail_length)
            if trail.empty:
                continue
            latest = trail.iloc[-1]
            sector_code = str(sector)
            sector_label = display_names.get(sector_code, sector_code)
            state = str(latest["state"])
            state_color = STATE_COLORS.get(state, STATE_COLORS["Unclassified"])
            state_symbol = STATE_SYMBOLS.get(state, STATE_SYMBOLS["Unclassified"])
            fig.add_trace(
                go.Scatter(
                    x=trail["rs_centered"],
                    y=trail["mom"],
                    mode="lines",
                    line={"width": 3, "color": state_color},
                    opacity=0.42,
                    name=f"{sector_label} trail",
                    legendgroup=sector_label,
                    showlegend=False,
                    visible=visible,
                    hoverinfo="skip",
                )
            )
            trace_horizons.append(horizon)
            marker_size = max(13.0, min(28.0, 13.0 + abs(float(latest["acc_z"])) * 2.5 + float(latest["confidence"]) * 0.35))
            fig.add_trace(
                go.Scatter(
                    x=[latest["rs_centered"]],
                    y=[latest["mom"]],
                    mode="markers+text",
                    text=[sector_label],
                    textposition=_text_position_for_state(state),
                    marker={
                        "size": marker_size,
                        "color": state_color,
                        "symbol": state_symbol,
                        "opacity": 0.96,
                        "line": {"color": "#111827", "width": 2},
                    },
                    customdata=[
                        [
                            sector_code,
                            latest["date"],
                            latest["state"],
                            latest["turning_label"],
                            latest["persistence"],
                            latest["rs_centered"],
                            latest["mom"],
                            latest["acc_z"],
                        ]
                    ],
                    hovertemplate=(
                        "sector=%{text}<br>"
                        "code=%{customdata[0]}<br>"
                        "date=%{customdata[1]|%Y-%m-%d}<br>"
                        "state=%{customdata[2]}<br>"
                        "turning=%{customdata[3]}<br>"
                        "persistence=%{customdata[4]}<br>"
                        "RS centered=%{customdata[5]:.4f}<br>"
                        "MOM=%{customdata[6]:.4f}<br>"
                        "ACC z=%{customdata[7]:.4f}<extra></extra>"
                    ),
                    name=sector_label,
                    legendgroup=sector_label,
                    visible=visible,
                )
            )
            trace_horizons.append(horizon)
        for trace in _state_legend_traces_2d(visible=visible):
            fig.add_trace(trace)
            trace_horizons.append(horizon)

    buttons = []
    for horizon in horizons:
        visibility = [trace_horizon == horizon for trace_horizon in trace_horizons]
        buttons.append(
            {
                "label": horizon,
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {"title": f"{title} - {horizon}"},
                ],
            }
        )

    fig.update_layout(
        title=f"{title} - {horizons[0]}" if horizons else title,
        shapes=_quadrant_shapes(x_range=x_range, y_range=y_range),
        annotations=_quadrant_annotations(),
        xaxis={
            "title": "Relative Strength (centered log RS)",
            "range": x_range,
            "zeroline": True,
            "zerolinewidth": 2,
            "zerolinecolor": "#111827",
            "showgrid": True,
            "gridcolor": "#e5e7eb",
        },
        yaxis={
            "title": "Momentum",
            "range": y_range,
            "zeroline": True,
            "zerolinewidth": 2,
            "zerolinecolor": "#111827",
            "showgrid": True,
            "gridcolor": "#e5e7eb",
        },
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.0, "y": 1.12}] if buttons else [],
        legend={"title": {"text": "Sector / State"}, "itemsizing": "constant"},
        margin={"l": 70, "r": 220, "b": 70, "t": 80},
        height=900,
    )
    return fig


def _reference_traces(horizon_frame: pd.DataFrame, *, z_column: str, visible: bool) -> list[go.Scatter3d]:
    usable = horizon_frame.dropna(subset=["rs_centered", "mom", z_column])
    if usable.empty:
        return []
    x_min, x_max = _span(usable["rs_centered"])
    y_min, y_max = _span(usable["mom"])
    z_min, z_max = _span(usable[z_column])
    line = {"width": 3, "color": "rgba(17, 24, 39, 0.32)", "dash": "dash"}
    return [
        go.Scatter3d(
            x=[0.0, 0.0],
            y=[y_min, y_max],
            z=[0.0, 0.0],
            mode="lines",
            line=line,
            name="MOM axis",
            showlegend=False,
            visible=visible,
            hoverinfo="skip",
        ),
        go.Scatter3d(
            x=[x_min, x_max],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            mode="lines",
            line=line,
            name="RS axis",
            showlegend=False,
            visible=visible,
            hoverinfo="skip",
        ),
        go.Scatter3d(
            x=[0.0, 0.0],
            y=[0.0, 0.0],
            z=[z_min, z_max],
            mode="lines",
            line=line,
            name="ACC axis",
            showlegend=False,
            visible=visible,
            hoverinfo="skip",
        ),
    ]


def _state_legend_traces(*, visible: bool) -> list[go.Scatter3d]:
    traces: list[go.Scatter3d] = []
    for state in ("Leading", "Improving", "Lagging", "Weakening"):
        traces.append(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker={
                    "size": 12,
                    "color": STATE_COLORS[state],
                    "symbol": STATE_SYMBOLS[state],
                    "line": {"color": "#111827", "width": 1},
                },
                name=state,
                legendgroup=f"state-{state}",
                showlegend=visible,
                visible=visible,
                hoverinfo="skip",
            )
        )
    return traces


def _state_legend_traces_2d(*, visible: bool) -> list[go.Scatter]:
    traces: list[go.Scatter] = []
    for state in ("Leading", "Improving", "Lagging", "Weakening"):
        traces.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "size": 12,
                    "color": STATE_COLORS[state],
                    "symbol": STATE_SYMBOLS[state],
                    "line": {"color": "#111827", "width": 1},
                },
                name=state,
                legendgroup=f"state-{state}",
                showlegend=visible,
                visible=visible,
                hoverinfo="skip",
            )
        )
    return traces


def _quadrant_shapes(*, x_range: tuple[float, float], y_range: tuple[float, float]) -> list[dict[str, object]]:
    x_min, x_max = x_range
    y_min, y_max = y_range
    return [
        _rect(0.0, x_max, 0.0, y_max, "rgba(31, 157, 85, 0.13)"),
        _rect(x_min, 0.0, 0.0, y_max, "rgba(37, 99, 235, 0.12)"),
        _rect(x_min, 0.0, y_min, 0.0, "rgba(220, 38, 38, 0.12)"),
        _rect(0.0, x_max, y_min, 0.0, "rgba(245, 158, 11, 0.16)"),
        {"type": "line", "xref": "paper", "yref": "y", "x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 0.0, "line": {"color": "#111827", "width": 2}},
        {"type": "line", "xref": "x", "yref": "paper", "x0": 0.0, "x1": 0.0, "y0": 0.0, "y1": 1.0, "line": {"color": "#111827", "width": 2}},
    ]


def _quadrant_annotations() -> list[dict[str, object]]:
    return [
        _annotation(0.985, 0.985, "Leading", "#166534"),
        _annotation(0.015, 0.985, "Improving", "#1d4ed8"),
        _annotation(0.015, 0.025, "Lagging", "#991b1b"),
        _annotation(0.985, 0.025, "Weakening", "#92400e"),
    ]


def _rect(x0: float, x1: float, y0: float, y1: float, fillcolor: str) -> dict[str, object]:
    return {
        "type": "rect",
        "xref": "x",
        "yref": "y",
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "fillcolor": fillcolor,
        "line": {"width": 0},
        "layer": "below",
    }


def _annotation(x: float, y: float, text: str, color: str) -> dict[str, object]:
    return {
        "xref": "paper",
        "yref": "paper",
        "x": x,
        "y": y,
        "text": text,
        "showarrow": False,
        "font": {"size": 18, "color": color},
        "bgcolor": "rgba(255, 255, 255, 0.78)",
        "bordercolor": color,
        "borderwidth": 1,
    }


def _text_position_for_state(state: str) -> str:
    if state == "Leading":
        return "top right"
    if state == "Improving":
        return "top left"
    if state == "Lagging":
        return "bottom left"
    if state == "Weakening":
        return "bottom right"
    return "top center"


def _axis_ranges_2d(frame: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    usable = frame.dropna(subset=["rs_centered", "mom"])
    if usable.empty:
        return (-1.0, 1.0), (-1.0, 1.0)
    x_min, x_max = _span(usable["rs_centered"])
    y_min, y_max = _span(usable["mom"])
    x_min = min(x_min, -0.01)
    x_max = max(x_max, 0.01)
    y_min = min(y_min, -0.01)
    y_max = max(y_max, 0.01)
    return (x_min, x_max), (y_min, y_max)


def _span(values: pd.Series) -> tuple[float, float]:
    lower = float(values.min())
    upper = float(values.max())
    if lower == upper:
        padding = max(abs(lower) * 0.1, 0.01)
        return lower - padding, upper + padding
    padding = (upper - lower) * 0.08
    return lower - padding, upper + padding
