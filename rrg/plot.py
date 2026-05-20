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
    trail_length: int = 20,
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


def _span(values: pd.Series) -> tuple[float, float]:
    lower = float(values.min())
    upper = float(values.max())
    if lower == upper:
        padding = max(abs(lower) * 0.1, 0.01)
        return lower - padding, upper + padding
    padding = (upper - lower) * 0.08
    return lower - padding, upper + padding
