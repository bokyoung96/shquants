from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def make_rrg_3d_figure(
    rrg_frame: pd.DataFrame,
    *,
    trail_length: int = 20,
    z_column: str = "acc_z",
    title: str = "Advanced 3D Relative Rotation Graph",
) -> go.Figure:
    if trail_length <= 0:
        raise ValueError("trail_length must be positive")
    required = {"date", "sector", "horizon", "rs_centered", "mom", z_column, "acc_z", "state", "turning_label", "persistence", "confidence"}
    missing = required.difference(rrg_frame.columns)
    if missing:
        raise ValueError(f"missing RRG columns: {sorted(missing)}")

    horizons = [str(horizon) for horizon in rrg_frame["horizon"].dropna().drop_duplicates()]
    frame = rrg_frame.sort_values(["sector", "date"]).copy()
    fig = go.Figure()
    trace_horizons: list[str] = []
    for horizon in horizons:
        horizon_frame = frame[frame["horizon"].eq(horizon)]
        visible = horizon == horizons[0]
        for sector, sector_frame in horizon_frame.groupby("sector", sort=True):
            trail = sector_frame.dropna(subset=["rs_centered", "mom", z_column]).tail(trail_length)
            if trail.empty:
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=trail["rs_centered"],
                    y=trail["mom"],
                    z=trail[z_column],
                    mode="lines",
                    line={"width": 3, "color": "rgba(120, 120, 120, 0.45)"},
                    name=f"{sector} trail",
                    legendgroup=str(sector),
                    showlegend=False,
                    visible=visible,
                    hoverinfo="skip",
                )
            )
            trace_horizons.append(horizon)
            latest = trail.iloc[-1]
            fig.add_trace(
                go.Scatter3d(
                    x=[latest["rs_centered"]],
                    y=[latest["mom"]],
                    z=[latest[z_column]],
                    mode="markers+text",
                    text=[str(sector)],
                    textposition="top center",
                    marker={
                        "size": max(5.0, min(18.0, 6.0 + float(latest["confidence"]))),
                        "color": [latest["acc_z"]],
                        "colorscale": "RdYlGn",
                        "cmin": -2.0,
                        "cmax": 2.0,
                        "colorbar": {"title": "ACC z"} if not fig.data else None,
                    },
                    customdata=[
                        [
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
                        "date=%{customdata[0]|%Y-%m-%d}<br>"
                        "state=%{customdata[1]}<br>"
                        "turning=%{customdata[2]}<br>"
                        "persistence=%{customdata[3]}<br>"
                        "RS centered=%{customdata[4]:.4f}<br>"
                        "MOM=%{customdata[5]:.4f}<br>"
                        "Z=%{customdata[6]:.4f}<extra></extra>"
                    ),
                    name=str(sector),
                    legendgroup=str(sector),
                    visible=visible,
                )
            )
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
        },
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.0, "y": 1.12}] if buttons else [],
        margin={"l": 0, "r": 0, "b": 0, "t": 60},
    )
    return fig
