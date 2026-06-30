from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_performance_outputs(
    intraday: pd.DataFrame,
    overnight: pd.DataFrame,
    output: Path,
    title: str = "Scheme Performance",
    *,
    mark_to_market: pd.DataFrame | None = None,
) -> None:
    curves = equity_curves(intraday, overnight, mark_to_market=mark_to_market)
    curves.to_csv(output / "equity_curves.csv", index_label="date")
    curves.to_csv(output / "return_curves.csv", index_label="date")
    plot_performance_subplots(curves, intraday, overnight, output / "performance_subplots.png", title)


def equity_curves(intraday: pd.DataFrame, overnight: pd.DataFrame, *, mark_to_market: pd.DataFrame | None = None) -> pd.DataFrame:
    position_slots = _position_slots(intraday, overnight)
    if _can_mark_to_market(mark_to_market):
        returns = _mark_to_market_returns(
            {"intraday": intraday, "overnight": overnight},
            mark_to_market=mark_to_market,
            position_slots=position_slots,
        )
    else:
        returns = pd.concat(
            [
                _daily_returns(intraday, "intraday", position_slots),
                _daily_returns(overnight, "overnight", position_slots),
            ],
            axis=1,
        ).fillna(0.0)
    returns["combined"] = returns.sum(axis=1)
    if returns.empty:
        return pd.DataFrame(columns=["intraday", "overnight", "combined"])
    return (1.0 + returns).cumprod()


def plot_performance_subplots(
    curves: pd.DataFrame,
    intraday: pd.DataFrame,
    overnight: pd.DataFrame,
    path: Path,
    title: str,
) -> None:
    position_slots = _position_slots(intraday, overnight)
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(18, 12.5),
        dpi=160,
        facecolor="#fbfaf7",
        gridspec_kw={"height_ratios": [1.18, 1.0, 1.0]},
    )
    fig.patch.set_facecolor("#fbfaf7")
    _plot_cumulative_return(axes[0, 0], curves)
    _plot_summary_panel(axes[0, 1], intraday, overnight, curves, position_slots)
    _plot_drawdown(axes[1, 0], curves)
    _plot_active_positions(axes[1, 1], intraday, overnight, curves)
    _plot_monthly_returns(axes[2, 0], curves)
    _plot_yearly_returns(axes[2, 1], curves)
    fig.suptitle(title, fontsize=18, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955), h_pad=2.0, w_pad=2.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _daily_returns(frame: pd.DataFrame, name: str, position_slots: int) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float, name=name)
    exits = pd.to_datetime(frame["exit_time"]).dt.normalize()
    return frame.groupby(exits)["net_return"].sum().div(position_slots).sort_index().rename(name)


def _can_mark_to_market(mark_to_market: pd.DataFrame | None) -> bool:
    return mark_to_market is not None and not mark_to_market.empty and {"date", "ticker", "close"}.issubset(mark_to_market.columns)


def _mark_to_market_returns(
    frames: dict[str, pd.DataFrame],
    *,
    mark_to_market: pd.DataFrame,
    position_slots: int,
) -> pd.DataFrame:
    all_trades = pd.concat([frame for frame in frames.values() if not frame.empty], ignore_index=True)
    if all_trades.empty:
        return pd.DataFrame(columns=list(frames))
    prices = mark_to_market.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    price_matrix = prices.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    start = pd.to_datetime(all_trades["entry_time"]).min().normalize()
    end = pd.to_datetime(all_trades["exit_time"]).max().normalize()
    price_matrix = price_matrix.loc[start:end]
    trade_dates = pd.DatetimeIndex(
        pd.concat(
            [
                pd.to_datetime(all_trades["entry_time"]).dt.normalize(),
                pd.to_datetime(all_trades["exit_time"]).dt.normalize(),
            ],
            ignore_index=True,
        )
    )
    curve_index = pd.DatetimeIndex(price_matrix.index.append(trade_dates).unique()).sort_values()
    price_matrix = price_matrix.reindex(curve_index).ffill()
    returns = pd.DataFrame(0.0, index=price_matrix.index, columns=list(frames))
    for name, frame in frames.items():
        if frame.empty:
            continue
        for trade in frame.itertuples(index=False):
            _add_trade_mark_to_market_returns(returns[name], price_matrix, trade, position_slots)
    return returns


def _add_trade_mark_to_market_returns(
    daily_returns: pd.Series,
    price_matrix: pd.DataFrame,
    trade: object,
    position_slots: int,
) -> None:
    entry_date = pd.Timestamp(trade.entry_time).normalize()
    exit_date = pd.Timestamp(trade.exit_time).normalize()
    dates = price_matrix.index[(price_matrix.index >= entry_date) & (price_matrix.index <= exit_date)]
    if dates.empty:
        return
    entry_price = float(trade.entry_price)
    previous_price = entry_price
    for date in dates:
        if date < exit_date:
            if trade.ticker not in price_matrix.columns or pd.isna(price_matrix.at[date, trade.ticker]):
                continue
            current_price = float(price_matrix.at[date, trade.ticker])
            leg_return = current_price / previous_price - 1.0
            previous_price = current_price
        else:
            cumulative_before_exit = previous_price / entry_price if previous_price else 1.0
            leg_return = (1.0 + float(trade.net_return)) / cumulative_before_exit - 1.0
        daily_returns.loc[date] += leg_return / position_slots


def _position_slots(*frames: pd.DataFrame) -> int:
    trades = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if trades.empty or "entry_time" not in trades.columns:
        return 1
    entries = pd.DataFrame({"ts": pd.to_datetime(trades["entry_time"]), "delta": 1})
    exits = pd.DataFrame({"ts": pd.to_datetime(trades["exit_time"]), "delta": -1})
    events = pd.concat([entries, exits], ignore_index=True).sort_values(["ts", "delta"], ascending=[True, False])
    concurrent = events["delta"].cumsum()
    return max(1, int(concurrent.max()))


def _plot_cumulative_return(ax: plt.Axes, curves: pd.DataFrame) -> None:
    if not curves.empty:
        plotted = _curve_columns_to_plot(curves)
        returns = (curves[plotted] - 1.0) * 100.0
        returns.plot(ax=ax, linewidth=2.4, color=_colors(plotted))
        if "combined" in returns.columns:
            ax.fill_between(returns.index, returns["combined"].to_numpy(dtype=float), 0.0, color="#365f8c", alpha=0.08)
        last = (curves[plotted].iloc[-1] - 1.0) * 100.0
        for column, value in last.items():
            ax.text(curves.index[-1], value, f" {column} {value:.1f}%", va="center", fontsize=9.5, color="#333333")
    ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.7)
    ax.set_title("Cumulative return path", loc="left", fontweight="bold")
    ax.set_ylabel("Portfolio return (%)")
    ax.set_xlabel("")
    _style_axis(ax)


def _plot_drawdown(ax: plt.Axes, curves: pd.DataFrame) -> None:
    if not curves.empty:
        plotted = _curve_columns_to_plot(curves)
        drawdown = curves[plotted].div(curves[plotted].cummax()).sub(1.0) * 100.0
        drawdown.plot(ax=ax, linewidth=1.8, color=_colors(plotted))
        combined = drawdown["combined"] if "combined" in drawdown.columns else drawdown.iloc[:, 0]
        ax.fill_between(combined.index, combined.to_numpy(dtype=float), 0.0, color="#9d1730", alpha=0.12)
        mdd_date = combined.idxmin()
        mdd = float(combined.loc[mdd_date])
        ax.scatter(mdd_date, mdd, color="#9d1730", s=55, zorder=5)
        ax.annotate(
            f"MDD {mdd:.1f}%",
            xy=(mdd_date, mdd),
            xytext=(10, -22),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#555555"},
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d0d0d0", "alpha": 0.95},
        )
    ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.7)
    ax.set_title("Drawdown pressure", loc="left", fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("")
    _style_axis(ax)


def _plot_yearly_returns(ax: plt.Axes, curves: pd.DataFrame) -> None:
    yearly = _period_returns(curves, "Y")
    if not yearly.empty:
        if "combined" not in yearly.columns:
            yearly["combined"] = yearly.sum(axis=1)
        columns = _return_columns_to_plot(yearly)
        (yearly[columns] * 100.0).plot(kind="bar", ax=ax, width=0.76, color=_colors(columns), alpha=0.92)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", fontsize=8, padding=2)
    ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.7)
    ax.set_title("Yearly scorecard", loc="left", fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    _style_axis(ax)


def _plot_monthly_returns(ax: plt.Axes, curves: pd.DataFrame) -> None:
    monthly = _period_returns(curves, "M")
    if not monthly.empty:
        if "combined" not in monthly.columns:
            monthly["combined"] = monthly.sum(axis=1)
        values = monthly["combined"] * 100.0
        colors = ["#2ca25f" if value >= 0.0 else "#c44e52" for value in values]
        ax.bar(range(len(values)), values.to_numpy(dtype=float), color=colors, alpha=0.86, width=0.86)
        tick_index = list(range(0, len(values), max(1, len(values) // 12)))
        ax.set_xticks(tick_index)
        ax.set_xticklabels([values.index[index] for index in tick_index], rotation=35, ha="right")
        best = values.idxmax()
        worst = values.idxmin()
        ax.text(
            0.98,
            0.92,
            f"best {best}: {values.loc[best]:.1f}%\nworst {worst}: {values.loc[worst]:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#d0d0d0", "alpha": 0.95},
        )
    ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.7)
    ax.set_title("Monthly return tape", loc="left", fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("")
    _style_axis(ax)


def _plot_active_positions(ax: plt.Axes, intraday: pd.DataFrame, overnight: pd.DataFrame, curves: pd.DataFrame) -> None:
    trades = pd.concat([frame for frame in (intraday, overnight) if not frame.empty], ignore_index=True)
    if not curves.empty and not trades.empty and {"entry_time", "exit_time"}.issubset(trades.columns):
        index = pd.DatetimeIndex(curves.index)
        active = pd.Series(0, index=index)
        for trade in trades.itertuples(index=False):
            entry = pd.Timestamp(trade.entry_time).normalize()
            exit_ = pd.Timestamp(trade.exit_time).normalize()
            active.loc[(index >= entry) & (index <= exit_)] += 1
        ax.fill_between(active.index, active.to_numpy(dtype=float), step="mid", color="#4f86c6", alpha=0.20)
        ax.plot(active.index, active, color="#365f8c", linewidth=1.8)
        ax.axhline(active.mean(), color="#d55e00", linewidth=1.6, linestyle="--", label=f"avg {active.mean():.1f}")
        ax.legend(frameon=False, loc="upper right")
    ax.set_title("Exposure / active positions", loc="left", fontweight="bold")
    ax.set_ylabel("Positions")
    ax.set_xlabel("")
    _style_axis(ax)


def _plot_summary_panel(ax: plt.Axes, intraday: pd.DataFrame, overnight: pd.DataFrame, curves: pd.DataFrame, position_slots: int) -> None:
    ax.set_title("Strategy snapshot", loc="left", fontweight="bold")
    ax.set_axis_off()
    trades = pd.concat([frame for frame in (intraday, overnight) if not frame.empty], ignore_index=True)
    if trades.empty:
        text = "No trades"
    else:
        returns = trades["net_return"]
        wins = returns[returns > 0.0]
        losses = returns[returns <= 0.0]
        hit_rate = returns.gt(0.0).mean()
        avg_net_bps = returns.mean() * 10_000.0
        net_sum = returns.sum()
        final_return = (curves["combined"].iloc[-1] - 1.0) if not curves.empty and "combined" in curves.columns else 0.0
        avg_win_bps = wins.mean() * 10_000.0 if not wins.empty else 0.0
        avg_loss_bps = losses.mean() * 10_000.0 if not losses.empty else 0.0
        payoff = wins.mean() / abs(losses.mean()) if not wins.empty and not losses.empty else 0.0
        profit_factor = wins.sum() / abs(losses.sum()) if not wins.empty and losses.sum() < 0.0 else 0.0
        exits = pd.to_datetime(trades["exit_time"])
        text = "\n".join(
            [
                "Summary",
                f"period: {exits.min():%Y-%m-%d} to {exits.max():%Y-%m-%d}",
                f"trades: {len(trades):,}",
                f"position slots: {position_slots}",
                f"portfolio return: {final_return * 100.0:.1f}%",
                f"raw trade return sum: {net_sum:.2f}",
                f"avg net: {avg_net_bps:.1f} bps",
                f"hit rate: {hit_rate * 100.0:.1f}%",
                "",
                "Payoff profile",
                f"avg win: {avg_win_bps:.1f} bps",
                f"avg loss: {avg_loss_bps:.1f} bps",
                f"payoff ratio: {payoff:.2f}x",
                f"profit factor: {profit_factor:.2f}x",
            ]
        )
    ax.text(0.02, 0.72, "final return", transform=ax.transAxes, fontsize=10, color="#666666", ha="left")
    if not curves.empty and "combined" in curves.columns:
        final_return = (curves["combined"].iloc[-1] - 1.0) * 100.0
        ax.text(0.02, 0.56, f"{final_return:.1f}%", transform=ax.transAxes, fontsize=30, fontweight="bold", color="#1b8f6a" if final_return >= 0 else "#b44b4b", ha="left")
    ax.text(
        0.45,
        0.92,
        text,
        ha="left",
        va="top",
        fontsize=10.5,
        linespacing=1.35,
        transform=ax.transAxes,
    )


def _period_returns(curves: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if curves.empty:
        return pd.DataFrame(columns=curves.columns)
    daily = curves.pct_change().fillna(curves.iloc[0] - 1.0)
    periods = pd.DatetimeIndex(curves.index).to_period(frequency).astype(str)
    return daily.groupby(periods).apply(lambda frame: (1.0 + frame).prod() - 1.0)


def _curve_columns_to_plot(curves: pd.DataFrame) -> list[str]:
    columns = ["combined"]
    combined = curves["combined"] if "combined" in curves.columns else None
    for column in ("intraday", "overnight"):
        if column not in curves.columns or curves[column].eq(1.0).all():
            continue
        if combined is not None and curves[column].equals(combined):
            continue
        columns.append(column)
    return [column for column in dict.fromkeys(columns) if column in curves.columns]


def _return_columns_to_plot(frame: pd.DataFrame) -> list[str]:
    columns = ["combined"]
    combined = frame["combined"] if "combined" in frame.columns else None
    for column in ("intraday", "overnight"):
        if column not in frame.columns or frame[column].eq(0.0).all():
            continue
        if combined is not None and frame[column].equals(combined):
            continue
        columns.append(column)
    return [column for column in dict.fromkeys(columns) if column in frame.columns]


def _colors(columns: list[str]) -> list[str]:
    palette = {"combined": "#1d3557", "intraday": "#4c78a8", "overnight": "#f58518"}
    return [palette.get(column, "#777777") for column in columns]


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#fbfaf7")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d7d0c6")
    ax.tick_params(colors="#333333")
