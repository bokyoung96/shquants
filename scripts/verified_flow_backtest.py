from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_TRADE_COLUMNS = {
    "ticker",
    "entry_time",
    "exit_time",
    "entry_price",
    "net_return",
}


@dataclass(frozen=True, slots=True)
class BacktestAudit:
    trades: int
    tickers: int
    first_entry: str
    last_exit: str
    position_slots: int
    end_of_day_max_active_positions: int
    end_of_day_avg_active_positions: float
    same_ticker_overlap_violations: int
    missing_price_points: int
    fixed_notional_final_return: float
    fixed_notional_mdd: float
    rebalanced_final_return: float
    rebalanced_mdd: float
    raw_trade_return_sum: float
    avg_trade_return: float
    hit_rate: float
    profit_factor: float


@dataclass(frozen=True, slots=True)
class FixedSlotSelectionAudit:
    max_positions: int
    slot_weight: float
    input_trades: int
    selected_trades: int
    skipped_trades: int
    max_active_positions: int
    fixed_notional_final_return: float
    fixed_notional_mdd: float
    rebalanced_final_return: float
    rebalanced_mdd: float
    selected_avg_trade_return: float
    selected_hit_rate: float
    selected_profit_factor: float
    skipped_avg_trade_return: float
    skipped_hit_rate: float


def load_trades(path: Path) -> pd.DataFrame:
    trades = pd.read_csv(path, parse_dates=["signal_time", "entry_time", "exit_time"])
    missing = REQUIRED_TRADE_COLUMNS.difference(trades.columns)
    if missing:
        raise ValueError(f"missing trade columns: {sorted(missing)}")
    return trades.sort_values(["entry_time", "ticker"]).reset_index(drop=True)


def load_close_prices(path: Path, trades: pd.DataFrame) -> pd.DataFrame:
    tickers = sorted(trades["ticker"].drop_duplicates())
    start = pd.to_datetime(trades["entry_time"]).min().normalize()
    end = pd.to_datetime(trades["exit_time"]).max().normalize()
    close = pd.read_parquet(path, columns=tickers, engine="pyarrow")
    close.index = pd.to_datetime(close.index).normalize()
    return close.loc[start:end].reindex(columns=tickers)


def position_slots(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 1
    entries = pd.DataFrame({"ts": pd.to_datetime(trades["entry_time"]), "delta": 1})
    exits = pd.DataFrame({"ts": pd.to_datetime(trades["exit_time"]), "delta": -1})
    events = pd.concat([entries, exits], ignore_index=True).sort_values(["ts", "delta"], ascending=[True, False])
    return max(1, int(events["delta"].cumsum().max()))


def same_ticker_overlap_violations(trades: pd.DataFrame) -> int:
    violations = 0
    for _ticker, group in trades.sort_values(["ticker", "entry_time"]).groupby("ticker", sort=True):
        previous_exit = pd.Timestamp.min
        for trade in group.itertuples(index=False):
            if pd.Timestamp(trade.entry_time) <= previous_exit:
                violations += 1
            previous_exit = max(previous_exit, pd.Timestamp(trade.exit_time))
    return violations


def select_fixed_slot_trades(trades: pd.DataFrame, *, max_positions: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if trades.empty:
        selected = trades.copy()
        selected["portfolio_skip_reason"] = pd.NA
        return selected, selected.copy()

    selected_rows: list[pd.Series] = []
    skipped_rows: list[pd.Series] = []
    open_exits: list[pd.Timestamp] = []
    ordered = trades.sort_values(["entry_time", "ticker", "signal_time" if "signal_time" in trades.columns else "exit_time"]).reset_index(drop=True)
    for _, trade in ordered.iterrows():
        entry_time = pd.Timestamp(trade["entry_time"])
        open_exits = [exit_time for exit_time in open_exits if exit_time > entry_time]
        if len(open_exits) >= max_positions:
            skipped = trade.copy()
            skipped["portfolio_skip_reason"] = "max_positions"
            skipped_rows.append(skipped)
            continue
        accepted = trade.copy()
        accepted["portfolio_skip_reason"] = pd.NA
        selected_rows.append(accepted)
        open_exits.append(pd.Timestamp(trade["exit_time"]))

    selected = pd.DataFrame(selected_rows, columns=[*ordered.columns, "portfolio_skip_reason"])
    skipped = pd.DataFrame(skipped_rows, columns=[*ordered.columns, "portfolio_skip_reason"])
    return selected.reset_index(drop=True), skipped.reset_index(drop=True)


def fixed_slot_selection_audit(
    trades: pd.DataFrame,
    close: pd.DataFrame,
    *,
    max_positions: int,
) -> tuple[FixedSlotSelectionAudit, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected, skipped = select_fixed_slot_trades(trades, max_positions=max_positions)
    fixed, _fixed_missing = fixed_notional_mtm_ledger(selected, close, slots=max_positions)
    rebalanced, _rebalanced_missing = rebalanced_mtm_ledger(selected, close, slots=max_positions)
    selected_returns = selected["net_return"] if not selected.empty else pd.Series(dtype=float)
    skipped_returns = skipped["net_return"] if not skipped.empty else pd.Series(dtype=float)
    audit = FixedSlotSelectionAudit(
        max_positions=int(max_positions),
        slot_weight=float(1.0 / max_positions),
        input_trades=int(len(trades)),
        selected_trades=int(len(selected)),
        skipped_trades=int(len(skipped)),
        max_active_positions=position_slots(selected),
        fixed_notional_final_return=float(fixed["equity"].iloc[-1] - 1.0) if not fixed.empty else 0.0,
        fixed_notional_mdd=float(fixed["drawdown"].min()) if not fixed.empty else 0.0,
        rebalanced_final_return=float(rebalanced["equity"].iloc[-1] - 1.0) if not rebalanced.empty else 0.0,
        rebalanced_mdd=float(rebalanced["drawdown"].min()) if not rebalanced.empty else 0.0,
        selected_avg_trade_return=float(selected_returns.mean()) if not selected_returns.empty else 0.0,
        selected_hit_rate=float(selected_returns.gt(0.0).mean()) if not selected_returns.empty else 0.0,
        selected_profit_factor=profit_factor(selected_returns),
        skipped_avg_trade_return=float(skipped_returns.mean()) if not skipped_returns.empty else 0.0,
        skipped_hit_rate=float(skipped_returns.gt(0.0).mean()) if not skipped_returns.empty else 0.0,
    )
    return audit, selected, skipped, fixed, rebalanced


def fixed_notional_mtm_ledger(trades: pd.DataFrame, close: pd.DataFrame, *, slots: int | None = None) -> tuple[pd.DataFrame, int]:
    resolved_slots = slots or position_slots(trades)
    dates = _ledger_index(trades, close)
    contribution = pd.Series(0.0, index=dates)
    active_positions = pd.Series(0, index=dates)
    missing_prices = 0
    close = close.reindex(dates).ffill()

    for trade in trades.itertuples(index=False):
        entry_date = pd.Timestamp(trade.entry_time).normalize()
        exit_date = pd.Timestamp(trade.exit_time).normalize()
        active = dates[(dates >= entry_date) & (dates < exit_date)]
        if len(active):
            if trade.ticker not in close.columns:
                missing_prices += len(active)
            else:
                prices = close.loc[active, trade.ticker]
                missing_prices += int(prices.isna().sum())
                contribution.loc[active] += (prices.ffill() / float(trade.entry_price) - 1.0).fillna(0.0) / resolved_slots
                active_positions.loc[active] += 1
        contribution.loc[dates >= exit_date] += float(trade.net_return) / resolved_slots

    equity = 1.0 + contribution
    ledger = pd.DataFrame(
        {
            "equity": equity,
            "daily_return": equity.pct_change().fillna(equity.iloc[0] - 1.0),
            "drawdown": equity / equity.cummax() - 1.0,
            "active_positions": active_positions,
        }
    )
    return ledger, missing_prices


def rebalanced_mtm_ledger(trades: pd.DataFrame, close: pd.DataFrame, *, slots: int | None = None) -> tuple[pd.DataFrame, int]:
    resolved_slots = slots or position_slots(trades)
    dates = _ledger_index(trades, close)
    close = close.reindex(dates).ffill()
    daily_return = pd.Series(0.0, index=dates)
    active_positions = pd.Series(0, index=dates)
    missing_prices = 0

    for trade in trades.itertuples(index=False):
        entry_date = pd.Timestamp(trade.entry_time).normalize()
        exit_date = pd.Timestamp(trade.exit_time).normalize()
        trade_dates = dates[(dates >= entry_date) & (dates <= exit_date)]
        if len(trade_dates) == 0:
            continue
        previous_price = float(trade.entry_price)
        for date in trade_dates:
            if date < exit_date:
                if trade.ticker not in close.columns or pd.isna(close.at[date, trade.ticker]):
                    missing_prices += 1
                    continue
                current_price = float(close.at[date, trade.ticker])
                leg_return = current_price / previous_price - 1.0
                previous_price = current_price
                active_positions.loc[date] += 1
            else:
                cumulative_before_exit = previous_price / float(trade.entry_price) if previous_price else 1.0
                leg_return = (1.0 + float(trade.net_return)) / cumulative_before_exit - 1.0
            daily_return.loc[date] += leg_return / resolved_slots

    equity = (1.0 + daily_return).cumprod()
    ledger = pd.DataFrame(
        {
            "equity": equity,
            "daily_return": daily_return,
            "drawdown": equity / equity.cummax() - 1.0,
            "active_positions": active_positions,
        }
    )
    return ledger, missing_prices


def audit_backtest(trades: pd.DataFrame, close: pd.DataFrame, *, slots: int | None = None) -> tuple[BacktestAudit, pd.DataFrame, pd.DataFrame]:
    resolved_slots = slots or position_slots(trades)
    fixed, fixed_missing = fixed_notional_mtm_ledger(trades, close, slots=resolved_slots)
    rebalanced, rebalanced_missing = rebalanced_mtm_ledger(trades, close, slots=resolved_slots)
    returns = trades["net_return"] if not trades.empty else pd.Series(dtype=float)
    audit = BacktestAudit(
        trades=int(len(trades)),
        tickers=int(trades["ticker"].nunique()) if not trades.empty else 0,
        first_entry=str(pd.to_datetime(trades["entry_time"]).min()) if not trades.empty else "",
        last_exit=str(pd.to_datetime(trades["exit_time"]).max()) if not trades.empty else "",
        position_slots=int(resolved_slots),
        end_of_day_max_active_positions=int(fixed["active_positions"].max()) if not fixed.empty else 0,
        end_of_day_avg_active_positions=float(fixed["active_positions"].mean()) if not fixed.empty else 0.0,
        same_ticker_overlap_violations=same_ticker_overlap_violations(trades),
        missing_price_points=int(fixed_missing + rebalanced_missing),
        fixed_notional_final_return=float(fixed["equity"].iloc[-1] - 1.0) if not fixed.empty else 0.0,
        fixed_notional_mdd=float(fixed["drawdown"].min()) if not fixed.empty else 0.0,
        rebalanced_final_return=float(rebalanced["equity"].iloc[-1] - 1.0) if not rebalanced.empty else 0.0,
        rebalanced_mdd=float(rebalanced["drawdown"].min()) if not rebalanced.empty else 0.0,
        raw_trade_return_sum=float(returns.sum()) if not returns.empty else 0.0,
        avg_trade_return=float(returns.mean()) if not returns.empty else 0.0,
        hit_rate=float(returns.gt(0.0).mean()) if not returns.empty else 0.0,
        profit_factor=profit_factor(returns),
    )
    return audit, fixed, rebalanced


def profit_factor(returns: pd.Series) -> float:
    gains = float(returns[returns > 0.0].sum())
    losses = float(returns[returns < 0.0].sum())
    if losses == 0.0:
        return float("inf") if gains > 0.0 else 0.0
    return gains / abs(losses)


def _ledger_index(trades: pd.DataFrame, close: pd.DataFrame) -> pd.DatetimeIndex:
    trade_dates = pd.DatetimeIndex(
        pd.concat(
            [
                pd.to_datetime(trades["entry_time"]).dt.normalize(),
                pd.to_datetime(trades["exit_time"]).dt.normalize(),
            ],
            ignore_index=True,
        )
    )
    return pd.DatetimeIndex(close.index.append(trade_dates).unique()).sort_values()


def write_audit_outputs(
    *,
    trades_path: Path,
    close_path: Path,
    output_dir: Path,
    slots: int | None = None,
) -> BacktestAudit:
    trades = load_trades(trades_path)
    close = load_close_prices(close_path, trades)
    audit, fixed, rebalanced = audit_backtest(trades, close, slots=slots)
    output_dir.mkdir(parents=True, exist_ok=True)
    verified_dir = output_dir / "verified"
    fixed20_dir = output_dir / "fixed20"
    distributions_dir = output_dir / "distributions"
    for child_dir in (verified_dir, fixed20_dir, distributions_dir):
        child_dir.mkdir(parents=True, exist_ok=True)

    fixed.to_csv(verified_dir / "fixed_notional_ledger.csv", index_label="date")
    rebalanced.to_csv(verified_dir / "rebalanced_ledger.csv", index_label="date")
    (verified_dir / "backtest_audit.json").write_text(
        json.dumps(asdict(audit), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_verified_report(audit, verified_dir / "backtest_report.md")
    plot_verified_ledgers(fixed, rebalanced, verified_dir / "backtest_curves.png")
    write_trade_return_distribution(trades, distributions_dir / "all_trade_return_distribution.png", title="All Confirmed Episode Trade Return Distribution")
    fixed20_audit, selected20, skipped20, fixed20, rebalanced20 = fixed_slot_selection_audit(trades, close, max_positions=20)
    selected20.to_csv(fixed20_dir / "selected_trades.csv", index=False)
    skipped20.to_csv(fixed20_dir / "skipped_trades.csv", index=False)
    fixed20.to_csv(fixed20_dir / "fixed_notional_ledger.csv", index_label="date")
    rebalanced20.to_csv(fixed20_dir / "rebalanced_ledger.csv", index_label="date")
    (fixed20_dir / "audit.json").write_text(
        json.dumps(asdict(fixed20_audit), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_fixed_slot_report(fixed20_audit, fixed20_dir / "report.md")
    plot_verified_ledgers(fixed20, rebalanced20, fixed20_dir / "curves.png")
    write_trade_return_distribution(selected20, distributions_dir / "fixed20_selected_return_distribution.png", title="Fixed 20 Selected Trade Return Distribution")
    if not skipped20.empty:
        write_trade_return_distribution(skipped20, distributions_dir / "fixed20_skipped_return_distribution.png", title="Fixed 20 Skipped Trade Return Distribution")
    return audit


def write_fixed_slot_report(audit: FixedSlotSelectionAudit, path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Fixed 20 Slot Confirmed Episode Backtest",
                "",
                "Portfolio rules:",
                f"- max_positions: {audit.max_positions}",
                f"- slot_weight: {audit.slot_weight * 100.0:.2f}%",
                "- entries are accepted in confirmed entry-time order",
                "- new entries are skipped while all slots are occupied",
                "- empty slots remain cash",
                "",
                "Audit results:",
                f"- input_trades: {audit.input_trades:,}",
                f"- selected_trades: {audit.selected_trades:,}",
                f"- skipped_trades: {audit.skipped_trades:,}",
                f"- max_active_positions: {audit.max_active_positions}",
                f"- fixed_notional_final_return: {audit.fixed_notional_final_return * 100.0:.2f}%",
                f"- fixed_notional_mdd: {audit.fixed_notional_mdd * 100.0:.2f}%",
                f"- rebalanced_final_return: {audit.rebalanced_final_return * 100.0:.2f}%",
                f"- rebalanced_mdd: {audit.rebalanced_mdd * 100.0:.2f}%",
                f"- selected_avg_trade_return_bps: {audit.selected_avg_trade_return * 10_000.0:.2f}",
                f"- selected_hit_rate: {audit.selected_hit_rate * 100.0:.2f}%",
                f"- selected_profit_factor: {audit.selected_profit_factor:.4f}",
                f"- skipped_avg_trade_return_bps: {audit.skipped_avg_trade_return * 10_000.0:.2f}",
                f"- skipped_hit_rate: {audit.skipped_hit_rate * 100.0:.2f}%",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_trade_return_distribution(trades: pd.DataFrame, path: Path, *, title: str) -> None:
    returns = trades["net_return"].dropna().astype(float)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), dpi=160, facecolor="#fbfaf7")
    fig.patch.set_facecolor("#fbfaf7")
    if returns.empty:
        for ax in axes.ravel():
            ax.axis("off")
        axes[0, 0].text(0.5, 0.5, "No trades", ha="center", va="center", fontsize=14)
    else:
        bps = returns * 10_000.0
        q01, q05, q25, q50, q75, q95, q99 = bps.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        central_low = max(-350.0, min(-100.0, q01))
        central_high = min(350.0, max(100.0, q95))
        central = bps[(bps >= central_low) & (bps <= central_high)]

        axes[0, 0].hist(central, bins=80, color="#365f8c", edgecolor="#fbfaf7", linewidth=0.7, alpha=0.92)
        axes[0, 0].axvspan(q25, q75, color="#f2b84b", alpha=0.18, label="IQR")
        axes[0, 0].axvline(0.0, color="#242424", linewidth=1.1)
        axes[0, 0].axvline(bps.mean(), color="#1b8f6a", linewidth=2.1, label=f"mean {bps.mean():.1f} bps")
        axes[0, 0].axvline(q50, color="#d55e00", linewidth=2.1, label=f"median {q50:.1f} bps")
        axes[0, 0].set_title("Central return shape", loc="left", fontweight="bold")
        axes[0, 0].set_xlabel(f"Trade return central window ({central_low:.0f} to {central_high:.0f} bps)")
        axes[0, 0].set_ylabel("Trades")
        axes[0, 0].legend(frameon=False, ncols=3, fontsize=9)

        losses = bps[bps <= 0.0]
        if losses.empty:
            axes[0, 1].text(0.5, 0.5, "No loss / flat trades", ha="center", va="center", transform=axes[0, 1].transAxes)
        else:
            loss_floor = losses.quantile(0.01)
            loss_view = losses.clip(lower=loss_floor)
            axes[0, 1].hist(loss_view, bins=min(55, max(12, len(loss_view) // 4)), color="#b44b4b", edgecolor="#fbfaf7", linewidth=0.7, alpha=0.9)
            axes[0, 1].axvline(losses.median(), color="#612525", linewidth=2.0, label=f"loss median {losses.median():.1f} bps")
            axes[0, 1].axvline(losses.quantile(0.10), color="#e0a65b", linewidth=2.0, label=f"loss p10 {losses.quantile(0.10):.1f} bps")
            axes[0, 1].axvline(0.0, color="#242424", linewidth=1.0)
            axes[0, 1].legend(frameon=False, fontsize=9)
        axes[0, 1].set_title("Loss cluster", loc="left", fontweight="bold")
        axes[0, 1].set_xlabel("Loss / flat return (bps)")
        axes[0, 1].set_ylabel("Trades")

        wins = bps[bps > 0.0].sort_values(ascending=False)
        if wins.empty:
            axes[1, 0].text(0.5, 0.5, "No winning trades", ha="center", va="center", transform=axes[1, 0].transAxes)
        else:
            top_wins = wins.head(min(30, len(wins))).iloc[::-1]
            colors = np.where(top_wins >= q99, "#1b8f6a", "#4f86c6")
            labels = [f"#{rank}" for rank in range(len(top_wins), 0, -1)]
            axes[1, 0].barh(labels, top_wins.values, color=colors, alpha=0.9)
            axes[1, 0].axvline(q95, color="#253858", linewidth=1.5, linestyle="--", label=f"p95 {q95:.1f} bps")
            axes[1, 0].legend(frameon=False, fontsize=9)
        axes[1, 0].set_title("Right-tail winners", loc="left", fontweight="bold")
        axes[1, 0].set_xlabel("Winning trade return (bps)")
        axes[1, 0].set_ylabel("Top winners")

        if "exit_reason" in trades.columns:
            by_reason = (
                trades.assign(return_bps=trades["net_return"].astype(float) * 10_000.0)
                .dropna(subset=["return_bps"])
                .groupby("exit_reason", dropna=False)
                .agg(trades=("return_bps", "size"), median_bps=("return_bps", "median"), mean_bps=("return_bps", "mean"))
                .sort_values("trades", ascending=True)
            )
            min_reason_trades = max(3, int(len(bps) * 0.005)) if len(bps) >= 100 else 1
            display_reason = by_reason[by_reason["trades"] >= min_reason_trades].tail(8)
            if display_reason.empty:
                display_reason = by_reason.tail(8)
            y = np.arange(len(display_reason))
            bar_colors = np.where(display_reason["median_bps"] >= 0.0, "#1b8f6a", "#b44b4b")
            axes[1, 1].barh(y, display_reason["median_bps"], color=bar_colors, alpha=0.9)
            axes[1, 1].scatter(display_reason["mean_bps"], y, color="#242424", s=28, zorder=3, label="mean")
            axes[1, 1].set_yticks(y, [f"{reason} ({count})" for reason, count in zip(display_reason.index, display_reason["trades"])])
            axes[1, 1].axvline(0.0, color="#242424", linewidth=1.0)
            axes[1, 1].legend(frameon=False, fontsize=9)
        else:
            axes[1, 1].text(0.5, 0.5, "No exit_reason column", ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Exit reason profile", loc="left", fontweight="bold")
        axes[1, 1].set_xlabel("Median return by exit reason (bps); dot = mean")

        stats = (
            f"trades {len(bps):,}   mean {bps.mean():.1f} bps   median {q50:.1f} bps   "
            f"hit rate {bps.gt(0.0).mean() * 100.0:.1f}%   p5 {q05:.1f} bps   p95 {q95:.1f} bps   max {bps.max():.1f} bps"
        )
        fig.text(0.01, 0.935, stats, ha="left", va="top", fontsize=10.5, color="#3a3a3a")

    fig.suptitle(title, fontsize=16, fontweight="bold", x=0.01, ha="left")
    for ax in axes.ravel():
        ax.set_facecolor("#fbfaf7")
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#d7d0c6")
        ax.tick_params(colors="#333333")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_verified_report(audit: BacktestAudit, path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Verified Confirmed Episode Backtest",
                "",
                "Accounting assumptions:",
                "- Uses the generated confirmed episode trade log as the source of truth.",
                "- Uses daily close mark-to-market while positions are open.",
                "- Uses trade-level net_return on each exit date, so transaction costs already embedded in net_return are preserved.",
                "- `fixed_notional` allocates initial capital / position_slots to each trade and does not reinvest.",
                "- `rebalanced` applies equal slot exposure to daily position returns and compounds daily portfolio returns.",
                "",
                "Audit results:",
                f"- trades: {audit.trades:,}",
                f"- tickers: {audit.tickers:,}",
                f"- period: {audit.first_entry} to {audit.last_exit}",
                f"- position_slots: {audit.position_slots}",
                f"- end_of_day_max_active_positions: {audit.end_of_day_max_active_positions}",
                f"- end_of_day_avg_active_positions: {audit.end_of_day_avg_active_positions:.2f}",
                f"- same_ticker_overlap_violations: {audit.same_ticker_overlap_violations}",
                f"- missing_price_points: {audit.missing_price_points}",
                f"- fixed_notional_final_return: {audit.fixed_notional_final_return * 100.0:.2f}%",
                f"- fixed_notional_mdd: {audit.fixed_notional_mdd * 100.0:.2f}%",
                f"- rebalanced_final_return: {audit.rebalanced_final_return * 100.0:.2f}%",
                f"- rebalanced_mdd: {audit.rebalanced_mdd * 100.0:.2f}%",
                f"- raw_trade_return_sum: {audit.raw_trade_return_sum:.6f}",
                f"- avg_trade_return_bps: {audit.avg_trade_return * 10_000.0:.2f}",
                f"- hit_rate: {audit.hit_rate * 100.0:.2f}%",
                f"- profit_factor: {audit.profit_factor:.4f}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def plot_verified_ledgers(fixed: pd.DataFrame, rebalanced: pd.DataFrame, path: Path) -> None:
    curves = pd.DataFrame(
        {
            "fixed_notional": fixed["equity"],
            "rebalanced": rebalanced["equity"],
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(18, 12.5), dpi=160, facecolor="#fbfaf7", gridspec_kw={"height_ratios": [1.18, 1.0, 1.0]})
    fig.patch.set_facecolor("#fbfaf7")
    _plot_verified_cumulative(axes[0, 0], curves)
    _plot_verified_snapshot(axes[0, 1], fixed, rebalanced)
    _plot_verified_drawdown(axes[1, 0], curves)
    _plot_verified_active_positions(axes[1, 1], fixed)
    _plot_verified_monthly(axes[2, 0], curves)
    _plot_verified_yearly(axes[2, 1], curves)
    fig.suptitle("Verified MTM strategy dashboard", fontsize=18, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955), h_pad=2.0, w_pad=2.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_verified_cumulative(ax: plt.Axes, curves: pd.DataFrame) -> None:
    returns = (curves - 1.0) * 100.0
    returns.plot(ax=ax, linewidth=2.4, color=_verified_colors(list(returns.columns)))
    ax.fill_between(returns.index, returns["fixed_notional"].to_numpy(dtype=float), 0.0, color="#365f8c", alpha=0.08)
    last = returns.iloc[-1]
    for column, value in last.items():
        ax.text(returns.index[-1], value, f" {column} {value:.1f}%", va="center", fontsize=9.5, color="#333333")
    ax.axhline(0.0, color="#242424", linewidth=0.8, alpha=0.7)
    ax.set_title("Cumulative return path", loc="left", fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("")
    _style_verified_axis(ax)


def _plot_verified_drawdown(ax: plt.Axes, curves: pd.DataFrame) -> None:
    drawdowns = curves.div(curves.cummax()).sub(1.0) * 100.0
    drawdowns.plot(ax=ax, linewidth=1.8, color=_verified_colors(list(drawdowns.columns)))
    focus = drawdowns["fixed_notional"] if "fixed_notional" in drawdowns.columns else drawdowns.iloc[:, 0]
    ax.fill_between(focus.index, focus.to_numpy(dtype=float), 0.0, color="#9d1730", alpha=0.12)
    mdd_date = focus.idxmin()
    mdd = float(focus.loc[mdd_date])
    ax.scatter(mdd_date, mdd, color="#9d1730", s=55, zorder=5)
    ax.annotate(
        f"MDD {mdd:.1f}%",
        xy=(mdd_date, mdd),
        xytext=(10, -22),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d0d0d0", "alpha": 0.95},
    )
    ax.axhline(0.0, color="#242424", linewidth=0.8, alpha=0.7)
    ax.set_title("Drawdown pressure", loc="left", fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("")
    _style_verified_axis(ax)


def _plot_verified_active_positions(ax: plt.Axes, fixed: pd.DataFrame) -> None:
    active = fixed["active_positions"].astype(float)
    ax.fill_between(active.index, active.to_numpy(dtype=float), step="mid", color="#4f86c6", alpha=0.20)
    ax.plot(active.index, active, color="#365f8c", linewidth=1.8)
    ax.axhline(active.mean(), color="#d55e00", linewidth=1.6, linestyle="--", label=f"avg {active.mean():.1f}")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Exposure / active positions", loc="left", fontweight="bold")
    ax.set_ylabel("Positions")
    ax.set_xlabel("")
    _style_verified_axis(ax)


def _plot_verified_monthly(ax: plt.Axes, curves: pd.DataFrame) -> None:
    monthly = _verified_period_returns(curves, "M")
    if not monthly.empty:
        values = monthly["fixed_notional"] * 100.0
        colors = ["#1b8f6a" if value >= 0.0 else "#b44b4b" for value in values]
        ax.bar(range(len(values)), values.to_numpy(dtype=float), color=colors, alpha=0.88, width=0.86)
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
    ax.axhline(0.0, color="#242424", linewidth=0.8, alpha=0.7)
    ax.set_title("Monthly return tape", loc="left", fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("")
    _style_verified_axis(ax)


def _plot_verified_yearly(ax: plt.Axes, curves: pd.DataFrame) -> None:
    yearly = _verified_period_returns(curves, "Y")
    if not yearly.empty:
        columns = list(yearly.columns)
        (yearly[columns] * 100.0).plot(kind="bar", ax=ax, width=0.76, color=_verified_colors(columns), alpha=0.92)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", fontsize=8, padding=2)
    ax.axhline(0.0, color="#242424", linewidth=0.8, alpha=0.7)
    ax.set_title("Yearly scorecard", loc="left", fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    _style_verified_axis(ax)


def _plot_verified_snapshot(ax: plt.Axes, fixed: pd.DataFrame, rebalanced: pd.DataFrame) -> None:
    ax.set_title("Portfolio snapshot", loc="left", fontweight="bold")
    ax.set_axis_off()
    fixed_return = fixed["equity"].iloc[-1] - 1.0
    rebalanced_return = rebalanced["equity"].iloc[-1] - 1.0
    fixed_mdd = fixed["drawdown"].min()
    rebalanced_mdd = rebalanced["drawdown"].min()
    active = fixed["active_positions"]
    ax.text(0.02, 0.72, "fixed-notional return", transform=ax.transAxes, fontsize=10, color="#666666", ha="left")
    ax.text(
        0.02,
        0.55,
        f"{fixed_return * 100.0:.1f}%",
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        color="#1b8f6a" if fixed_return >= 0.0 else "#b44b4b",
        ha="left",
    )
    lines = [
        f"period: {fixed.index.min():%Y-%m-%d} to {fixed.index.max():%Y-%m-%d}",
        f"fixed MDD: {fixed_mdd * 100.0:.2f}%",
        f"rebalanced return: {rebalanced_return * 100.0:.1f}%",
        f"rebalanced MDD: {rebalanced_mdd * 100.0:.2f}%",
        f"max active positions: {int(active.max())}",
        f"avg active positions: {active.mean():.2f}",
    ]
    ax.text(0.45, 0.92, "\n".join(lines), ha="left", va="top", fontsize=10.5, linespacing=1.35, transform=ax.transAxes)


def _verified_period_returns(curves: pd.DataFrame, frequency: str) -> pd.DataFrame:
    daily = curves.pct_change().fillna(curves.iloc[0] - 1.0)
    periods = pd.DatetimeIndex(curves.index).to_period(frequency).astype(str)
    return daily.groupby(periods).apply(lambda frame: (1.0 + frame).prod() - 1.0)


def _verified_colors(columns: list[str]) -> list[str]:
    palette = {"fixed_notional": "#365f8c", "rebalanced": "#1b8f6a"}
    return [palette.get(column, "#777777") for column in columns]


def _style_verified_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#fbfaf7")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d7d0c6")
    ax.tick_params(colors="#333333")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write verified MTM portfolio ledgers for a confirmed episode trade log.")
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--close", type=Path, default=Path("parquet/qw_adj_c.parquet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slots", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = write_audit_outputs(trades_path=args.trades, close_path=args.close, output_dir=args.output_dir, slots=args.slots)
    print(json.dumps(asdict(audit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
