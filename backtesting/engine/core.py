from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from backtesting.engine.result import BacktestResult
from backtesting.execution.costs import CostModel
from backtesting.execution.fill import fill_prices
from backtesting.execution.schedule import RebalanceSchedule


@dataclass(slots=True)
class BacktestEngine:
    cost: CostModel

    def run(
        self,
        close: pd.DataFrame,
        weights: pd.DataFrame,
        capital: float,
        open: pd.DataFrame | None = None,
        tradable: pd.DataFrame | None = None,
        schedule: pd.Series | RebalanceSchedule | None = None,
        fill_mode: str = "next_open",
        allow_fractional: bool = True,
        show_progress: bool = False,
    ) -> BacktestResult:
        close = close.astype(float)
        weights = weights.reindex_like(close).fillna(0.0).astype(float)
        tradable = self._tradable(tradable=tradable, close=close)
        schedule = self._schedule(schedule=schedule, close=close)

        equity = pd.Series(0.0, index=close.index, dtype=float)
        qty = pd.DataFrame(0.0, index=close.index, columns=close.columns, dtype=float)
        turnover = pd.Series(0.0, index=close.index, dtype=float)

        cash = float(capital)
        current_qty = pd.Series(0.0, index=close.columns, dtype=float)
        dates = close.index

        if fill_mode == "next_open":
            open_ = None if open is None else open.reindex_like(close).astype(float)
            exec_prices = fill_prices(close=close, open_=open_, fill_mode="next_open")
            equity.iloc[0] = capital
            qty.iloc[0] = current_qty

            for ts in tqdm(dates[1:], desc="backtest", disable=not show_progress):
                prev_ts = dates[dates.get_loc(ts) - 1]
                if schedule.loc[prev_ts]:
                    cash, current_qty, turn = self._rebalance(
                        cash=cash,
                        current_qty=current_qty,
                        fill_price=exec_prices.loc[prev_ts],
                        target_weight=weights.loc[prev_ts],
                        tradable=tradable.loc[ts],
                        allow_fractional=allow_fractional,
                    )
                else:
                    turn = 0.0
                equity.loc[ts] = cash + current_qty.mul(close.loc[ts]).sum()
                qty.loc[ts] = current_qty
                turnover.loc[ts] = turn
        elif fill_mode == "close":
            first_ts = dates[0]
            if schedule.loc[first_ts]:
                cash, current_qty, turn = self._rebalance(
                    cash=cash,
                    current_qty=current_qty,
                    fill_price=close.loc[first_ts],
                    target_weight=weights.loc[first_ts],
                    tradable=tradable.loc[first_ts],
                    allow_fractional=allow_fractional,
                )
            else:
                turn = 0.0
            equity.loc[first_ts] = cash + current_qty.mul(close.loc[first_ts]).sum()
            qty.loc[first_ts] = current_qty
            turnover.loc[first_ts] = turn

            for ts in tqdm(dates[1:], desc="backtest", disable=not show_progress):
                if schedule.loc[ts]:
                    cash, current_qty, turn = self._rebalance(
                        cash=cash,
                        current_qty=current_qty,
                        fill_price=close.loc[ts],
                        target_weight=weights.loc[ts],
                        tradable=tradable.loc[ts],
                        allow_fractional=allow_fractional,
                    )
                else:
                    turn = 0.0
                equity.loc[ts] = cash + current_qty.mul(close.loc[ts]).sum()
                qty.loc[ts] = current_qty
                turnover.loc[ts] = turn
        else:
            fill_prices(close=close, open_=open, fill_mode=fill_mode)

        returns = equity.pct_change().fillna(0.0)
        return BacktestResult(
            equity=equity,
            returns=returns,
            weights=weights,
            qty=qty,
            turnover=turnover,
        )

    def _rebalance(
        self,
        cash: float,
        current_qty: pd.Series,
        fill_price: pd.Series,
        target_weight: pd.Series,
        tradable: pd.Series,
        allow_fractional: bool,
    ) -> tuple[float, pd.Series, float]:
        tradable = tradable.reindex(fill_price.index).fillna(False).astype(bool)
        safe_price = fill_price.where(fill_price.ne(0.0))
        can_trade = tradable & safe_price.notna()
        nav = cash + current_qty.mul(fill_price.fillna(0.0)).sum()

        target_qty = target_weight.mul(nav).div(safe_price).fillna(0.0)
        target_qty = target_qty.where(can_trade, current_qty)
        target_qty = self._normalize_quantity(target_qty=target_qty, allow_fractional=allow_fractional)

        raw_delta = target_qty.sub(current_qty).fillna(0.0)
        sell_delta = raw_delta.where(raw_delta.lt(0.0), 0.0)
        buy_delta = raw_delta.where(raw_delta.gt(0.0), 0.0)

        next_cash = cash
        for symbol, qty_delta in sell_delta.items():
            if qty_delta == 0.0:
                continue

            price = float(fill_price.loc[symbol])
            gross = abs(price * float(qty_delta))
            trade_cost = self.cost.calc(price=price, qty=float(qty_delta), side="sell")
            next_cash += gross - trade_cost.total

        buy_delta = self._cap_buy_delta(
            buy_delta=buy_delta,
            fill_price=fill_price,
            cash=next_cash,
            allow_fractional=allow_fractional,
        )
        delta = sell_delta.add(buy_delta, fill_value=0.0)
        trade_value = delta.mul(fill_price.fillna(0.0)).abs()

        for symbol, qty_delta in buy_delta.items():
            if qty_delta == 0.0:
                continue

            price = float(fill_price.loc[symbol])
            gross = abs(price * float(qty_delta))
            trade_cost = self.cost.calc(price=price, qty=float(qty_delta), side="buy")
            next_cash -= gross + trade_cost.total

        turn = 0.0 if nav == 0.0 else float(trade_value.sum() / nav)
        next_qty = current_qty.add(delta, fill_value=0.0).astype(float)
        return next_cash, next_qty, turn

    @staticmethod
    def _tradable(tradable: pd.DataFrame | None, close: pd.DataFrame) -> pd.DataFrame:
        if tradable is None:
            return pd.DataFrame(True, index=close.index, columns=close.columns)
        return tradable.reindex_like(close).fillna(False).astype(bool)

    @staticmethod
    def _schedule(
        schedule: pd.Series | RebalanceSchedule | None,
        close: pd.DataFrame,
    ) -> pd.Series:
        if schedule is None:
            return pd.Series(True, index=close.index, dtype=bool)
        if isinstance(schedule, RebalanceSchedule):
            return schedule.flags(close.index).reindex(close.index).fillna(False).astype(bool)
        return schedule.reindex(close.index).fillna(False).astype(bool)

    @staticmethod
    def _normalize_quantity(target_qty: pd.Series, allow_fractional: bool) -> pd.Series:
        target_qty = target_qty.fillna(0.0).astype(float)
        if allow_fractional:
            return target_qty
        normalized = target_qty.copy()
        normalized.loc[normalized >= 0.0] = np.floor(normalized.loc[normalized >= 0.0])
        normalized.loc[normalized < 0.0] = np.ceil(normalized.loc[normalized < 0.0])
        return normalized

    def _cap_buy_delta(
        self,
        buy_delta: pd.Series,
        fill_price: pd.Series,
        cash: float,
        allow_fractional: bool,
    ) -> pd.Series:
        buy_delta = buy_delta.fillna(0.0).astype(float)
        if buy_delta.le(0.0).all() or cash <= 0.0:
            return buy_delta.where(buy_delta.gt(0.0), 0.0)

        desired_spend = 0.0
        for symbol, qty_delta in buy_delta.items():
            if qty_delta <= 0.0:
                continue
            price = float(fill_price.loc[symbol])
            gross = price * float(qty_delta)
            trade_cost = self.cost.calc(price=price, qty=float(qty_delta), side="buy")
            desired_spend += gross + trade_cost.total

        if desired_spend <= cash:
            return self._normalize_quantity(target_qty=buy_delta, allow_fractional=allow_fractional)

        scale = cash / desired_spend
        scaled_buy_delta = buy_delta.mul(scale)
        scaled_buy_delta = self._normalize_quantity(
            target_qty=scaled_buy_delta,
            allow_fractional=allow_fractional,
        )

        spend = 0.0
        for symbol, qty_delta in scaled_buy_delta.items():
            if qty_delta <= 0.0:
                continue
            price = float(fill_price.loc[symbol])
            gross = price * float(qty_delta)
            trade_cost = self.cost.calc(price=price, qty=float(qty_delta), side="buy")
            spend += gross + trade_cost.total

        if spend <= cash:
            return scaled_buy_delta

        adjusted_buy_delta = scaled_buy_delta.copy()
        for symbol in adjusted_buy_delta.sort_values(ascending=False).index:
            qty_delta = float(adjusted_buy_delta.loc[symbol])
            if qty_delta <= 0.0:
                continue

            price = float(fill_price.loc[symbol])
            step = 1.0 if allow_fractional else 1.0
            while qty_delta > 0.0 and spend > cash:
                next_qty = max(0.0, qty_delta - step)
                removed_qty = qty_delta - next_qty
                if removed_qty == 0.0:
                    break
                gross = price * removed_qty
                trade_cost = self.cost.calc(price=price, qty=removed_qty, side="buy")
                spend -= gross + trade_cost.total
                qty_delta = next_qty
            adjusted_buy_delta.loc[symbol] = qty_delta
            if spend <= cash:
                break

        return adjusted_buy_delta.astype(float)
