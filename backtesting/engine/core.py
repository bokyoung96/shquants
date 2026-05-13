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

        cash = float(capital)
        dates = close.index
        columns = close.columns
        close_values = close.to_numpy(dtype=float, copy=False)
        weight_values = weights.to_numpy(dtype=float, copy=False)
        tradable_values = tradable.to_numpy(dtype=bool, copy=False)
        schedule_values = schedule.to_numpy(dtype=bool, copy=False)

        equity_values = np.zeros(len(dates), dtype=float)
        qty_values = np.zeros(close_values.shape, dtype=float)
        turnover_values = np.zeros(len(dates), dtype=float)
        current_qty = np.zeros(len(columns), dtype=float)

        if fill_mode == "next_open":
            open_ = None if open is None else open.reindex_like(close).astype(float)
            exec_prices = fill_prices(close=close, open_=open_, fill_mode="next_open")
            exec_values = exec_prices.reindex(index=dates[:-1], columns=columns).to_numpy(dtype=float, copy=False)
            equity_values[0] = capital
            qty_values[0] = current_qty

            for index in tqdm(range(1, len(dates)), desc="backtest", disable=not show_progress):
                if schedule_values[index - 1]:
                    cash, current_qty, turn = self._rebalance_values(
                        cash=cash,
                        current_qty=current_qty,
                        fill_price=exec_values[index - 1],
                        target_weight=weight_values[index - 1],
                        tradable=tradable_values[index],
                        allow_fractional=allow_fractional,
                    )
                else:
                    turn = 0.0
                equity_values[index] = cash + np.nansum(current_qty * self._zero_nan(close_values[index]))
                qty_values[index] = current_qty
                turnover_values[index] = turn
        elif fill_mode == "close":
            if schedule_values[0]:
                cash, current_qty, turn = self._rebalance_values(
                    cash=cash,
                    current_qty=current_qty,
                    fill_price=close_values[0],
                    target_weight=weight_values[0],
                    tradable=tradable_values[0],
                    allow_fractional=allow_fractional,
                )
            else:
                turn = 0.0
            equity_values[0] = cash + np.nansum(current_qty * self._zero_nan(close_values[0]))
            qty_values[0] = current_qty
            turnover_values[0] = turn

            for index in tqdm(range(1, len(dates)), desc="backtest", disable=not show_progress):
                if schedule_values[index]:
                    cash, current_qty, turn = self._rebalance_values(
                        cash=cash,
                        current_qty=current_qty,
                        fill_price=close_values[index],
                        target_weight=weight_values[index],
                        tradable=tradable_values[index],
                        allow_fractional=allow_fractional,
                    )
                else:
                    turn = 0.0
                equity_values[index] = cash + np.nansum(current_qty * self._zero_nan(close_values[index]))
                qty_values[index] = current_qty
                turnover_values[index] = turn
        else:
            fill_prices(close=close, open_=open, fill_mode=fill_mode)

        equity = pd.Series(equity_values, index=dates, dtype=float)
        qty = pd.DataFrame(qty_values, index=dates, columns=columns, dtype=float)
        turnover = pd.Series(turnover_values, index=dates, dtype=float)
        returns = equity.pct_change().fillna(0.0)
        return BacktestResult(
            equity=equity,
            returns=returns,
            weights=weights,
            qty=qty,
            turnover=turnover,
        )

    def _rebalance_values(
        self,
        cash: float,
        current_qty: np.ndarray,
        fill_price: np.ndarray,
        target_weight: np.ndarray,
        tradable: np.ndarray,
        allow_fractional: bool,
    ) -> tuple[float, np.ndarray, float]:
        safe_price = np.where(fill_price != 0.0, fill_price, np.nan)
        can_trade = tradable & ~np.isnan(safe_price)
        nav = cash + float(np.nansum(current_qty * self._zero_nan(fill_price)))

        with np.errstate(divide="ignore", invalid="ignore"):
            target_qty = target_weight * nav / safe_price
        target_qty = self._zero_nan(target_qty)
        target_qty = np.where(can_trade, target_qty, current_qty)
        target_qty = self._normalize_quantity_values(target_qty, allow_fractional=allow_fractional)

        raw_delta = self._zero_nan(target_qty - current_qty)
        sell_delta = np.where(raw_delta < 0.0, raw_delta, 0.0)
        buy_delta = np.where(raw_delta > 0.0, raw_delta, 0.0)

        price = self._zero_nan(fill_price)
        sell_gross = np.abs(price * sell_delta)
        next_cash = cash + float(sell_gross.sum() * (1.0 - self._cost_rate("sell")))

        buy_delta = self._cap_buy_values(
            buy_delta=buy_delta,
            fill_price=fill_price,
            cash=next_cash,
            allow_fractional=allow_fractional,
        )
        delta = sell_delta + buy_delta
        trade_value = np.abs(delta * price)

        buy_gross = np.abs(price * buy_delta)
        next_cash -= float(buy_gross.sum() * (1.0 + self._cost_rate("buy")))

        turn = 0.0 if nav == 0.0 else float(trade_value.sum() / nav)
        next_qty = (current_qty + delta).astype(float, copy=False)
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
    def _normalize_quantity_values(target_qty: np.ndarray, allow_fractional: bool) -> np.ndarray:
        target_qty = BacktestEngine._zero_nan(target_qty).astype(float, copy=True)
        if allow_fractional:
            return target_qty
        target_qty[target_qty >= 0.0] = np.floor(target_qty[target_qty >= 0.0])
        target_qty[target_qty < 0.0] = np.ceil(target_qty[target_qty < 0.0])
        return target_qty

    def _cap_buy_values(
        self,
        buy_delta: np.ndarray,
        fill_price: np.ndarray,
        cash: float,
        allow_fractional: bool,
    ) -> np.ndarray:
        buy_delta = self._zero_nan(buy_delta).astype(float, copy=True)
        buy_delta = np.where(buy_delta > 0.0, buy_delta, 0.0)
        if buy_delta.sum() <= 0.0 or cash <= 0.0:
            return buy_delta

        price = self._zero_nan(fill_price)
        rate = self._cost_rate("buy")
        desired_spend = float((price * buy_delta * (1.0 + rate)).sum())
        if desired_spend <= cash:
            return self._normalize_quantity_values(buy_delta, allow_fractional=allow_fractional)

        scaled_buy_delta = self._normalize_quantity_values(
            buy_delta * (cash / desired_spend),
            allow_fractional=allow_fractional,
        )
        spend = float((price * scaled_buy_delta * (1.0 + rate)).sum())
        if spend <= cash:
            return scaled_buy_delta

        adjusted_buy_delta = scaled_buy_delta.copy()
        for index in np.argsort(-adjusted_buy_delta):
            qty_delta = float(adjusted_buy_delta[index])
            if qty_delta <= 0.0:
                continue

            step = 1.0
            while qty_delta > 0.0 and spend > cash:
                next_qty = max(0.0, qty_delta - step)
                removed_qty = qty_delta - next_qty
                if removed_qty == 0.0:
                    break
                spend -= float(price[index] * removed_qty * (1.0 + rate))
                qty_delta = next_qty
            adjusted_buy_delta[index] = qty_delta
            if spend <= cash:
                break

        return adjusted_buy_delta.astype(float, copy=False)

    def _cost_rate(self, side: str) -> float:
        if side == "buy":
            return self.cost.fee + self.cost.slippage
        if side == "sell":
            return self.cost.fee + self.cost.sell_tax + self.cost.slippage
        self.cost.calc(price=0.0, qty=0.0, side=side)
        raise AssertionError("unreachable")

    @staticmethod
    def _zero_nan(values: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(values), 0.0, values)
