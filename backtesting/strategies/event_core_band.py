from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.long_only import LongOnlyTopN
from backtesting.policy.base import BUCKET_LEDGER_COLUMNS, PositionPlan
from backtesting.signals.base import SignalBundle

from .base import RegisteredStrategy
from .composable import SignalProducer


@dataclass(slots=True)
class EventCoreBandStrategy(RegisteredStrategy):
    signal_producer: SignalProducer = field(init=False)
    top_n: int = 10
    core_fraction: float = 0.85
    active_fractions: tuple[float, ...] = (0.05, 0.05, 0.05)

    def __post_init__(self) -> None:
        if self.core_fraction <= 0.0 or self.core_fraction >= 1.0:
            raise ValueError("core_fraction must be between 0 and 1")
        if abs(self.core_fraction + sum(self.active_fractions) - 1.0) > 1e-9:
            raise ValueError("core_fraction plus active_fractions must sum to 1")
        self.construction_rule = LongOnlyTopN(top_n=self.top_n)

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return tuple(dict.fromkeys((DatasetId.QW_MKTCAP, *self.signal_producer.datasets)))

    def build_signal(self, market: pd.DataFrame) -> pd.DataFrame:
        return self.signal_producer.build(market).alpha

    def build_plan(self, market) -> PositionPlan:
        bundle = self.signal_producer.build(market)
        bundle = self._apply_universe(bundle=bundle, market=market)
        construction = self.construction_rule.build(bundle)
        base_active = construction.base_target_weights.fillna(0.0).astype(float)
        core = self._core_weights(market=market, template=base_active)

        entry_mask = self._aligned_mask(bundle.context["eligible_entry"], base_active)
        add_1_mask = self._aligned_mask(bundle.context["eligible_add_1"], base_active)
        add_2_mask = self._aligned_mask(bundle.context["eligible_add_2"], base_active)
        exit_mask = self._aligned_mask(bundle.context["eligible_exit"], base_active)

        dates = base_active.index
        symbols = list(base_active.columns)
        bucket_ids = ("core", "active_entry", "active_add_1", "active_add_2")
        active_state = [pd.Series(False, index=symbols, dtype=bool) for _ in self.active_fractions]
        rows: list[dict[str, object]] = []
        target_rows: dict[pd.Timestamp, pd.Series] = {}

        for date in dates:
            base_row = base_active.loc[date]
            selected_now = base_row.ne(0.0)
            active_state[0] = (active_state[0] | entry_mask.loc[date]) & selected_now & ~exit_mask.loc[date]
            active_state[1] = (active_state[1] | (add_1_mask.loc[date] & active_state[0])) & selected_now & ~exit_mask.loc[date]
            active_state[2] = (active_state[2] | (add_2_mask.loc[date] & active_state[1])) & selected_now & ~exit_mask.loc[date]

            active_entry = base_row.where(active_state[0], 0.0) * self.active_fractions[0]
            active_add_1 = base_row.where(active_state[1], 0.0) * self.active_fractions[1]
            active_add_2 = base_row.where(active_state[2], 0.0) * self.active_fractions[2]
            active_budget_in_use = float(active_entry.sum() + active_add_1.sum() + active_add_2.sum())
            bucket_rows = {
                "core": core.loc[date] * max(0.0, 1.0 - active_budget_in_use),
                "active_entry": active_entry,
                "active_add_1": active_add_1,
                "active_add_2": active_add_2,
            }
            final_row = bucket_rows["core"].copy()
            for bucket_id in bucket_ids[1:]:
                final_row = final_row.add(bucket_rows[bucket_id], fill_value=0.0)
            target_rows[date] = final_row

            for bucket_index, bucket_id in enumerate(bucket_ids):
                bucket_row = bucket_rows[bucket_id]
                nonzero = bucket_row[bucket_row != 0.0]
                for symbol, value in nonzero.items():
                    rows.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "side": "long",
                            "bucket_id": bucket_id,
                            "stage_index": bucket_index,
                            "target_weight": float(value),
                            "actual_weight": float(value),
                            "target_qty": 0.0,
                            "actual_qty": 0.0,
                            "entry_price": None,
                            "mark_price": None,
                            "bucket_return": 0.0,
                            "state": "active",
                            "event": "core" if bucket_id == "core" else "event_driven",
                            "construction_group": None,
                            "budget_id": bucket_id,
                        }
                    )

        target_weights = pd.DataFrame.from_dict(target_rows, orient="index").reindex(index=dates, columns=symbols).fillna(0.0)
        ledger = pd.DataFrame.from_records(rows, columns=BUCKET_LEDGER_COLUMNS)
        return PositionPlan(target_weights=target_weights, bucket_ledger=ledger, bucket_meta={}, validation={})

    def target_weights(self, signal: pd.Series) -> pd.Series:
        raise NotImplementedError("EventCoreBandStrategy uses build_plan directly")

    @staticmethod
    def _core_weights(market, template: pd.DataFrame) -> pd.DataFrame:
        market_cap = market.frames["market_cap"].reindex(index=template.index, columns=template.columns)
        if market.universe is not None:
            universe = market.universe.reindex_like(template).fillna(False).astype(bool)
            market_cap = market_cap.where(universe)
        market_cap = market_cap.where(market_cap.gt(0.0))
        row_sums = market_cap.sum(axis=1).replace(0.0, pd.NA)
        core = market_cap.div(row_sums, axis=0)
        fallback = template.notna().astype(float)
        fallback = fallback.div(fallback.sum(axis=1).replace(0.0, pd.NA), axis=0).fillna(0.0)
        core = core.where(core.notna(), fallback)
        core = core.fillna(0.0)
        return core.div(core.sum(axis=1).replace(0.0, pd.NA), axis=0).fillna(0.0)

    @staticmethod
    def _aligned_mask(frame: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
        return frame.reindex(index=base.index, columns=base.columns).fillna(False).astype(bool)

    @staticmethod
    def _apply_universe(bundle: SignalBundle, market) -> SignalBundle:
        if market.universe is None:
            return bundle
        universe = market.universe.reindex(index=bundle.alpha.index, columns=bundle.alpha.columns).fillna(False).astype(bool)
        context = dict(bundle.context)
        tradable = context.get("tradable")
        if isinstance(tradable, pd.DataFrame):
            context["tradable"] = tradable.reindex_like(universe).fillna(False).astype(bool) & universe
        else:
            context["tradable"] = universe
        return SignalBundle(alpha=bundle.alpha.where(universe), context=context, meta=dict(bundle.meta))
