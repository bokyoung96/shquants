from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.base import ConstructionResult
from backtesting.construction.sector_rotation import SectorRotationLongShort, _optional_frame, _required_frame
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle

from .benchmark_overlay import _BenchmarkOverlayConstruction
from .composable import ComposableStrategy


@dataclass(slots=True)
class RrgSectorRotation(ComposableStrategy):
    top_n: int = 25
    bottom_n: int = 25
    lookback: int = 20
    flow_lookback: int = 20
    flow_impulse_lookback: int = 5
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    gross_long: float = 1.0
    gross_short: float = 1.0
    fwd_partial_confidence: float = 0.7
    weighting: str = "equal"
    hold_weakening_longs: bool = False
    hold_long_mode: str = "force"
    alpha_mode: str = "combined"
    sector_budget_mode: str = "market_cap"
    use_name_cap: bool = True
    fwd_entry_rule: str = "state_conditioned"

    def __post_init__(self) -> None:
        if self.alpha_mode not in {"combined", "flow_only", "fwd_only"}:
            raise ValueError(f"unsupported alpha_mode: {self.alpha_mode}")
        if self.sector_budget_mode not in {"market_cap", "state_equal"}:
            raise ValueError(f"unsupported sector_budget_mode: {self.sector_budget_mode}")
        if self.fwd_entry_rule not in {"state_conditioned", "dual_family", "majority_horizons", "net_positive"}:
            raise ValueError(f"unsupported fwd_entry_rule: {self.fwd_entry_rule}")
        self.signal_producer = _RrgSectorRotationSignal(
            lookback=self.lookback,
            flow_lookback=self.flow_lookback,
            flow_impulse_lookback=self.flow_impulse_lookback,
            rrg_medium_lookback=self.rrg_medium_lookback,
            rrg_momentum_lookback=self.rrg_momentum_lookback,
            rrg_short_lookback=self.rrg_short_lookback,
            fwd_partial_confidence=self.fwd_partial_confidence,
            hold_weakening_longs=self.hold_weakening_longs,
            alpha_mode=self.alpha_mode,
            sector_budget_mode=self.sector_budget_mode,
            fwd_entry_rule=self.fwd_entry_rule,
        )
        self.construction_rule = SectorRotationLongShort(
            long_count=self.top_n if self.use_name_cap else None,
            short_count=self.bottom_n,
            gross_long=self.gross_long,
            gross_short=self.gross_short,
            weighting=self.weighting,
            hold_long_mode=self.hold_long_mode,
        )


@dataclass(slots=True)
class RrgFwdBenchmarkTilt(ComposableStrategy):
    lookback: int = 20
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    tilt_rule: str = "majority_horizons"
    active_share_target: float = 0.06
    max_stock_active: float = 0.0075
    max_sector_active: float = 0.03

    def __post_init__(self) -> None:
        if self.tilt_rule not in {"state_conditioned", "dual_family", "majority_horizons", "supermajority_horizons", "net_positive"}:
            raise ValueError(f"unsupported tilt_rule: {self.tilt_rule}")
        self.signal_producer = _RrgFwdBenchmarkTiltSignal(
            lookback=self.lookback,
            rrg_medium_lookback=self.rrg_medium_lookback,
            rrg_momentum_lookback=self.rrg_momentum_lookback,
            rrg_short_lookback=self.rrg_short_lookback,
            tilt_rule=self.tilt_rule,
        )
        self.construction_rule = _SparseBenchmarkOverlayConstruction(
            active_share_target=self.active_share_target,
            max_stock_active=self.max_stock_active,
            max_sector_active=self.max_sector_active,
            min_names=1,
        )


@dataclass(slots=True)
class RrgPureSectorRotation(ComposableStrategy):
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    selection_rule: str = "leading_improving"
    weighting_rule: str = "equal"

    def __post_init__(self) -> None:
        valid_selection = {
            "leading_improving",
            "leading",
            "improving",
            "momentum_positive",
            "rs_positive",
            "score_positive",
            "leading_improving_ex_weakening",
            "leading_improving_weakening",
            "leading_improving_resilient_weakening",
        }
        valid_weighting = {"equal", "score", "momentum", "relative_strength", "state_rank"}
        if self.selection_rule not in valid_selection:
            raise ValueError(f"unsupported selection_rule: {self.selection_rule}")
        if self.weighting_rule not in valid_weighting:
            raise ValueError(f"unsupported weighting_rule: {self.weighting_rule}")
        self.signal_producer = _RrgPureSectorSignal(
            rrg_medium_lookback=self.rrg_medium_lookback,
            rrg_momentum_lookback=self.rrg_momentum_lookback,
            rrg_short_lookback=self.rrg_short_lookback,
            selection_rule=self.selection_rule,
            weighting_rule=self.weighting_rule,
        )
        self.construction_rule = _RrgPureSectorConstruction()


@dataclass(slots=True)
class _RrgPureSectorSignal:
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    selection_rule: str = "leading_improving"
    weighting_rule: str = "equal"

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_BM,
            DatasetId.QW_K200_YN,
            DatasetId.QW_WICS_SEC_BIG,
            DatasetId.QW_MKTCAP,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"].astype(float)
        k200 = market.frames["k200_yn"].reindex_like(close).fillna(False).astype(bool)
        active_columns = k200.columns[k200.any(axis=0)]
        if len(active_columns) > 0:
            close = close.loc[:, active_columns]
            k200 = k200.loc[:, active_columns]

        sector = market.frames["sector_big"].reindex(index=close.index, columns=close.columns).ffill()
        market_cap = market.frames["market_cap"].reindex(index=close.index, columns=close.columns).ffill().astype(float)
        benchmark_frame = market.frames["benchmark"]
        benchmark = benchmark_frame["IKS200"] if "IKS200" in benchmark_frame.columns else benchmark_frame.iloc[:, 0]
        benchmark = benchmark.reindex(close.index).ffill().astype(float)

        rrg = _build_rrg_measures(
            close=close,
            benchmark=benchmark,
            sector=sector,
            membership=k200,
            market_cap=market_cap,
            medium_lookback=self.rrg_medium_lookback,
            momentum_lookback=self.rrg_momentum_lookback,
            short_lookback=self.rrg_short_lookback,
        )
        sector_budget = _build_pure_sector_budget(
            rrg_state=rrg["state"],
            relative_strength=rrg["relative_strength"],
            momentum=rrg["momentum"],
            selection_rule=self.selection_rule,
            weighting_rule=self.weighting_rule,
        )
        state_by_symbol = _map_sector_state_to_symbols(sector=sector, rrg_state=rrg["state"])
        alpha = pd.DataFrame(0.0, index=close.index, columns=close.columns, dtype=float)
        alpha = alpha.mask(state_by_symbol.notna() & k200, 1.0)

        return SignalBundle(
            alpha=alpha.where(k200),
            context={
                "tradable": k200,
                "sector": sector,
                "sector_weight_basis": market_cap.where(k200),
                "sector_budget": sector_budget,
            },
            meta={
                "rrg_state": rrg["state"],
                "relative_strength": rrg["relative_strength"],
                "momentum": rrg["momentum"],
                "sector_budget": sector_budget,
            },
        )


@dataclass(slots=True)
class _RrgPureSectorConstruction:
    def build(self, bundle: SignalBundle) -> ConstructionResult:
        alpha = bundle.alpha.astype(float)
        sector = _required_frame(bundle, "sector").reindex(index=alpha.index, columns=alpha.columns)
        basis = _required_frame(bundle, "sector_weight_basis").reindex(index=alpha.index, columns=alpha.columns)
        basis = basis.fillna(0.0).astype(float).clip(lower=0.0)
        tradable = _optional_frame(bundle, "tradable", alpha.notna()).reindex(index=alpha.index, columns=alpha.columns)
        tradable = tradable.fillna(False).astype(bool)
        sector_budget = _required_frame(bundle, "sector_budget").reindex(index=alpha.index).fillna(0.0).astype(float)

        weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns, dtype=float)
        for ts in alpha.index:
            row_sector = sector.loc[ts]
            row_basis = basis.loc[ts].where(tradable.loc[ts] & row_sector.notna(), 0.0)
            row_budget = sector_budget.loc[ts]
            for sector_name, budget in row_budget[row_budget > 0.0].items():
                members = row_sector[row_sector.eq(sector_name)].index
                member_basis = row_basis.reindex(members).fillna(0.0).clip(lower=0.0)
                denom = float(member_basis.sum())
                if denom <= 0.0:
                    continue
                weights.loc[ts, members] = float(budget) * (member_basis / denom)

        selection_mask = weights.gt(0.0)
        return ConstructionResult(
            base_target_weights=weights,
            selection_mask=selection_mask,
            group_long_budget=sector_budget,
            group_short_budget=None,
            meta={
                "group_long_budget": sector_budget,
            },
        )


@dataclass(slots=True)
class _RrgFwdBenchmarkTiltSignal:
    lookback: int = 20
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    tilt_rule: str = "majority_horizons"

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_BM,
            DatasetId.QW_K200_YN,
            DatasetId.QW_WICS_SEC_BIG,
            DatasetId.QW_MKTCAP,
            DatasetId.QW_EPS_NFQ1,
            DatasetId.QW_EPS_NFQ2,
            DatasetId.QW_EPS_NFY1,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_OP_NFQ2,
            DatasetId.QW_OP_NFY1,
        )

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"].astype(float)
        k200 = market.frames["k200_yn"].reindex_like(close).fillna(False).astype(bool)
        active_columns = k200.columns[k200.any(axis=0)]
        if len(active_columns) > 0:
            close = close.loc[:, active_columns]
            k200 = k200.loc[:, active_columns]

        sector = market.frames["sector_big"].reindex(index=close.index, columns=close.columns).ffill()
        market_cap = market.frames["market_cap"].reindex(index=close.index, columns=close.columns).ffill().astype(float)
        benchmark_frame = market.frames["benchmark"]
        benchmark = benchmark_frame["IKS200"] if "IKS200" in benchmark_frame.columns else benchmark_frame.iloc[:, 0]
        benchmark = benchmark.reindex(close.index).ffill().astype(float)

        rrg_state, _long_sector, _short_sector = _build_rrg_context(
            close=close,
            benchmark=benchmark,
            sector=sector,
            membership=k200,
            market_cap=market_cap,
            medium_lookback=self.rrg_medium_lookback,
            momentum_lookback=self.rrg_momentum_lookback,
            short_lookback=self.rrg_short_lookback,
        )
        fwd_score, fwd_confidence, _fwd_coverage = _build_forward_score(
            frames=market.frames,
            index=close.index,
            columns=close.columns,
            sector=sector,
            lookback=self.lookback,
            partial_confidence=0.7,
        )
        eps_delta, eps_count, eps_positive_count = _estimate_family_delta(
            frames=market.frames,
            keys=("eps_fwd_q1", "eps_fwd_q2", "eps_fwd"),
            index=close.index,
            columns=close.columns,
            sector=sector,
            lookback=self.lookback,
        )
        op_delta, op_count, op_positive_count = _estimate_family_delta(
            frames=market.frames,
            keys=("op_fwd_q1", "op_fwd_q2", "op_fwd"),
            index=close.index,
            columns=close.columns,
            sector=sector,
            lookback=self.lookback,
        )
        state_by_symbol = _map_sector_state_to_symbols(
            sector=sector,
            rrg_state=rrg_state,
        )
        positive_count = eps_positive_count.add(op_positive_count, fill_value=0.0)
        available_count = eps_count.add(op_count, fill_value=0.0)
        family_count = eps_delta.notna().astype(int) + op_delta.notna().astype(int)
        net_delta = eps_delta.fillna(0.0).add(op_delta.fillna(0.0)).divide(family_count.replace(0, np.nan))
        positive_family = eps_delta.gt(0.0).astype(int) + op_delta.gt(0.0).astype(int)
        negative_family = eps_delta.lt(0.0).astype(int) + op_delta.lt(0.0).astype(int)

        active_state = state_by_symbol.isin(("Leading", "Improving"))
        weak_state = state_by_symbol.isin(("Weakening", "Lagging"))
        if self.tilt_rule == "state_conditioned":
            ow = (state_by_symbol.eq("Leading") & positive_family.eq(2)) | (state_by_symbol.eq("Improving") & positive_family.ge(1))
            uw = weak_state & negative_family.ge(1)
        elif self.tilt_rule == "dual_family":
            ow = active_state & positive_family.eq(2)
            uw = weak_state & negative_family.eq(2)
        elif self.tilt_rule == "majority_horizons":
            ow = active_state & positive_count.gt(available_count / 2.0)
            uw = weak_state & positive_count.lt(available_count / 2.0)
        elif self.tilt_rule == "supermajority_horizons":
            ow = active_state & positive_count.ge(available_count.mul(2.0 / 3.0).apply(np.ceil))
            uw = weak_state & positive_count.le(available_count.mul(1.0 / 3.0).apply(np.floor))
        elif self.tilt_rule == "net_positive":
            ow = active_state & net_delta.gt(0.0)
            uw = weak_state & net_delta.lt(0.0)
        else:
            raise ValueError(f"unsupported tilt_rule: {self.tilt_rule}")

        score = fwd_score.mul(fwd_confidence).where(k200)
        alpha = pd.DataFrame(0.0, index=close.index, columns=close.columns, dtype=float)
        alpha = alpha.mask(ow & score.notna(), score)
        alpha = alpha.mask(uw & score.notna(), -(1.0 - score))
        alpha = alpha.where(k200, 0.0).fillna(0.0).astype(float)

        benchmark_base = market_cap.where(k200)
        benchmark_weights = benchmark_base.div(benchmark_base.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        membership = k200 & benchmark_weights.gt(0.0)
        inclusion = alpha.ne(0.0) & membership
        overlay_scale = pd.Series(1.0, index=close.index, dtype=float).where(inclusion.any(axis=1), 0.0)

        return SignalBundle(
            alpha=alpha,
            context={
                "sector": sector,
                "benchmark_weights": benchmark_weights,
                "benchmark_membership": membership,
                "overlay_scale": overlay_scale,
                "inclusion": inclusion,
                "rrg_state": rrg_state,
            },
            meta={
                "fwd_score": fwd_score,
                "fwd_confidence": fwd_confidence,
            },
        )


class _SparseBenchmarkOverlayConstruction(_BenchmarkOverlayConstruction):
    def _build_active_overlay_values(
        self,
        *,
        signal: np.ndarray,
        base: np.ndarray,
        sector: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        active = np.zeros(signal.shape, dtype=float)
        keep = np.flatnonzero(np.abs(signal) > 1e-12)
        if keep.size == 0:
            return active

        raw = signal[keep].astype(float, copy=True)
        raw = raw - float((raw * base[keep]).sum())
        if float(np.abs(raw).sum()) <= 0.0:
            return active

        gross_budget = max(self.active_share_target * scale, 0.0)
        if gross_budget <= 0.0:
            return active

        pos = np.clip(raw, 0.0, None)
        neg = -np.clip(raw, None, 0.0)
        pos_sum = float(pos.sum())
        neg_sum = float(neg.sum())
        if pos_sum <= 0.0 or neg_sum <= 0.0:
            return active

        active[keep] += (gross_budget / 2.0) * (pos / pos_sum)
        active[keep] -= (gross_budget / 2.0) * (neg / neg_sum)
        active = np.clip(active, -self.max_stock_active, self.max_stock_active)
        active = self._recenter_values(active, base)
        active = self._cap_sector_values(active, sector)
        active = self._recenter_values(active, base)
        active = np.minimum(np.maximum(active, -base), self.max_stock_active)
        active = self._recenter_values(active, base)
        return np.nan_to_num(active, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(slots=True)
class _RrgSectorRotationSignal:
    lookback: int = 20
    flow_lookback: int = 20
    flow_impulse_lookback: int = 5
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    fwd_partial_confidence: float = 0.7
    hold_weakening_longs: bool = False
    alpha_mode: str = "combined"
    sector_budget_mode: str = "market_cap"
    fwd_entry_rule: str = "state_conditioned"

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        base = (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_BM,
            DatasetId.QW_K200_YN,
            DatasetId.QW_WICS_SEC_BIG,
            DatasetId.QW_MKTCAP,
        )
        flow = (
            DatasetId.QW_V,
            DatasetId.QW_FOREIGN,
            DatasetId.QW_INSTITUTION,
            DatasetId.QW_RETAIL,
        )
        fwd = (
            DatasetId.QW_EPS_NFQ1,
            DatasetId.QW_EPS_NFQ2,
            DatasetId.QW_EPS_NFY1,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_OP_NFQ2,
            DatasetId.QW_OP_NFY1,
        )
        if self.alpha_mode == "flow_only":
            return (*base, *flow)
        if self.alpha_mode == "fwd_only":
            return (*base, *fwd)
        return (*base, *flow, *fwd)

    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"].astype(float)
        k200 = market.frames["k200_yn"].reindex_like(close).fillna(False).astype(bool)
        active_columns = k200.columns[k200.any(axis=0)]
        if len(active_columns) > 0:
            close = close.loc[:, active_columns]
            k200 = k200.loc[:, active_columns]

        sector = market.frames["sector_big"].reindex(index=close.index, columns=close.columns).ffill()
        market_cap = market.frames["market_cap"].reindex(index=close.index, columns=close.columns).ffill().astype(float)
        benchmark_frame = market.frames["benchmark"]
        benchmark = benchmark_frame["IKS200"] if "IKS200" in benchmark_frame.columns else benchmark_frame.iloc[:, 0]
        benchmark = benchmark.reindex(close.index).ffill().astype(float)

        rrg_state, long_sector, short_sector = _build_rrg_context(
            close=close,
            benchmark=benchmark,
            sector=sector,
            membership=k200,
            market_cap=market_cap,
            medium_lookback=self.rrg_medium_lookback,
            momentum_lookback=self.rrg_momentum_lookback,
            short_lookback=self.rrg_short_lookback,
        )
        meta = {
            "rrg_state": rrg_state,
        }
        if self.alpha_mode == "flow_only":
            flow_score_20d, flow_score_5d = _build_flow_scores(
                frames=market.frames,
                close=close,
                sector=sector,
                flow_lookback=self.flow_lookback,
                impulse_lookback=self.flow_impulse_lookback,
            )
            alpha = flow_score_20d.where(k200 & flow_score_20d.notna())
            tradable = k200 & flow_score_20d.notna()
            long_entry = tradable
            meta.update(
                {
                    "flow_score_20d": flow_score_20d,
                    "flow_score_5d": flow_score_5d,
                }
            )
        else:
            fwd_score, fwd_confidence, fwd_coverage = _build_forward_score(
                frames=market.frames,
                index=close.index,
                columns=close.columns,
                sector=sector,
                lookback=self.lookback,
                partial_confidence=self.fwd_partial_confidence,
            )
            fwd_entry = _build_forward_entry_mask(
                frames=market.frames,
                index=close.index,
                columns=close.columns,
                sector=sector,
                rrg_state=rrg_state,
                lookback=self.lookback,
                entry_rule=self.fwd_entry_rule,
            )
            if self.alpha_mode == "fwd_only":
                alpha = fwd_score.mul(fwd_confidence).where(k200 & fwd_score.notna())
            else:
                flow_score_20d, flow_score_5d = _build_flow_scores(
                    frames=market.frames,
                    close=close,
                    sector=sector,
                    flow_lookback=self.flow_lookback,
                    impulse_lookback=self.flow_impulse_lookback,
                )
                alpha = (0.5 * fwd_score.mul(fwd_confidence) + 0.5 * flow_score_20d).where(k200 & fwd_score.notna())
                meta.update(
                    {
                        "flow_score_20d": flow_score_20d,
                        "flow_score_5d": flow_score_5d,
                    }
                )
            tradable = k200 & fwd_score.notna()
            long_entry = tradable & fwd_entry
            meta.update(
                {
                    "fwd_score": fwd_score,
                    "fwd_confidence": fwd_confidence,
                    "fwd_coverage": fwd_coverage,
                }
            )

        if self.sector_budget_mode == "state_equal":
            sector_weight_basis = _build_state_equal_sector_weight_basis(
                sector=sector,
                membership=k200,
                rrg_state=rrg_state,
            )
        else:
            sector_weight_basis = market_cap.where(k200)

        context = {
            "tradable": tradable,
            "long_entry": long_entry,
            "sector": sector,
            "long_sector": long_sector,
            "short_sector": short_sector,
            "sector_weight_basis": sector_weight_basis,
        }
        if self.hold_weakening_longs:
            context["hold_long_sector"] = long_sector | rrg_state.eq("Weakening")

        return SignalBundle(
            alpha=alpha,
            context=context,
            meta=meta,
        )


def _build_rrg_context(
    *,
    close: pd.DataFrame,
    benchmark: pd.Series,
    sector: pd.DataFrame,
    membership: pd.DataFrame,
    market_cap: pd.DataFrame,
    medium_lookback: int,
    momentum_lookback: int,
    short_lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = close.pct_change(fill_method=None)
    benchmark_returns = benchmark.pct_change(fill_method=None)
    sector_returns = _sector_weighted_returns(
        returns=returns,
        sector=sector,
        membership=membership,
        weights=market_cap,
    )
    sector_index = (1.0 + sector_returns.fillna(0.0)).cumprod()
    benchmark_index = (1.0 + benchmark_returns.fillna(0.0)).cumprod()
    relative = sector_index.divide(benchmark_index, axis=0)

    medium_mean = relative.rolling(medium_lookback, min_periods=max(5, medium_lookback // 3)).mean()
    short_mean = relative.rolling(short_lookback, min_periods=max(5, short_lookback // 3)).mean()
    relative_strength = relative.divide(medium_mean.replace(0.0, np.nan)) - 1.0
    short_relative = relative.divide(short_mean.replace(0.0, np.nan)) - 1.0
    momentum = short_relative - short_relative.shift(momentum_lookback)

    return _classify_rrg_states(
        relative_strength=relative_strength,
        momentum=momentum,
    )


def _build_rrg_measures(
    *,
    close: pd.DataFrame,
    benchmark: pd.Series,
    sector: pd.DataFrame,
    membership: pd.DataFrame,
    market_cap: pd.DataFrame,
    medium_lookback: int,
    momentum_lookback: int,
    short_lookback: int,
) -> dict[str, pd.DataFrame]:
    returns = close.pct_change(fill_method=None)
    benchmark_returns = benchmark.pct_change(fill_method=None)
    sector_returns = _sector_weighted_returns(
        returns=returns,
        sector=sector,
        membership=membership,
        weights=market_cap,
    )
    sector_index = (1.0 + sector_returns.fillna(0.0)).cumprod()
    benchmark_index = (1.0 + benchmark_returns.fillna(0.0)).cumprod()
    relative = sector_index.divide(benchmark_index, axis=0)

    medium_mean = relative.rolling(medium_lookback, min_periods=max(5, medium_lookback // 3)).mean()
    short_mean = relative.rolling(short_lookback, min_periods=max(5, short_lookback // 3)).mean()
    relative_strength = relative.divide(medium_mean.replace(0.0, np.nan)) - 1.0
    short_relative = relative.divide(short_mean.replace(0.0, np.nan)) - 1.0
    momentum = short_relative - short_relative.shift(momentum_lookback)
    state, _long_sector, _short_sector = _classify_rrg_states(
        relative_strength=relative_strength,
        momentum=momentum,
    )
    return {
        "state": state,
        "relative_strength": relative_strength,
        "momentum": momentum,
    }


def _build_pure_sector_budget(
    *,
    rrg_state: pd.DataFrame,
    relative_strength: pd.DataFrame,
    momentum: pd.DataFrame,
    selection_rule: str,
    weighting_rule: str,
) -> pd.DataFrame:
    selected = _select_pure_sectors(
        rrg_state=rrg_state,
        relative_strength=relative_strength,
        momentum=momentum,
        selection_rule=selection_rule,
    )
    raw_score = _pure_sector_score(
        rrg_state=rrg_state,
        relative_strength=relative_strength,
        momentum=momentum,
        weighting_rule=weighting_rule,
    )
    positive_score = raw_score.where(selected).clip(lower=0.0).fillna(0.0)
    equal_score = selected.astype(float)
    score = positive_score.copy()
    fallback_rows = score.sum(axis=1).le(0.0)
    if bool(fallback_rows.any()):
        score.loc[fallback_rows] = equal_score.loc[fallback_rows]
    score = score.where(selected, 0.0).fillna(0.0)
    denom = score.sum(axis=1).replace(0.0, np.nan)
    return score.div(denom, axis=0).fillna(0.0).astype(float)


def _select_pure_sectors(
    *,
    rrg_state: pd.DataFrame,
    relative_strength: pd.DataFrame,
    momentum: pd.DataFrame,
    selection_rule: str,
) -> pd.DataFrame:
    if selection_rule == "leading_improving":
        selected = rrg_state.isin(("Leading", "Improving"))
    elif selection_rule == "leading":
        selected = rrg_state.eq("Leading")
    elif selection_rule == "improving":
        selected = rrg_state.eq("Improving")
    elif selection_rule == "momentum_positive":
        selected = momentum.gt(0.0)
    elif selection_rule == "rs_positive":
        selected = relative_strength.gt(0.0)
    elif selection_rule == "score_positive":
        selected = relative_strength.add(momentum, fill_value=0.0).gt(0.0)
    elif selection_rule == "leading_improving_ex_weakening":
        selected = rrg_state.isin(("Leading", "Improving")) & momentum.ge(0.0)
    elif selection_rule == "leading_improving_weakening":
        selected = rrg_state.isin(("Leading", "Improving", "Weakening"))
    elif selection_rule == "leading_improving_resilient_weakening":
        resilient_weakening = rrg_state.eq("Weakening") & relative_strength.add(momentum, fill_value=0.0).gt(0.0)
        selected = rrg_state.isin(("Leading", "Improving")) | resilient_weakening
    else:
        raise ValueError(f"unsupported selection_rule: {selection_rule}")
    valid = relative_strength.notna() & momentum.notna()
    return (selected & valid).fillna(False).astype(bool)


def _pure_sector_score(
    *,
    rrg_state: pd.DataFrame,
    relative_strength: pd.DataFrame,
    momentum: pd.DataFrame,
    weighting_rule: str,
) -> pd.DataFrame:
    if weighting_rule == "equal":
        return pd.DataFrame(1.0, index=rrg_state.index, columns=rrg_state.columns)
    if weighting_rule == "score":
        return relative_strength.add(momentum, fill_value=0.0)
    if weighting_rule == "momentum":
        return momentum
    if weighting_rule == "relative_strength":
        return relative_strength
    if weighting_rule == "state_rank":
        state_score = pd.DataFrame(0.0, index=rrg_state.index, columns=rrg_state.columns, dtype=float)
        state_score = state_score.mask(rrg_state.eq("Leading"), 4.0)
        state_score = state_score.mask(rrg_state.eq("Improving"), 3.0)
        state_score = state_score.mask(rrg_state.eq("Weakening"), 2.0)
        state_score = state_score.mask(rrg_state.eq("Lagging"), 1.0)
        return state_score
    raise ValueError(f"unsupported weighting_rule: {weighting_rule}")


def _build_state_equal_sector_weight_basis(
    *,
    sector: pd.DataFrame,
    membership: pd.DataFrame,
    rrg_state: pd.DataFrame,
) -> pd.DataFrame:
    basis = pd.DataFrame(0.0, index=sector.index, columns=sector.columns, dtype=float)
    aligned_membership = membership.reindex(index=sector.index, columns=sector.columns).fillna(False).astype(bool)
    aligned_state = rrg_state.reindex(index=sector.index)
    active_states = ("Leading", "Improving")

    for ts in sector.index:
        row_sector = sector.loc[ts]
        row_membership = aligned_membership.loc[ts]
        row_state = aligned_state.loc[ts]
        present_states = [
            state_name
            for state_name in active_states
            if bool(row_state.eq(state_name).any())
        ]
        if not present_states:
            continue
        state_budget = 1.0 / len(present_states)
        for state_name in present_states:
            state_sectors = row_state[row_state.eq(state_name)].index
            state_sectors = [sector_name for sector_name in state_sectors if bool((row_sector.eq(sector_name) & row_membership).any())]
            if not state_sectors:
                continue
            sector_budget = state_budget / len(state_sectors)
            for sector_name in state_sectors:
                members = row_sector[row_sector.eq(sector_name) & row_membership].index
                if len(members) == 0:
                    continue
                basis.loc[ts, members] = sector_budget / len(members)
    return basis


def _sector_weighted_returns(
    *,
    returns: pd.DataFrame,
    sector: pd.DataFrame,
    membership: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    rows: dict[pd.Timestamp, dict[object, float]] = {}
    for ts in returns.index:
        valid = membership.loc[ts].astype(bool) & returns.loc[ts].notna() & sector.loc[ts].notna()
        row: dict[object, float] = {}
        for sector_name in pd.unique(sector.loc[ts, valid]):
            names = returns.columns[valid & sector.loc[ts].eq(sector_name)]
            sector_weights = weights.loc[ts, names].fillna(0.0).clip(lower=0.0)
            if float(sector_weights.sum()) <= 0.0:
                sector_weights = pd.Series(1.0, index=names, dtype=float)
            sector_weights = sector_weights / float(sector_weights.sum())
            row[sector_name] = float((returns.loc[ts, names] * sector_weights).sum())
        rows[ts] = row
    return pd.DataFrame.from_dict(rows, orient="index").reindex(index=returns.index)


def _build_forward_score(
    *,
    frames: dict[str, pd.DataFrame],
    index: pd.Index,
    columns: pd.Index,
    sector: pd.DataFrame,
    lookback: int,
    partial_confidence: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eps_score, eps_count = _estimate_family_score(
        frames=frames,
        keys=("eps_fwd_q1", "eps_fwd_q2", "eps_fwd"),
        index=index,
        columns=columns,
        sector=sector,
        lookback=lookback,
    )
    op_score, op_count = _estimate_family_score(
        frames=frames,
        keys=("op_fwd_q1", "op_fwd_q2", "op_fwd"),
        index=index,
        columns=columns,
        sector=sector,
        lookback=lookback,
    )

    family_count = eps_score.notna().astype(int) + op_score.notna().astype(int)
    score = (eps_score.fillna(0.0) + op_score.fillna(0.0)).divide(family_count.replace(0, np.nan))
    confidence = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    confidence = confidence.mask(family_count.eq(1), partial_confidence)
    confidence = confidence.mask(family_count.ge(2), 1.0)
    coverage = (eps_count.fillna(0.0) + op_count.fillna(0.0)).astype(float)
    return score, confidence, coverage


def _build_forward_entry_mask(
    *,
    frames: dict[str, pd.DataFrame],
    index: pd.Index,
    columns: pd.Index,
    sector: pd.DataFrame,
    rrg_state: pd.DataFrame,
    lookback: int,
    entry_rule: str = "state_conditioned",
) -> pd.DataFrame:
    eps_delta, eps_count, eps_positive_count = _estimate_family_delta(
        frames=frames,
        keys=("eps_fwd_q1", "eps_fwd_q2", "eps_fwd"),
        index=index,
        columns=columns,
        sector=sector,
        lookback=lookback,
    )
    op_delta, op_count, op_positive_count = _estimate_family_delta(
        frames=frames,
        keys=("op_fwd_q1", "op_fwd_q2", "op_fwd"),
        index=index,
        columns=columns,
        sector=sector,
        lookback=lookback,
    )
    eps_positive = eps_delta.gt(0.0)
    op_positive = op_delta.gt(0.0)
    family_count = eps_delta.notna().astype(int) + op_delta.notna().astype(int)
    net_delta = eps_delta.fillna(0.0).add(op_delta.fillna(0.0)).divide(family_count.replace(0, np.nan))
    available_count = eps_count.add(op_count, fill_value=0.0)
    positive_count = eps_positive_count.add(op_positive_count, fill_value=0.0)
    state_by_symbol = _map_sector_state_to_symbols(
        sector=sector.reindex(index=index, columns=columns),
        rrg_state=rrg_state.reindex(index=index),
    )
    allowed_state = state_by_symbol.isin(("Leading", "Improving", "Weakening"))
    if entry_rule == "state_conditioned":
        leading = state_by_symbol.eq("Leading") & eps_positive & op_positive
        improving = state_by_symbol.eq("Improving") & (eps_positive | op_positive)
        weakening_survival = state_by_symbol.eq("Weakening") & (eps_positive | op_positive)
        entry = leading | improving | weakening_survival
    elif entry_rule == "dual_family":
        entry = allowed_state & eps_positive & op_positive
    elif entry_rule == "majority_horizons":
        entry = allowed_state & positive_count.gt(available_count / 2.0)
    elif entry_rule == "net_positive":
        entry = allowed_state & net_delta.gt(0.0)
    else:
        raise ValueError(f"unsupported fwd entry rule: {entry_rule}")
    return entry.fillna(False).astype(bool)


def _estimate_family_delta(
    *,
    frames: dict[str, pd.DataFrame],
    keys: tuple[str, ...],
    index: pd.Index,
    columns: pd.Index,
    sector: pd.DataFrame,
    lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    deltas: list[pd.DataFrame] = []
    available: list[pd.DataFrame] = []
    positive: list[pd.DataFrame] = []
    for key in keys:
        estimate = frames[key].reindex(index=index, columns=columns).ffill().astype(float)
        delta = _bounded_delta(current=estimate, prior=estimate.shift(lookback), sector=sector)
        deltas.append(delta)
        available.append(delta.notna().astype(float))
        positive.append(delta.gt(0.0).astype(float).where(delta.notna(), 0.0))

    delta_sum = sum(frame.fillna(0.0) for frame in deltas)
    count = sum(frame for frame in available)
    positive_count = sum(frame for frame in positive)
    average_delta = delta_sum.divide(count.replace(0.0, np.nan))
    return average_delta, count, positive_count


def _map_sector_state_to_symbols(*, sector: pd.DataFrame, rrg_state: pd.DataFrame) -> pd.DataFrame:
    state_by_symbol = pd.DataFrame(index=sector.index, columns=sector.columns, dtype=object)
    for ts in sector.index:
        row_state = rrg_state.loc[ts] if ts in rrg_state.index else pd.Series(dtype=object)
        state_by_symbol.loc[ts] = sector.loc[ts].map(row_state)
    return state_by_symbol


def _estimate_family_score(
    *,
    frames: dict[str, pd.DataFrame],
    keys: tuple[str, ...],
    index: pd.Index,
    columns: pd.Index,
    sector: pd.DataFrame,
    lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores: list[pd.DataFrame] = []
    available: list[pd.DataFrame] = []
    for key in keys:
        estimate = frames[key].reindex(index=index, columns=columns).ffill().astype(float)
        delta = _bounded_delta(current=estimate, prior=estimate.shift(lookback), sector=sector)
        ranked = _sector_rank(delta, sector=sector, ascending=True)
        scores.append(ranked)
        available.append(ranked.notna().astype(float))

    score_sum = sum(frame.fillna(0.0) for frame in scores)
    count = sum(frame for frame in available)
    score = score_sum.divide(count.replace(0.0, np.nan))
    return score, count


def _build_flow_scores(
    *,
    frames: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    sector: pd.DataFrame,
    flow_lookback: int,
    impulse_lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    foreign = frames["foreign_flow"].reindex_like(close).fillna(0.0).astype(float)
    inst = frames["inst_flow"].reindex_like(close).fillna(0.0).astype(float)
    retail = frames["retail_flow"].reindex_like(close).fillna(0.0).astype(float)
    volume = frames["volume"].reindex_like(close).fillna(0.0).astype(float)
    trading_value = close.mul(volume).replace(0.0, np.nan)
    flow_pressure = foreign.add(inst, fill_value=0.0).sub(retail, fill_value=0.0).divide(trading_value)
    flow_mean_20d = flow_pressure.rolling(flow_lookback, min_periods=max(2, flow_lookback // 2)).mean()
    flow_mean_5d = flow_pressure.rolling(impulse_lookback, min_periods=max(2, impulse_lookback // 2)).mean()
    flow_score_20d = _sector_rank(_rolling_zscore(flow_mean_20d, flow_lookback), sector=sector, ascending=True)
    flow_score_5d = _sector_rank(_rolling_zscore(flow_mean_5d, impulse_lookback), sector=sector, ascending=True)
    return flow_score_20d, flow_score_5d


def _rolling_zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=max(2, window // 2)).mean()
    std = frame.rolling(window, min_periods=max(2, window // 2)).std(ddof=0)
    return frame.sub(mean).divide(std.replace(0.0, np.nan))


def _bounded_delta(
    *,
    current: pd.DataFrame,
    prior: pd.DataFrame,
    sector: pd.DataFrame | None = None,
) -> pd.DataFrame:
    current = current.astype(float)
    prior = prior.reindex_like(current).astype(float)
    scale = current.abs().combine(prior.abs(), np.maximum)
    if sector is not None:
        sector_scale = _sector_abs_estimate_scale(current=current, prior=prior, sector=sector)
        scale = scale.combine(sector_scale, np.maximum)
    scale = scale.replace(0.0, np.nan)
    delta = current.sub(prior).divide(scale)
    return delta.clip(lower=-1.0, upper=1.0)


def _sector_abs_estimate_scale(
    *,
    current: pd.DataFrame,
    prior: pd.DataFrame,
    sector: pd.DataFrame,
) -> pd.DataFrame:
    scale = pd.DataFrame(np.nan, index=current.index, columns=current.columns, dtype=float)
    aligned_sector = sector.reindex(index=current.index, columns=current.columns)
    absolute_estimate = current.abs().combine(prior.abs(), np.maximum)
    for ts in current.index:
        row_sector = aligned_sector.loc[ts]
        row_abs = absolute_estimate.loc[ts]
        for sector_name in pd.unique(row_sector.dropna()):
            members = row_sector[row_sector.eq(sector_name)].index
            member_abs = row_abs.reindex(members)
            if not bool(member_abs.notna().any()):
                continue
            median_abs = member_abs.median(skipna=True)
            if pd.notna(median_abs):
                scale.loc[ts, members] = float(median_abs)
    return scale


def _sector_rank(values: pd.DataFrame, *, sector: pd.DataFrame, ascending: bool) -> pd.DataFrame:
    result = pd.DataFrame(np.nan, index=values.index, columns=values.columns, dtype=float)
    aligned_sector = sector.reindex(index=values.index, columns=values.columns)
    for ts in values.index:
        row_values = values.loc[ts]
        row_sector = aligned_sector.loc[ts]
        for sector_name in pd.unique(row_sector.dropna()):
            members = row_sector[row_sector.eq(sector_name)].index
            member_values = row_values.reindex(members).dropna()
            if member_values.empty:
                continue
            result.loc[ts, member_values.index] = member_values.rank(
                ascending=ascending,
                pct=True,
                method="average",
            )
    return result


def _classify_rrg_states(
    *,
    relative_strength: pd.DataFrame,
    momentum: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    leading = relative_strength.ge(0.0) & momentum.ge(0.0)
    improving = relative_strength.lt(0.0) & momentum.ge(0.0)
    lagging = relative_strength.lt(0.0) & momentum.lt(0.0)
    weakening = relative_strength.ge(0.0) & momentum.lt(0.0)

    valid = relative_strength.notna() & momentum.notna()
    states = pd.DataFrame("Unclassified", index=relative_strength.index, columns=relative_strength.columns, dtype=object)
    states = states.mask(leading, "Leading")
    states = states.mask(improving, "Improving")
    states = states.mask(lagging, "Lagging")
    states = states.mask(weakening, "Weakening")

    long_sector = (leading | improving) & valid
    short_sector = (lagging | weakening) & valid
    return states, long_sector.astype(bool), short_sector.astype(bool)
