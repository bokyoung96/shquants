from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from backtesting.catalog import DataCatalog, DatasetId
from backtesting.features import feature_dataset_ids, feature_warmup_days
from backtesting.ingest.io import find_raw_path
from backtesting.selection import selection_fields
from backtesting.strategies import build_strategy
from backtesting.weighting import weighting_fields
from backtesting.universe import UniverseSpec

from .hooks import get_hook
from .models import DataPolicySpec, ExecutionSpec, ResolvedExecutionSpec


def _dataset_exists(catalog: DataCatalog, parquet_dir: Path, dataset_id: DatasetId) -> bool:
    return (parquet_dir / f"{catalog.get(dataset_id).stem}.parquet").exists()


def _dataset_source_available(catalog: DataCatalog, parquet_dir: Path, raw_dir: Path | None, dataset_id: DatasetId) -> bool:
    if _dataset_exists(catalog, parquet_dir, dataset_id):
        return True
    if raw_dir is None:
        return False
    try:
        find_raw_path(raw_dir, catalog.get(dataset_id).stem)
    except FileNotFoundError:
        return False
    return True


def _resolve_universe_dataset(universe_spec: UniverseSpec | None, dataset_id: DatasetId) -> DatasetId:
    if universe_spec is None:
        return dataset_id
    return universe_spec.resolve_dataset(dataset_id)



def _spec_feature_fields(spec: ExecutionSpec) -> tuple[str, ...]:
    ordered: list[str] = []
    if spec.selection is not None:
        ordered.extend(selection_fields(spec.selection))
    if spec.weighting is not None:
        ordered.extend(weighting_fields(spec.weighting))
    return tuple(dict.fromkeys(ordered))


def resolve_execution_spec(
    spec: ExecutionSpec,
    *,
    catalog: DataCatalog,
    parquet_dir: Path,
    raw_dir: Path | None,
    universe_spec: UniverseSpec | None,
) -> ResolvedExecutionSpec:
    dataset_ids: list[DatasetId] = [_resolve_universe_dataset(universe_spec, DatasetId.QW_ADJ_C)]
    notes: list[str] = []
    data_policy = spec.data_policy
    execution = spec

    if spec.fill_mode == "next_open":
        dataset_ids.append(_resolve_universe_dataset(universe_spec, DatasetId.QW_ADJ_O))

    if universe_spec is not None and universe_spec.membership_dataset is not None:
        dataset_ids.append(universe_spec.membership_dataset)

    feature_fields = _spec_feature_fields(spec)
    if feature_fields:
        dataset_ids.extend(
            _resolve_universe_dataset(universe_spec, dataset_id)
            for dataset_id in feature_dataset_ids(feature_fields)
        )
        required_warmup = max(spec.warmup_days, feature_warmup_days(feature_fields))
        if required_warmup > spec.warmup_days:
            execution = replace(execution, warmup_days=required_warmup)
            notes.append(f"warmup_days increased to {required_warmup} for feature lookbacks")

    if not spec.uses_composable_plan:
        if spec.weight_source.kind == "strategy":
            strategy = build_strategy(
                spec.strategy,
                top_n=spec.top_n,
                lookback=spec.lookback,
                flow_lookback=spec.flow_lookback,
                momentum_lookback=spec.momentum_lookback,
                liquidity_lookback=spec.liquidity_lookback,
                momentum_weight=spec.momentum_weight,
            )
            dataset_ids.extend(_resolve_universe_dataset(universe_spec, dataset_id) for dataset_id in strategy.datasets)
        elif spec.weight_source.kind == "hook":
            hook = get_hook(spec.weight_source.hook_id or "")
            dataset_ids.extend(_resolve_universe_dataset(universe_spec, dataset_id) for dataset_id in hook.required_datasets)
        else:
            raise ValueError(f"unsupported weight_source kind: {spec.weight_source.kind}")

    requested_basis = execution.data_policy.requested_weight_basis
    if requested_basis == "float_market_cap":
        float_id = _resolve_universe_dataset(universe_spec, DatasetId.QW_MKTCAP_FLT)
        market_id = _resolve_universe_dataset(universe_spec, DatasetId.QW_MKTCAP)
        if _dataset_source_available(catalog, parquet_dir, raw_dir, float_id):
            data_policy = replace(
                data_policy,
                resolved_weight_basis="float_market_cap",
            )
            dataset_ids.append(float_id)
        elif _dataset_source_available(catalog, parquet_dir, raw_dir, market_id) and "market_cap" in execution.data_policy.fallback_order:
            data_policy = replace(
                data_policy,
                resolved_weight_basis="market_cap",
                fallbacks_applied=(
                    {"from": "float_market_cap", "to": "market_cap", "reason": "missing qw_mktcap_flt source"},
                ),
            )
            dataset_ids.append(market_id)
            notes.append("float_market_cap unavailable; resolved to market_cap")
        else:
            raise ValueError("float_market_cap requested but parquet is unavailable and no fallback is configured")
    elif requested_basis == "market_cap":
        dataset_ids.append(_resolve_universe_dataset(universe_spec, DatasetId.QW_MKTCAP))
        data_policy = replace(data_policy, resolved_weight_basis="market_cap")

    deduped = tuple(dict.fromkeys(dataset_ids))
    execution = replace(execution, data_policy=data_policy)
    return ResolvedExecutionSpec(
        execution=execution,
        dataset_ids=deduped,
        schedule=execution.schedule,
        hook_id=execution.weight_source.hook_id,
        resolution_notes=tuple(notes),
    )
