# EMP008 Factor Registry and Quantile Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize EMP008 factor metadata and add reproducible Q1-Q5 single-factor tests using both equal and total-market-cap weighting.

**Architecture:** An enum-backed registry owns factor builders, dependencies, preprocessing, direction, and factor-set membership. A shared preparation bundle feeds optimization, attribution, and a new quantile evaluator so all three consume identical model exposures; the evaluator writes auditable long-form artifacts and is available from both `run_full` and a standalone CLI.

**Tech Stack:** Python 3.11+, pandas, NumPy, SciPy through pandas Spearman correlation, PyArrow, pytest, Ruff, argparse.

---

## File Structure

- Create `backtesting/strategies/emp008/mfbt_emp008_factor_builders.py`: pure raw-factor calculations and month-end alignment helpers.
- Create `backtesting/strategies/emp008/mfbt_emp008_factor_registry.py`: enums, immutable definitions, validated registries, and lookup helpers.
- Modify `backtesting/strategies/emp008/mfbt_emp008_factors.py`: compatibility facade that builds an ordered factor mapping from the registry.
- Modify `backtesting/strategies/emp008/mfbt_emp008_data.py`: normalize factor-set IDs and derive dataset requirements and forward-snapshot allowance from the registry.
- Create `backtesting/strategies/emp008/mfbt_emp008_factor_pipeline.py`: shared market alignment, preprocessing, benchmark completion, and prepared-factor bundle.
- Modify `backtesting/strategies/emp008/mfbt_emp008.py`: consume the prepared bundle and apply direction-aware expected-alpha constraints.
- Modify `backtesting/strategies/emp008/attribution.py`: consume the same prepared bundle.
- Modify `backtesting/strategies/emp008/mfbt_emp008_experiments/active_weight_factor_plots.py`: use shared preparation instead of duplicating preprocessing.
- Create `backtesting/strategies/emp008/mfbt_emp008_factor_quantiles.py`: deterministic buckets, two weighting modes, returns, IC, metrics, validation, and artifact writer.
- Create `backtesting/strategies/emp008/run_factor_quantiles.py`: standalone CLI.
- Modify `backtesting/strategies/emp008/run_weights.py`: enum-derived choices and simplified configuration.
- Modify `backtesting/strategies/emp008/run_full.py`: prepare once, run quantiles by default, and record artifacts in the run summary.
- Modify `backtesting/strategies/emp008/README.md`: registry extension instructions, timing contract, commands, and outputs.
- Create `tests/strategies/test_emp008_factor_registry.py`: registry and dataset-contract tests.
- Create `tests/strategies/test_emp008_factor_pipeline.py`: preparation parity and shared-exposure tests.
- Create `tests/strategies/test_emp008_factor_quantiles.py`: bucket, weighting, timing, metrics, and artifact tests.
- Create `tests/scripts/test_run_emp008_factor_quantiles.py`: standalone CLI orchestration tests.
- Modify `tests/scripts/test_run_mfbt_emp008_full.py`: configuration, compatibility, and full-run integration tests.
- Modify `tests/strategies/test_mfbt_emp008_experiments.py`: shared-preparation regression expectations.

## Task 1: Introduce the typed factor registry

**Files:**
- Create: `tests/strategies/test_emp008_factor_registry.py`
- Create: `backtesting/strategies/emp008/mfbt_emp008_factor_builders.py`
- Create: `backtesting/strategies/emp008/mfbt_emp008_factor_registry.py`
- Modify: `backtesting/strategies/emp008/mfbt_emp008_factors.py`

- [ ] **Step 1: Write failing registry tests**

Add tests that lock the current public factor names, factor-set order, directions, and builder coverage:

```python
from backtesting.strategies.emp008.mfbt_emp008_factor_registry import (
    FACTOR_DEFINITIONS,
    FactorDirection,
    FactorId,
    FactorSetId,
    factor_definitions_for_set,
)


def test_factor_ids_and_registry_are_unique_and_complete() -> None:
    assert len(FactorId) == len({member.value for member in FactorId})
    assert set(FACTOR_DEFINITIONS) == set(FactorId)


def test_mfbt_factor_order_and_direction_are_explicit() -> None:
    definitions = factor_definitions_for_set(FactorSetId.MFBT)
    assert [definition.id.value for definition in definitions] == [
        "price_momentum",
        "earnings_momentum",
        "dividend_yield",
        "retail_flow",
        "value",
        "ln_market_cap",
    ]
    assert definitions[-1].direction is FactorDirection.LOW
    assert all(definition.direction is FactorDirection.HIGH for definition in definitions[:-1])


def test_origin_variants_keep_existing_model_names() -> None:
    origin = factor_definitions_for_set(FactorSetId.ORIGIN)
    origin_new = factor_definitions_for_set(FactorSetId.ORIGIN_NEW_DIVIDEND)
    assert [item.id.value for item in origin] == ["LnMktcap", "Momentum_12M", "DY"]
    assert [item.id.value for item in origin_new] == ["LnMktcap", "Momentum_12M", "dividend_yield"]
```

- [ ] **Step 2: Run the registry tests and confirm the missing module failure**

Run: `python -m pytest tests/strategies/test_emp008_factor_registry.py -v`

Expected: collection fails with `ModuleNotFoundError: ...mfbt_emp008_factor_registry`.

- [ ] **Step 3: Move pure calculations into the builder module**

Move `month_end_observations`, `align_like_close`, `_monthly_output`, and every current raw-factor function from `mfbt_emp008_factors.py` into `mfbt_emp008_factor_builders.py`. Give the factor entry functions public names while keeping formula helpers private:

```python
def build_price_momentum(market: MarketData, config: MfbtEmp008Config) -> pd.DataFrame:
    del config
    close = market.frames["close"].astype(float)
    trailing_high = close.rolling(252, min_periods=252).max()
    ratio = close.divide(trailing_high.where(trailing_high.gt(0.0)))
    return _monthly_output(close, month_end_observations(ratio))


def build_ln_market_cap(market: MarketData, config: MfbtEmp008Config) -> pd.DataFrame:
    del config
    close = market.frames["close"].astype(float)
    market_cap = align_like_close(market, "market_cap").astype(float)
    values = np.log(month_end_observations(market_cap).where(lambda frame: frame.gt(0.0)))
    return _monthly_output(close, values)
```

Use `TYPE_CHECKING` for the configuration import so `mfbt_emp008_data.py` can import the registry without a runtime cycle:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mfbt_emp008_data import MfbtEmp008Config
```

- [ ] **Step 4: Implement the registry and validation**

Define the enum and metadata contract, then instantiate all current definitions and sets:

```python
@unique
class FactorId(StrEnum):
    PRICE_MOMENTUM = "price_momentum"
    POSITIVITY_MOMENTUM = "positivity_momentum"
    EARNINGS_MOMENTUM = "earnings_momentum"
    DIVIDEND_YIELD = "dividend_yield"
    RETAIL_FLOW = "retail_flow"
    VALUE = "value"
    LN_MARKET_CAP = "ln_market_cap"
    ORIGIN_LN_MARKET_CAP = "LnMktcap"
    ORIGIN_MOMENTUM_12M = "Momentum_12M"
    ORIGIN_DIVIDEND_YIELD = "DY"


@unique
class FactorSetId(StrEnum):
    MFBT = "mfbt"
    MFBT_POS = "mfbt_pos"
    MFBT_ORIGIN_SMALLCAP = "mfbt_origin_smallcap"
    ORIGIN = "origin"
    ORIGIN_NEW_DIVIDEND = "origin_new_dividend"


@unique
class FactorDirection(StrEnum):
    HIGH = "high"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class FactorDefinition:
    id: FactorId
    builder: FactorBuilder
    datasets: tuple[DatasetId, ...] = ()
    direction: FactorDirection = FactorDirection.HIGH
    rank_transform: bool = False
    winsor_config_attr: str | None = None
    zscore_cap_config_attr: str | None = None
    neutralize_large_benchmark_weight: bool = False
    requires_construction_sector: bool = False


@dataclass(frozen=True, slots=True)
class FactorSetDefinition:
    id: FactorSetId
    factors: tuple[FactorId, ...]
    constrain_expected_alpha_to_direction: bool = False
    snapshot_forward_days: int = 0
```

Define the callable type and the complete current registry explicitly:

```python
FactorBuilder = Callable[[MarketData, "MfbtEmp008Config"], pd.DataFrame]

_FACTOR_DEFINITIONS = {
    FactorId.PRICE_MOMENTUM: FactorDefinition(FactorId.PRICE_MOMENTUM, build_price_momentum),
    FactorId.POSITIVITY_MOMENTUM: FactorDefinition(FactorId.POSITIVITY_MOMENTUM, build_positivity_momentum),
    FactorId.EARNINGS_MOMENTUM: FactorDefinition(
        FactorId.EARNINGS_MOMENTUM,
        build_earnings_momentum,
        datasets=(DatasetId.QW_OP_FWD_12M,),
    ),
    FactorId.DIVIDEND_YIELD: FactorDefinition(
        FactorId.DIVIDEND_YIELD,
        build_dividend_yield,
        datasets=(DatasetId.QW_DPS_TTM,),
    ),
    FactorId.RETAIL_FLOW: FactorDefinition(
        FactorId.RETAIL_FLOW,
        build_retail_flow,
        datasets=(DatasetId.QW_RETAIL,),
        requires_construction_sector=True,
    ),
    FactorId.VALUE: FactorDefinition(
        FactorId.VALUE,
        build_value,
        datasets=(
            DatasetId.QW_FCF,
            DatasetId.QW_INT_BEARING_LIAB_NFQ0,
            DatasetId.QW_QUICK_ASSETS_NFQ0,
        ),
        winsor_config_attr="value_raw_winsor_quantile",
        zscore_cap_config_attr="value_zscore_cap",
    ),
    FactorId.LN_MARKET_CAP: FactorDefinition(
        FactorId.LN_MARKET_CAP,
        build_ln_market_cap,
        direction=FactorDirection.LOW,
        rank_transform=True,
        neutralize_large_benchmark_weight=True,
    ),
    FactorId.ORIGIN_LN_MARKET_CAP: FactorDefinition(
        FactorId.ORIGIN_LN_MARKET_CAP,
        build_ln_market_cap,
        direction=FactorDirection.LOW,
        rank_transform=True,
    ),
    FactorId.ORIGIN_MOMENTUM_12M: FactorDefinition(
        FactorId.ORIGIN_MOMENTUM_12M,
        build_origin_momentum_12m,
    ),
    FactorId.ORIGIN_DIVIDEND_YIELD: FactorDefinition(
        FactorId.ORIGIN_DIVIDEND_YIELD,
        build_origin_dividend_yield,
        datasets=(DatasetId.QW_DIVIDEND_YLD_FY0,),
    ),
}

_FACTOR_SET_DEFINITIONS = {
    FactorSetId.MFBT: FactorSetDefinition(
        FactorSetId.MFBT,
        (
            FactorId.PRICE_MOMENTUM,
            FactorId.EARNINGS_MOMENTUM,
            FactorId.DIVIDEND_YIELD,
            FactorId.RETAIL_FLOW,
            FactorId.VALUE,
            FactorId.LN_MARKET_CAP,
        ),
    ),
    FactorSetId.MFBT_POS: FactorSetDefinition(
        FactorSetId.MFBT_POS,
        (
            FactorId.POSITIVITY_MOMENTUM,
            FactorId.EARNINGS_MOMENTUM,
            FactorId.DIVIDEND_YIELD,
            FactorId.RETAIL_FLOW,
            FactorId.VALUE,
            FactorId.LN_MARKET_CAP,
        ),
    ),
    FactorSetId.MFBT_ORIGIN_SMALLCAP: FactorSetDefinition(
        FactorSetId.MFBT_ORIGIN_SMALLCAP,
        (
            FactorId.PRICE_MOMENTUM,
            FactorId.EARNINGS_MOMENTUM,
            FactorId.DIVIDEND_YIELD,
            FactorId.RETAIL_FLOW,
            FactorId.VALUE,
            FactorId.LN_MARKET_CAP,
        ),
        constrain_expected_alpha_to_direction=True,
    ),
    FactorSetId.ORIGIN: FactorSetDefinition(
        FactorSetId.ORIGIN,
        (FactorId.ORIGIN_LN_MARKET_CAP, FactorId.ORIGIN_MOMENTUM_12M, FactorId.ORIGIN_DIVIDEND_YIELD),
        constrain_expected_alpha_to_direction=True,
        snapshot_forward_days=7,
    ),
    FactorSetId.ORIGIN_NEW_DIVIDEND: FactorSetDefinition(
        FactorSetId.ORIGIN_NEW_DIVIDEND,
        (FactorId.ORIGIN_LN_MARKET_CAP, FactorId.ORIGIN_MOMENTUM_12M, FactorId.DIVIDEND_YIELD),
        constrain_expected_alpha_to_direction=True,
    ),
}

FACTOR_DEFINITIONS = MappingProxyType(_FACTOR_DEFINITIONS)
FACTOR_SET_DEFINITIONS = MappingProxyType(_FACTOR_SET_DEFINITIONS)
```

Validate `set(_FACTOR_DEFINITIONS) == set(FactorId)`, unique factor membership references, and non-empty sets at import. `get_factor_definition`, `get_factor_set_definition`, `factor_definitions_for_set`, and `factor_set_values` normalize strings once and raise a message listing supported values:

```python
def parse_factor_set(value: FactorSetId | str) -> FactorSetId:
    try:
        return value if isinstance(value, FactorSetId) else FactorSetId(value)
    except ValueError as exc:
        supported = ", ".join(member.value for member in FactorSetId)
        raise ValueError(f"unsupported factor_set {value!r}; choose one of: {supported}") from exc
```

- [ ] **Step 5: Replace factor-set branches with registry iteration**

Keep the existing import surface while making construction generic:

```python
def build_raw_mfbt_factors(
    market: MarketData,
    config: MfbtEmp008Config,
) -> dict[str, pd.DataFrame]:
    return {
        definition.id.value: definition.builder(market, config)
        for definition in factor_definitions_for_set(config.factor_set)
    }


__all__ = ["align_like_close", "build_raw_mfbt_factors", "month_end_observations"]
```

- [ ] **Step 6: Run focused and existing factor regression tests**

Run:

```powershell
python -m pytest tests/strategies/test_emp008_factor_registry.py tests/scripts/test_run_mfbt_emp008_full.py -v
python -m ruff check backtesting/strategies/emp008/mfbt_emp008_factor_builders.py backtesting/strategies/emp008/mfbt_emp008_factor_registry.py backtesting/strategies/emp008/mfbt_emp008_factors.py tests/strategies/test_emp008_factor_registry.py
```

Expected: all tests pass and Ruff prints `All checks passed!`.

- [ ] **Step 7: Commit the registry slice**

```powershell
git add backtesting/strategies/emp008/mfbt_emp008_factor_builders.py backtesting/strategies/emp008/mfbt_emp008_factor_registry.py backtesting/strategies/emp008/mfbt_emp008_factors.py tests/strategies/test_emp008_factor_registry.py
git commit -m "Make new EMP008 factors declare their full contract once" -m "Introduce unique typed identifiers and immutable factor-set definitions while preserving every current model-facing name and formula." -m "Constraint: Existing Origin and MFBT factor names remain output-compatible" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: Registry and existing EMP008 factor regression tests; Ruff"
```

## Task 2: Derive configuration and datasets from the registry

**Files:**
- Modify: `tests/strategies/test_emp008_factor_registry.py`
- Modify: `tests/scripts/test_run_mfbt_emp008_full.py`
- Modify: `backtesting/strategies/emp008/mfbt_emp008_data.py`
- Modify: `backtesting/strategies/emp008/run_weights.py`
- Modify: `backtesting/strategies/emp008/run_full.py`

- [ ] **Step 1: Write failing dataset and normalization tests**

```python
@pytest.mark.parametrize(
    ("factor_set", "present", "absent"),
    [
        (FactorSetId.ORIGIN, DatasetId.QW_DIVIDEND_YLD_FY0, DatasetId.QW_DPS_TTM),
        (FactorSetId.ORIGIN_NEW_DIVIDEND, DatasetId.QW_DPS_TTM, DatasetId.QW_DIVIDEND_YLD_FY0),
        (FactorSetId.MFBT, DatasetId.QW_FCF, DatasetId.QW_DIVIDEND_YLD_FY0),
    ],
)
def test_required_datasets_follow_registered_factor_dependencies(
    factor_set: FactorSetId,
    present: DatasetId,
    absent: DatasetId,
) -> None:
    datasets = required_datasets(MfbtEmp008Config(factor_set=factor_set))
    assert present in datasets
    assert absent not in datasets


def test_config_normalizes_compatible_factor_set_string() -> None:
    config = MfbtEmp008Config(factor_set="origin")
    assert config.factor_set is FactorSetId.ORIGIN


def test_factor_set_choices_are_registry_derived() -> None:
    action = next(action for action in run_full._parser()._actions if action.dest == "factor_set")
    assert tuple(action.choices) == tuple(member.value for member in FactorSetId)
```

- [ ] **Step 2: Run the focused tests and confirm failures against string branches**

Run: `python -m pytest tests/strategies/test_emp008_factor_registry.py tests/scripts/test_run_mfbt_emp008_full.py -v`

Expected: the new enum normalization or registry-derived choice assertions fail before implementation.

- [ ] **Step 3: Normalize configuration and remove duplicated factor lists**

Change `MfbtEmp008Config.factor_set` to `FactorSetId` and normalize frozen dataclass construction:

```python
factor_set: FactorSetId = FactorSetId.MFBT

def __post_init__(self) -> None:
    object.__setattr__(self, "factor_set", parse_factor_set(self.factor_set))
```

Remove `rank_transform_factors`, `large_bm_neutral_factor_names`, `expected_alpha_policy`, and `monthly_snapshot_forward_days`; their behavior now belongs to registry definitions. Keep numeric experiment overrides such as `value_raw_winsor_quantile`, `value_zscore_cap`, and `large_bm_neutral_weight_threshold`.

Build `required_datasets` from common strategy inputs plus registered factor dependencies. Add `config.sector_dataset` only when a selected definition has `requires_construction_sector`, and always add the optimizer's selected neutral sector:

```python
definitions = factor_definitions_for_set(config.factor_set)
factor_datasets = [dataset for definition in definitions for dataset in definition.datasets]
if any(definition.requires_construction_sector for definition in definitions):
    factor_datasets.append(config.sector_dataset)
neutral_dataset = config.sector_neutral_dataset or config.sector_dataset
ordered = [
    DatasetId.QW_ADJ_C,
    config.bm_weights_dataset,
    *factor_datasets,
    neutral_dataset,
    DatasetId.QW_MKTCAP,
    config.float_market_cap_dataset,
    config.universe_dataset,
]
return tuple(dict.fromkeys(ordered))
```

Use `get_factor_set_definition(config.factor_set).snapshot_forward_days` in `padded_snapshot_end` and `_trim_non_forward_snapshot_frames`.

- [ ] **Step 4: Simplify config construction and CLI choices**

Replace the factor-set policy branch in `build_emp008_config` with:

```python
if factor_set is not None:
    config = replace(config, factor_set=parse_factor_set(factor_set))
```

Use `factor_set_values()` for `--factor-set` choices in `run_weights.py` and `run_full.py`. Serialize `config.factor_set.value` in logs and JSON payloads.

- [ ] **Step 5: Run tests and static checks**

Run:

```powershell
python -m pytest tests/strategies/test_emp008_factor_registry.py tests/scripts/test_run_mfbt_emp008_full.py -v
python -m ruff check backtesting/strategies/emp008/mfbt_emp008_data.py backtesting/strategies/emp008/run_weights.py backtesting/strategies/emp008/run_full.py tests/strategies/test_emp008_factor_registry.py tests/scripts/test_run_mfbt_emp008_full.py
```

Expected: all pass.

- [ ] **Step 6: Commit the configuration slice**

```powershell
git add backtesting/strategies/emp008/mfbt_emp008_data.py backtesting/strategies/emp008/run_weights.py backtesting/strategies/emp008/run_full.py tests/strategies/test_emp008_factor_registry.py tests/scripts/test_run_mfbt_emp008_full.py
git commit -m "Prevent EMP008 factor metadata from drifting across entry points" -m "Derive dataset requirements, variant behavior, and CLI choices from the typed registry instead of repeated string conditionals." -m "Constraint: Origin must continue loading only its actual dividend source" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: Registry, config, dataset, parser, and EMP008 regression tests; Ruff"
```

## Task 3: Share prepared model exposures across consumers

**Files:**
- Create: `tests/strategies/test_emp008_factor_pipeline.py`
- Create: `backtesting/strategies/emp008/mfbt_emp008_factor_pipeline.py`
- Modify: `backtesting/strategies/emp008/mfbt_emp008.py`
- Modify: `backtesting/strategies/emp008/attribution.py`
- Modify: `backtesting/strategies/emp008/mfbt_emp008_experiments/active_weight_factor_plots.py`
- Modify: `tests/scripts/test_run_mfbt_emp008_full.py`
- Modify: `tests/strategies/test_mfbt_emp008_experiments.py`

- [ ] **Step 1: Write failing preparation tests**

Use a small `MarketData` fixture with two month ends, an in/out-universe ticker, benchmark weights, float and total market cap, and a monkeypatched raw builder. Assert registry preprocessing and shared fields:

```python
def test_prepare_emp008_factors_applies_registered_preprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "build_raw_mfbt_factors", lambda market, config: {"ln_market_cap": RAW})
    prepared = pipeline.prepare_emp008_factors(MARKET, MfbtEmp008Config())
    expected = preprocess_factor_frame(RAW, FLOAT_CAP, UNIVERSE, rank_transform=True)
    expected = expected.mask(BM_WEIGHTS.ge(0.10), 0.0)
    pd.testing.assert_frame_equal(prepared.alpha_factors["ln_market_cap"], expected)
    assert prepared.monthly_dates == tuple(RAW.dropna(how="all").index)


def test_optimizer_and_attribution_accept_the_same_prepared_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = make_prepared_bundle()
    monkeypatch.setattr(emp008, "load_and_prepare_emp008_factors", Mock(side_effect=AssertionError("reloaded")))
    monkeypatch.setattr(attribution, "load_and_prepare_emp008_factors", Mock(side_effect=AssertionError("reloaded")))
    emp008.run_mfbt_emp008(parquet_dir=Path("unused"), start="2024-02-29", end="2024-02-29", prepared=prepared)
    attribution.build_emp008_factor_attribution(
        parquet_dir=Path("unused"), run_root=RUN_ROOT, config=prepared.config, prepared=prepared
    )
```

Patch optimizer/attribution internals in the second test so it proves no data reload and object identity without requiring 36 months of synthetic risk history.

- [ ] **Step 2: Run preparation tests and confirm the missing API failures**

Run: `python -m pytest tests/strategies/test_emp008_factor_pipeline.py -v`

Expected: collection fails because the pipeline module and `prepared` parameters do not exist.

- [ ] **Step 3: Implement the immutable prepared bundle**

```python
@dataclass(frozen=True, slots=True)
class PreparedEmp008Factors:
    config: MfbtEmp008Config
    market: MarketData
    factor_set: FactorSetDefinition
    raw_factors: dict[str, pd.DataFrame]
    alpha_factors: dict[str, pd.DataFrame]
    sector_factors: dict[str, pd.DataFrame]
    close: pd.DataFrame
    market_cap: pd.DataFrame
    float_market_cap: pd.DataFrame
    universe: pd.DataFrame
    sector: pd.DataFrame
    benchmark_weights: pd.DataFrame
    monthly_dates: tuple[pd.Timestamp, ...]
```

Implement `prepare_emp008_factors(market, config)` and `load_and_prepare_emp008_factors(parquet_dir, start, end, config)`. For each selected definition, pass `rank_transform`, `getattr(config, winsor_config_attr)`, and `getattr(config, zscore_cap_config_attr)` into `preprocess_factor_frame`; then apply large-benchmark neutralization only where its metadata is true.

Move benchmark completion and common-month selection into this module with public names. Re-export aliases from `mfbt_emp008.py` so current internal experiment imports and the benchmark regression test remain compatible:

```python
from .mfbt_emp008_factor_pipeline import (
    common_month_end_dates as _common_month_end_dates,
    complete_benchmark_history as _complete_benchmark_history,
    neutralize_large_benchmark_weight_exposures as _neutralize_large_benchmark_weight_factor_exposures,
)
```

- [ ] **Step 4: Refactor optimizer and direction constraints**

Add `prepared: PreparedEmp008Factors | None = None` to `run_mfbt_emp008`; load only when absent, then replace local preparation variables with bundle fields.

Replace policy-name branching with registry direction enforcement:

```python
def _apply_expected_alpha_policy(expected_alpha: pd.Series, config: MfbtEmp008Config) -> pd.Series:
    factor_set = get_factor_set_definition(config.factor_set)
    if not factor_set.constrain_expected_alpha_to_direction:
        return expected_alpha
    adjusted = expected_alpha.copy()
    for definition in factor_definitions_for_set(factor_set.id):
        name = definition.id.value
        if name not in adjusted:
            continue
        if definition.direction is FactorDirection.HIGH and adjusted.loc[name] < 0.0:
            adjusted.loc[name] = 0.0
        if definition.direction is FactorDirection.LOW and adjusted.loc[name] > 0.0:
            adjusted.loc[name] = 0.0
    return adjusted
```

- [ ] **Step 5: Refactor attribution and active-weight research**

Add the same optional prepared bundle to `build_emp008_factor_attribution`. When supplied, use its factors, close, sector exposures, and dates; otherwise prepare normally. Replace the duplicate preprocessing block in `active_weight_factor_plots.py` with `prepare_emp008_factors(market, config)` and use `prepared.alpha_factors`, `prepared.sector_factors`, and `prepared.monthly_dates`.

- [ ] **Step 6: Run preparation, optimizer, attribution, and experiment regressions**

Run:

```powershell
python -m pytest tests/strategies/test_emp008_factor_pipeline.py tests/scripts/test_run_mfbt_emp008_full.py tests/strategies/test_mfbt_emp008_experiments.py -v
python -m ruff check backtesting/strategies/emp008/mfbt_emp008_factor_pipeline.py backtesting/strategies/emp008/mfbt_emp008.py backtesting/strategies/emp008/attribution.py backtesting/strategies/emp008/mfbt_emp008_experiments/active_weight_factor_plots.py tests/strategies/test_emp008_factor_pipeline.py
```

Expected: all pass.

- [ ] **Step 7: Commit the shared pipeline slice**

```powershell
git add backtesting/strategies/emp008/mfbt_emp008_factor_pipeline.py backtesting/strategies/emp008/mfbt_emp008.py backtesting/strategies/emp008/attribution.py backtesting/strategies/emp008/mfbt_emp008_experiments/active_weight_factor_plots.py tests/strategies/test_emp008_factor_pipeline.py tests/scripts/test_run_mfbt_emp008_full.py tests/strategies/test_mfbt_emp008_experiments.py
git commit -m "Keep every EMP008 analysis on the same model exposures" -m "Prepare factors once and share the resulting bundle across optimization, attribution, and research consumers." -m "Constraint: Preprocessing order and Origin sign behavior must remain unchanged" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: Preparation, optimizer, attribution, experiment, and compatibility tests; Ruff"
```

## Task 4: Build deterministic equal- and market-cap-weight quantiles

**Files:**
- Create: `tests/strategies/test_emp008_factor_quantiles.py`
- Create: `backtesting/strategies/emp008/mfbt_emp008_factor_quantiles.py`

- [ ] **Step 1: Write failing bucket and weighting tests**

Construct six tickers with duplicate signals and intentionally non-uniform caps:

```python
def test_weighting_modes_share_buckets_but_not_weights() -> None:
    result = evaluate_factor_quantiles(
        factors={"value": EXPOSURES},
        directions={"value": FactorDirection.HIGH},
        close=CLOSE,
        market_cap=MARKET_CAP,
        universe=UNIVERSE,
        start="2024-02-29",
        end="2024-02-29",
        q=3,
    )
    weights = result.portfolio_weights
    equal = weights.loc[weights["weighting"].eq("equal_weight")]
    cap = weights.loc[weights["weighting"].eq("market_cap_weight")]
    assert set(map(tuple, equal[["factor", "signal_date", "quantile", "ticker"]].to_numpy())) == set(
        map(tuple, cap[["factor", "signal_date", "quantile", "ticker"]].to_numpy())
    )
    assert equal.groupby(["signal_date", "factor", "quantile"])["weight"].sum().eq(1.0).all()
    assert cap.groupby(["signal_date", "factor", "quantile"])["weight"].sum().eq(1.0).all()
    q1_cap = cap.loc[cap["quantile"].eq("q1")].set_index("ticker")["weight"]
    assert q1_cap.loc["B"] / q1_cap.loc["A"] == pytest.approx(
        MARKET_CAP.loc[pd.Timestamp("2024-01-31"), "B"] / MARKET_CAP.loc[pd.Timestamp("2024-01-31"), "A"]
    )
```

Add separate tests asserting:

- duplicate signals use ticker as a stable tie-break;
- every eligible ticker appears exactly once per factor/date/weighting;
- out-of-universe, invalid-price, and non-positive-cap names are absent from both modes;
- signal-date cap, never return-date cap, determines weights;
- Q1 is lowest and Q5 is highest for a five-bucket sample;
- fewer than `q` eligible names leave later buckets absent without duplication;
- invalid `q` raises `ValueError("q must be at least 2")`.
- an evaluation with no eligible factor-month raises `ValueError` containing the factor names and requested date range;
- one sparse factor-month does not remove valid observations from other factors.

- [ ] **Step 2: Run the quantile tests and confirm the missing module failure**

Run: `python -m pytest tests/strategies/test_emp008_factor_quantiles.py -v`

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Define result and weighting contracts**

```python
@unique
class QuantileWeighting(StrEnum):
    EQUAL = "equal_weight"
    MARKET_CAP = "market_cap_weight"


@dataclass(frozen=True, slots=True)
class Emp008FactorQuantileResult:
    monthly_returns: pd.DataFrame
    portfolio_weights: pd.DataFrame
    rank_ic: pd.DataFrame
    cumulative_returns: pd.DataFrame
    summary: pd.DataFrame
```

Keep `QuantileWeighting` in this module because it is an evaluation concern, while factor and set identifiers remain in the registry.

Define the public prepared-bundle wrapper and the testable frame-level evaluator with stable signatures:

```python
def run_emp008_factor_quantiles(
    *,
    prepared: PreparedEmp008Factors,
    start: str,
    end: str,
    q: int = 5,
) -> Emp008FactorQuantileResult:
    directions = {
        definition.id.value: definition.direction
        for definition in factor_definitions_for_set(prepared.factor_set.id)
    }
    return evaluate_factor_quantiles(
        factors=prepared.alpha_factors,
        directions=directions,
        close=prepared.close,
        market_cap=prepared.market_cap,
        universe=prepared.universe,
        start=start,
        end=end,
        q=q,
    )
```

Implement the frame-level evaluator in Step 5 with the exact signature `evaluate_factor_quantiles(*, factors: Mapping[str, pd.DataFrame], directions: Mapping[str, FactorDirection], close: pd.DataFrame, market_cap: pd.DataFrame, universe: pd.DataFrame, start: str, end: str, q: int = 5) -> Emp008FactorQuantileResult`. During this construction slice, return the completed `monthly_returns`, `portfolio_weights`, and `rank_ic` frames with empty typed `cumulative_returns` and `summary` frames; Task 5 fills those two derived artifacts without changing either public signature.

- [ ] **Step 4: Implement common eligibility and deterministic buckets**

```python
def _bucket_members(
    signal: pd.Series,
    eligible: pd.Series,
    q: int,
) -> dict[str, pd.Index]:
    ranked = pd.DataFrame(
        {
            "signal": signal.loc[eligible].astype(float),
            "ticker_key": [str(value) for value in signal.index[eligible]],
        },
        index=signal.index[eligible],
    ).sort_values(["signal", "ticker_key"], kind="mergesort")
    groups = np.array_split(ranked.index.to_numpy(), min(q, len(ranked)))
    return {f"q{number}": pd.Index(group) for number, group in enumerate(groups, start=1) if len(group)}
```

Build common eligibility from signal-date membership, finite signal, both finite prices, positive signal-date price, and positive finite signal-date total market cap. Construct the bucket map once and reuse it for both weighting modes.

After all loops, raise a descriptive error only when `return_rows` is empty. Include the selected factor names, `start`, and `end` in the message; otherwise retain every valid row from non-empty factors.

- [ ] **Step 5: Implement returns and weight rows**

For every consecutive prepared month pair and every factor, calculate next-month constituent returns. Emit normalized rows with explicit fields:

```python
return_rows.append(
    {
        "signal_date": signal_date,
        "return_date": return_date,
        "factor": factor_name,
        "weighting": weighting.value,
        "portfolio": quantile_name,
        "return": float((constituent_returns * weights).sum()),
        "constituent_count": int(len(tickers)),
    }
)
```

Only emit pairs whose `return_date` falls inclusively between `start` and `end`. For each weighting mode append `high_minus_low` and `preferred_minus_avoided` rows after Q1 and Q5 exist. Give spread rows the sum of both leg counts. Calculate monthly Spearman IC once per factor/date and multiply by `+1` for HIGH or `-1` for LOW to produce `directional_rank_ic`.

- [ ] **Step 6: Run focused tests and Ruff**

Run:

```powershell
python -m pytest tests/strategies/test_emp008_factor_quantiles.py -v
python -m ruff check backtesting/strategies/emp008/mfbt_emp008_factor_quantiles.py tests/strategies/test_emp008_factor_quantiles.py
```

Expected: all construction tests pass and Ruff passes.

- [ ] **Step 7: Commit the construction slice**

```powershell
git add backtesting/strategies/emp008/mfbt_emp008_factor_quantiles.py tests/strategies/test_emp008_factor_quantiles.py
git commit -m "Expose each EMP008 factor through comparable quantile portfolios" -m "Use one deterministic membership assignment for both equal and signal-date market-cap weights, then retain raw and preferred-direction spreads." -m "Constraint: Portfolio construction must not use return-date information" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: Bucket, eligibility, timing, weight, return, spread, and IC unit tests; Ruff"
```

## Task 5: Add monthly metrics and auditable artifacts

**Files:**
- Modify: `tests/strategies/test_emp008_factor_quantiles.py`
- Modify: `backtesting/strategies/emp008/mfbt_emp008_factor_quantiles.py`

- [ ] **Step 1: Write failing metric and writer tests**

```python
def test_monthly_summary_uses_twelve_period_annualization() -> None:
    returns = pd.Series([0.01] * 12, index=pd.date_range("2024-01-31", periods=12, freq="ME"))
    metrics = summarize_monthly_returns(returns)
    assert metrics["annualized_return"] == pytest.approx(1.01**12 - 1.0)
    assert metrics["annualized_volatility"] == pytest.approx(0.0)
    assert metrics["max_drawdown"] == pytest.approx(0.0)
    assert metrics["positive_month_rate"] == pytest.approx(1.0)


def test_write_outputs_emits_documented_schema(tmp_path: Path) -> None:
    result = make_quantile_result()
    payload = result.write_outputs(tmp_path, factor_set=FactorSetId.MFBT, q=5)
    assert set(payload) >= {"monthly_returns_parquet", "weights_parquet", "rank_ic_parquet", "summary_json", "manifest"}
    assert (tmp_path / "monthly_returns.csv").is_file()
    assert (tmp_path / "monthly_returns.parquet").is_file()
    assert (tmp_path / "portfolio_weights.parquet").is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["weighting_modes"] == ["equal_weight", "market_cap_weight"]
    assert manifest["market_cap_field"] == "market_cap"
    assert manifest["timing"] == "month_end_t_to_next_month_end"
```

Add assertions for missing periods, maximum drawdown, non-zero volatility Sharpe, one-way turnover, IC mean/IR/positive rate, and HIGH/LOW monotonicity.

- [ ] **Step 2: Run the new tests and confirm missing metrics/output failures**

Run: `python -m pytest tests/strategies/test_emp008_factor_quantiles.py -v`

Expected: failures name `summarize_monthly_returns` and `write_outputs`.

- [ ] **Step 3: Implement monthly performance metrics**

```python
def summarize_monthly_returns(returns: pd.Series) -> dict[str, float | int]:
    clean = returns.astype(float).dropna()
    if clean.empty:
        return _empty_monthly_metrics()
    equity = clean.add(1.0).cumprod()
    years = len(clean) / 12.0
    growth = float(equity.iloc[-1])
    annualized_return = growth ** (1.0 / years) - 1.0 if growth > 0.0 else -1.0
    volatility = float(clean.std(ddof=0) * np.sqrt(12.0))
    sharpe = float(clean.mean() / clean.std(ddof=0) * np.sqrt(12.0)) if clean.std(ddof=0) > 0.0 else 0.0
    drawdown = equity.divide(equity.cummax()).sub(1.0)
    return {
        "observations": int(len(clean)),
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "positive_month_rate": float(clean.gt(0.0).mean()),
        "mean_monthly_return": float(clean.mean()),
    }
```

Build summaries by factor, weighting, and portfolio. Join factor-level IC statistics and direction-aware monotonicity. Compute one-way turnover from the long-form weight pivot as `0.5 * abs(w_t - w_t-1).sum(axis=1)`, excluding the first observation. Do not fill missing monthly returns with zero.

- [ ] **Step 4: Implement cumulative returns and atomic validation-before-write**

Create long-form cumulative results grouped by factor/weighting/portfolio. Before writing, validate required columns, finite non-empty bucket weights, unit sums within tolerance `1e-10`, and identical membership between the two weighting modes.

Implement `write_outputs(output_dir, factor_set, q)` using `mkdir(parents=True, exist_ok=True)`, CSV with `index=False`, Parquet with `engine="pyarrow"`, UTF-8 JSON, and a manifest containing ordered factors and directions from the registry. Return a compact JSON-serializable payload with paths and row counts.

- [ ] **Step 5: Run all quantile tests and static checks**

Run:

```powershell
python -m pytest tests/strategies/test_emp008_factor_quantiles.py -v
python -m ruff check backtesting/strategies/emp008/mfbt_emp008_factor_quantiles.py tests/strategies/test_emp008_factor_quantiles.py
```

Expected: all pass.

- [ ] **Step 6: Commit the artifact slice**

```powershell
git add backtesting/strategies/emp008/mfbt_emp008_factor_quantiles.py tests/strategies/test_emp008_factor_quantiles.py
git commit -m "Make EMP008 factor evidence reproducible outside the optimizer" -m "Add monthly metrics, turnover, IC diagnostics, cumulative returns, invariant checks, and long-form artifact output." -m "Constraint: Monthly diagnostics annualize with twelve periods and retain missing observations" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: Metric, turnover, monotonicity, schema, manifest, and writer tests; Ruff"
```

## Task 6: Add the standalone factor-quantile CLI

**Files:**
- Create: `tests/scripts/test_run_emp008_factor_quantiles.py`
- Create: `backtesting/strategies/emp008/run_factor_quantiles.py`

- [ ] **Step 1: Write failing parser and orchestration tests**

```python
def test_parser_exposes_registry_factor_sets_and_quantile_count() -> None:
    parser = run_factor_quantiles._parser()
    factor_action = next(action for action in parser._actions if action.dest == "factor_set")
    assert tuple(factor_action.choices) == tuple(member.value for member in FactorSetId)
    args = parser.parse_args(["--factor-set", "origin", "--quantiles", "4"])
    assert args.factor_set == "origin"
    assert args.quantiles == 4


def test_main_prepares_once_and_writes_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prepared = make_prepared_bundle()
    result = make_quantile_result()
    prepare = Mock(return_value=prepared)
    evaluate = Mock(return_value=result)
    write = Mock(return_value={"summary_csv": str(tmp_path / "summary.csv")})
    monkeypatch.setattr(run_factor_quantiles, "load_and_prepare_emp008_factors", prepare)
    monkeypatch.setattr(run_factor_quantiles, "run_emp008_factor_quantiles", evaluate)
    monkeypatch.setattr(type(result), "write_outputs", write)
    run_factor_quantiles.main(["--end", "2024-06-30", "--output-dir", str(tmp_path)])
    prepare.assert_called_once()
    evaluate.assert_called_once_with(prepared=prepared, start=DEFAULT_START, end="2024-06-30", q=5)
    write.assert_called_once()
```

- [ ] **Step 2: Run CLI tests and confirm the missing module failure**

Run: `python -m pytest tests/scripts/test_run_emp008_factor_quantiles.py -v`

Expected: collection fails because `run_factor_quantiles.py` does not exist.

- [ ] **Step 3: Implement the standalone command**

The parser accepts `--start`, `--end`, `--parquet-dir`, `--output-dir`, `--factor-set`, `--sector-neutral-dataset`, and `--quantiles`. Build config through `build_emp008_config`, resolve omitted end with `latest_common_end`, load and prepare once, evaluate, write, and print the returned payload as indented Korean-safe JSON:

```python
prepared = load_and_prepare_emp008_factors(
    parquet_dir=args.parquet_dir,
    start=args.start,
    end=end,
    config=config,
)
result = run_emp008_factor_quantiles(prepared=prepared, start=args.start, end=end, q=args.quantiles)
payload = result.write_outputs(args.output_dir, factor_set=config.factor_set, q=args.quantiles)
print(json.dumps(payload, ensure_ascii=False, indent=2))
```

- [ ] **Step 4: Run CLI tests, help smoke, and Ruff**

Run:

```powershell
python -m pytest tests/scripts/test_run_emp008_factor_quantiles.py -v
python -m backtesting.strategies.emp008.run_factor_quantiles --help
python -m ruff check backtesting/strategies/emp008/run_factor_quantiles.py tests/scripts/test_run_emp008_factor_quantiles.py
```

Expected: tests pass, help exits zero and lists both weighting modes in the description, Ruff passes.

- [ ] **Step 5: Commit the CLI slice**

```powershell
git add backtesting/strategies/emp008/run_factor_quantiles.py tests/scripts/test_run_emp008_factor_quantiles.py
git commit -m "Let factor research run without the production portfolio backtest" -m "Expose the registered EMP008 quantile evaluator through a focused standalone command with the same date and variant semantics as the main runner." -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: Parser and orchestration tests; CLI help smoke; Ruff"
```

## Task 7: Integrate quantiles into `run_full` and document extension

**Files:**
- Modify: `tests/scripts/test_run_mfbt_emp008_full.py`
- Modify: `backtesting/strategies/emp008/run_full.py`
- Modify: `backtesting/strategies/emp008/README.md`

- [ ] **Step 1: Write failing full-run integration tests**

Extend the existing mocked full-run test so it supplies one prepared bundle to all consumers and checks default quantile output:

```python
def test_full_run_prepares_once_and_runs_factor_quantiles_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prepared = make_prepared_bundle()
    prepare = Mock(return_value=prepared)
    optimizer = Mock(return_value=make_emp008_result())
    quantiles = Mock(return_value=make_quantile_result())
    monkeypatch.setattr(run_full, "load_and_prepare_emp008_factors", prepare)
    monkeypatch.setattr(run_full, "run_mfbt_emp008", optimizer)
    monkeypatch.setattr(run_full, "run_emp008_factor_quantiles", quantiles)
    patch_backtest_report_and_attribution(monkeypatch, tmp_path)
    run_full.main(["--end", "2024-06-30", "--output-root", str(tmp_path), "--no-comparison"])
    prepare.assert_called_once()
    assert optimizer.call_args.kwargs["prepared"] is prepared
    assert quantiles.call_args.kwargs["prepared"] is prepared
    summary = json.loads((tmp_path / "mfbt_emp008" / "run_summary.json").read_text(encoding="utf-8"))
    assert "factor_quantiles" in summary
```

Add `test_full_run_can_skip_factor_quantiles` using `--no-factor-quantiles` and assert the evaluator is not called and the summary key is absent.

- [ ] **Step 2: Run the integration tests and confirm missing parser/orchestration behavior**

Run: `python -m pytest tests/scripts/test_run_mfbt_emp008_full.py -v`

Expected: new tests fail because `run_full` neither prepares once nor invokes quantiles.

- [ ] **Step 3: Prepare once and integrate the evaluator**

At the start of the `try` block, create the prepared bundle in a timed `factor_preparation` stage. Pass it to optimization and attribution. After optimization and before the production backtest, run quantile evaluation unless opted out:

```python
if not args.no_factor_quantiles:
    with timed(logger, "factor_quantiles"):
        quantile_result = run_emp008_factor_quantiles(
            prepared=prepared,
            start=args.start,
            end=end,
            q=args.factor_quantiles,
        )
        summary["factor_quantiles"] = quantile_result.write_outputs(
            run_root / "factor_quantiles",
            factor_set=config.factor_set,
            q=args.factor_quantiles,
        )
```

Add `--factor-quantiles` as an integer defaulting to 5 and `--no-factor-quantiles` as the opt-out flag. Validate `q >= 2` in the evaluator so both CLIs share one error contract.

- [ ] **Step 4: Update EMP008 documentation**

Document:

- the registry fields required when adding a factor;
- how dataset requirements and CLI choices are derived;
- KOSPI200 signal-date eligibility and next-month return timing;
- equal and total-market-cap weighting definitions;
- raw and preferred-direction spreads;
- standalone and full-run commands;
- every file under `factor_quantiles/` and its long-form key columns;
- the diagnostic-only limitation: no costs, sector neutrality, or automatic factor-weight optimization.

Include runnable commands:

```powershell
python -m backtesting.strategies.emp008.run_factor_quantiles --factor-set mfbt --start 2020-01-31 --end 2026-06-30
python -m backtesting.strategies.emp008.run_full --factor-set mfbt --factor-quantiles 5 --end 2026-06-30
```

- [ ] **Step 5: Run integration and documentation-adjacent checks**

Run:

```powershell
python -m pytest tests/scripts/test_run_mfbt_emp008_full.py tests/scripts/test_run_emp008_factor_quantiles.py -v
python -m backtesting.strategies.emp008.run_full --help
python -m ruff check backtesting/strategies/emp008/run_full.py tests/scripts/test_run_mfbt_emp008_full.py
git diff --check
```

Expected: tests and Ruff pass, help lists both quantile flags, and `git diff --check` is silent.

- [ ] **Step 6: Commit the integration slice**

```powershell
git add backtesting/strategies/emp008/run_full.py backtesting/strategies/emp008/README.md tests/scripts/test_run_mfbt_emp008_full.py
git commit -m "Put single-factor evidence beside every full EMP008 run" -m "Prepare factors once, emit both quantile weighting modes by default, retain an explicit opt-out, and document the extension contract for future factors." -m "Constraint: Existing weight, backtest, comparison, and attribution artifacts retain their paths" -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: Full-run orchestration, opt-out, CLI help, and regression tests; Ruff; diff check"
```

## Task 8: Verify the complete contract on tests and real EMP008 data

**Files:**
- Modify only if verification exposes a defect in files already owned by Tasks 1-7.

- [ ] **Step 1: Run all focused test modules**

Run:

```powershell
python -m pytest tests/strategies/test_emp008_factor_registry.py tests/strategies/test_emp008_factor_pipeline.py tests/strategies/test_emp008_factor_quantiles.py tests/scripts/test_run_emp008_factor_quantiles.py tests/scripts/test_run_mfbt_emp008_full.py tests/strategies/test_mfbt_emp008_experiments.py tests/strategies/test_mfbt.py -v
```

Expected: all pass.

- [ ] **Step 2: Run repository-wide static verification**

Run:

```powershell
python -m ruff check backtesting tests
python -m compileall -q backtesting tests
git diff --check origin/main...HEAD
```

Expected: Ruff passes, compileall exits zero, and diff check is silent.

- [ ] **Step 3: Run the full test suite with an extended timeout**

Run: `python -m pytest`

Expected: all repository tests pass. Use at least a 10-minute command timeout because the 120-second baseline attempt did not finish.

- [ ] **Step 4: Make the existing parquet catalog available without copying or mutating it**

The feature worktree does not contain ignored parquet data. Pass the main workspace parquet directory explicitly:

```powershell
$parquetDir = 'C:\Users\CHECK\Documents\GitHub\shquants\parquet'
Test-Path -LiteralPath $parquetDir
```

Expected: `True`. Do not create a directory junction and do not modify or regenerate parquet files.

- [ ] **Step 5: Run real-data MFBT and Origin quantile pipelines through the common horizon**

Run:

```powershell
$parquetDir = 'C:\Users\CHECK\Documents\GitHub\shquants\parquet'
python -m backtesting.strategies.emp008.run_factor_quantiles --parquet-dir $parquetDir --factor-set mfbt --start 2020-01-31 --end 2026-06-30 --output-dir results/emp008_factor_quantiles_mfbt
python -m backtesting.strategies.emp008.run_factor_quantiles --parquet-dir $parquetDir --factor-set origin --start 2020-01-31 --end 2026-06-30 --output-dir results/emp008_factor_quantiles_origin
```

Expected: both commands exit zero and print artifact payloads.

- [ ] **Step 6: Audit real outputs against acceptance criteria**

Run a read-only inline check:

```powershell
@'
from pathlib import Path
import pandas as pd

for run in ("mfbt", "origin"):
    root = Path(f"results/emp008_factor_quantiles_{run}")
    returns = pd.read_parquet(root / "monthly_returns.parquet")
    weights = pd.read_parquet(root / "portfolio_weights.parquet")
    modes = set(returns["weighting"])
    assert modes == {"equal_weight", "market_cap_weight"}
    factors = set(returns["factor"])
    expected = 6 if run == "mfbt" else 3
    assert len(factors) == expected
    assert {"q1", "q2", "q3", "q4", "q5", "high_minus_low", "preferred_minus_avoided"} <= set(returns["portfolio"])
    sums = weights.groupby(["signal_date", "factor", "weighting", "quantile"])["weight"].sum()
    assert sums.sub(1.0).abs().max() < 1e-10
    membership = weights.groupby(["signal_date", "factor", "quantile", "weighting"])["ticker"].agg(lambda x: tuple(sorted(x)))
    wide = membership.unstack("weighting").dropna()
    assert wide["equal_weight"].equals(wide["market_cap_weight"])
    print(run, len(factors), len(returns), len(weights))
'@ | python -
```

Expected: prints one line for MFBT and one for Origin with no assertion failure.

- [ ] **Step 7: Inspect branch scope and commit any verification fix**

Run:

```powershell
git status --short
git diff --stat origin/main...HEAD
git log --oneline --decorate origin/main..HEAD
```

Expected: only planned EMP008 code, tests, and documentation are tracked. Real-data `results/` remain ignored or untracked and are not committed.

If Tasks 1-7 required no verification fix, do not create an empty commit. If a verified defect was fixed, commit only its owned files with a Lore message that states the failed invariant and exact rerun evidence.

## Final Completion Audit

- [ ] Every `FactorId` appears once in `FACTOR_DEFINITIONS` and every `FactorSetId` resolves to an ordered non-empty tuple.
- [ ] Dataset loading, builder selection, preprocessing, direction, and CLI factor-set choices are registry-derived.
- [ ] Optimization, attribution, and quantiles receive one shared prepared exposure bundle in `run_full`.
- [ ] Every selected factor yields Q1-Q5, `high_minus_low`, `preferred_minus_avoided`, raw IC, and directional IC.
- [ ] Equal and total-market-cap weighting exist for every factor and share exact bucket membership.
- [ ] Weights use signal-date total market cap and returns use only next-month adjusted-close outcomes.
- [ ] Standalone and full-run paths write every documented artifact.
- [ ] Existing EMP008 variants, experiment entry points, optimizer outputs, and attribution remain compatible.
- [ ] Focused tests, full tests, Ruff, compileall, diff check, and real MFBT/Origin runs all pass.
