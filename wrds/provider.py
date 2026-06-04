from __future__ import annotations

from backtesting.data import SourceRegistry
try:
    from .core.workflow import WorkflowRegistry
    from .derivatives.options.workflow import OptionsWorkflow
    from .marketdata.workflow import DataWorkflow
    from .marketdata.catalog import source_registry as marketdata_source_registry
    from .universes.factset.workflow import UniverseWorkflow
    from .universes.us.workflow import USWorkflow
except ImportError:  # pragma: no cover - direct script compatibility
    from core.workflow import WorkflowRegistry
    from derivatives.options.workflow import OptionsWorkflow
    from marketdata.workflow import DataWorkflow
    from marketdata.catalog import source_registry as marketdata_source_registry
    from universes.factset.workflow import UniverseWorkflow
    from universes.us.workflow import USWorkflow


def source_registry() -> SourceRegistry:
    return marketdata_source_registry()


def flow_registry() -> WorkflowRegistry:
    return WorkflowRegistry((UniverseWorkflow(), USWorkflow(), OptionsWorkflow(), DataWorkflow()))
