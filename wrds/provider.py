from __future__ import annotations

try:
    from .core.workflow import WorkflowRegistry
    from .derivatives.options.workflow import OptionsWorkflow
    from .marketdata.workflow import DataWorkflow
    from .universes.factset.workflow import UniverseWorkflow
    from .universes.us.workflow import USWorkflow
except ImportError:  # pragma: no cover - direct script compatibility
    from core.workflow import WorkflowRegistry
    from derivatives.options.workflow import OptionsWorkflow
    from marketdata.workflow import DataWorkflow
    from universes.factset.workflow import UniverseWorkflow
    from universes.us.workflow import USWorkflow


def flow_registry() -> WorkflowRegistry:
    return WorkflowRegistry((UniverseWorkflow(), USWorkflow(), OptionsWorkflow(), DataWorkflow()))
