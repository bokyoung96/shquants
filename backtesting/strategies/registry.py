from __future__ import annotations

import inspect
from typing import Callable

from .base import RegisteredStrategy
from .momentum import MomentumTopN
from .revision_sector_q1q5_oi_guard_ls import RevisionSectorQ1Q5OiConfidenceWeightedLs
from .revision_oi_short_squeeze_beta_exclusion_ls import RevisionOiShortSqueezeBetaExclusionLs
from .revision_oi_qualified_long_beta_boost_ls import RevisionOiQualifiedLongBetaBoostLs
from .consensus_beta_regime_rotation_ls import ConsensusBetaRegimeRotationLs
from .consensus_beta_sector_tilt_longonly import ConsensusBetaSectorTiltLongOnly
from .consensus_beta_breadth_scaled_longonly import ConsensusBetaBreadthScaledLongOnly
from .consensus_beta_soft_participation_longonly import ConsensusBetaSoftParticipationLongOnly
from .consensus_beta_soft_participation_index_overlay import ConsensusBetaSoftParticipationIndexOverlay
from .index_alpha_tilt_consensus_revision_oi_beta import IndexAlphaTiltConsensusRevisionOiBeta


StrategyFactory = Callable[..., RegisteredStrategy]

_REGISTRY: dict[str, StrategyFactory] = {}


def register_strategy(name: str, factory: StrategyFactory) -> None:
    if name in _REGISTRY:
        raise ValueError(f"strategy already registered: {name}")
    _REGISTRY[name] = factory


def build_strategy(name: str, **kwargs: object) -> RegisteredStrategy:
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown strategy '{name}'. Available: {available}") from exc
    params = inspect.signature(factory).parameters
    filtered = {key: value for key, value in kwargs.items() if key in params}
    return factory(**filtered)


def list_strategies() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


register_strategy("momentum", MomentumTopN)
register_strategy("revision_sector_q1q5_oi_confidence_weighted_ls", RevisionSectorQ1Q5OiConfidenceWeightedLs)
register_strategy("revision_oi_short_squeeze_beta_exclusion_ls", RevisionOiShortSqueezeBetaExclusionLs)
register_strategy("revision_oi_qualified_long_beta_boost_ls", RevisionOiQualifiedLongBetaBoostLs)
register_strategy("consensus_beta_regime_rotation_ls", ConsensusBetaRegimeRotationLs)
register_strategy("consensus_beta_sector_tilt_longonly", ConsensusBetaSectorTiltLongOnly)
register_strategy("consensus_beta_breadth_scaled_longonly", ConsensusBetaBreadthScaledLongOnly)
register_strategy("consensus_beta_soft_participation_longonly", ConsensusBetaSoftParticipationLongOnly)
register_strategy("consensus_beta_soft_participation_index_overlay", ConsensusBetaSoftParticipationIndexOverlay)
register_strategy("index_alpha_tilt_consensus_revision_oi_beta", IndexAlphaTiltConsensusRevisionOiBeta)
