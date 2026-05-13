"""Plan-mode subpackage.

Public entry point — :class:`PlanMode` plus its :class:`PlanModeConfig`
and the :class:`PlanResult` view. Materialized plan workspaces live on
:class:`PlanFolder` and transition to RunMode via :class:`PlanRunHandoff`.

Everything else is an implementation detail reachable by fully-qualified
path when needed (tests, internal consumers).
"""

from molexp.agent.modes.plan._mode import (
    PlanMode,
    PlanModeConfig,
    PlanResult,
)
from molexp.agent.modes.plan.handoff import PlanRunHandoff
from molexp.agent.modes.plan.plan_folder import PlanFolder

__all__ = [
    "PlanFolder",
    "PlanMode",
    "PlanModeConfig",
    "PlanResult",
    "PlanRunHandoff",
]
