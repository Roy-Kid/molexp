"""Plan-mode subpackage.

Public entry point — :class:`PlanMode` plus its :class:`PlanModeConfig`
and the :class:`PlanResult` view. Schema / protocol / task modules are
considered private; tests reach into them by their full dotted path.

Materialized plan workspaces live on :class:`PlanFolder` — an agent-owned
:class:`molexp.workspace.Folder` subclass (``kind = "agent.plan"``)
that mounts under any workspace ``Folder`` via the generic
``add_folder`` API. The manifest / validation-report data types are
re-exported here for downstream sub-specs to consume.

The workflow-orthogonal :class:`~molexp.agent.review.ReviewPolicy`
protocol plus its built-in policies (:class:`BypassPolicy`,
:class:`AutoPolicy`, :class:`HumanPolicy`) live at the agent layer
because any mode with a multi-step workflow consumes them — import
from :mod:`molexp.agent.review` or the top-level :mod:`molexp.agent`.
"""

from molexp.agent.modes.plan._mode import (
    PlanMode,
    PlanModeConfig,
    PlanResult,
)
from molexp.agent.modes.plan._pipeline import PLAN_WORKFLOW, build_plan_workflow
from molexp.agent.modes.plan._repair_loop import RepairBudgetExceeded
from molexp.agent.modes.plan.errors import SkeletonCompileError, StepRejected
from molexp.agent.modes.plan.handoff import PlanRunHandoff
from molexp.agent.modes.plan.plan_folder import (
    AGENT_PLAN_KIND,
    CheckResult,
    PlanFolder,
    PlanManifest,
    PlanStatus,
    RepairIterationRecord,
    ValidationReport,
)
from molexp.agent.modes.plan.policy import (
    PLAN_NODE_NAMES,
    STANDARD_PLAN_POLICY,
    PlanModelPolicy,
)

__all__ = [
    "AGENT_PLAN_KIND",
    "PLAN_NODE_NAMES",
    "PLAN_WORKFLOW",
    "STANDARD_PLAN_POLICY",
    "CheckResult",
    "PlanFolder",
    "PlanManifest",
    "PlanMode",
    "PlanModeConfig",
    "PlanModelPolicy",
    "PlanResult",
    "PlanRunHandoff",
    "PlanStatus",
    "RepairBudgetExceeded",
    "RepairIterationRecord",
    "SkeletonCompileError",
    "StepRejected",
    "ValidationReport",
    "build_plan_workflow",
]
