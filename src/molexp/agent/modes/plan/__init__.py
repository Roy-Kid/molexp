"""Plan-mode subpackage.

Public entry point — :class:`PlanMode` plus its :class:`PlanModeConfig`
and the :class:`PlanResult` view. Schema / protocol / task modules are
considered private; tests reach into them by their full dotted path.

Materialized experiment workspaces live under the agent-owned
subsystem kind ``agent.plan-experiments`` (reserved 2026-05-09 by the
``planmode-workspace-pipeline-*`` chain — see ``.claude/notes/architecture.md``).
The on-disk layout helper :class:`PlanWorkspaceHandle` and the
manifest / validation-report data types are re-exported here for
downstream sub-specs to consume.

The workflow-orthogonal gate **protocol** lives at the agent layer
because any mode with a multi-step workflow consumes it — import
:class:`~molexp.agent.policy.GatePolicy`,
:class:`~molexp.agent.policy.AutoApproveGatePolicy`, and
:func:`~molexp.agent.policy.static_gate_policy_lookup` from
:mod:`molexp.agent.policy` (or the top-level :mod:`molexp.agent`).

The PlanMode-specific *concrete* interactive gate
:class:`PromptGatePolicy` is re-exported from here because it binds the
protocol to PlanMode's view / decision pair and is the recommended
interactive default for callers who want a CLI prompt instead of
auto-approve.
"""

from molexp.agent.modes.plan._mode import (
    PlanMode,
    PlanModeConfig,
    PlanResult,
)
from molexp.agent.modes.plan._pipeline import PLAN_WORKFLOW, build_plan_workflow
from molexp.agent.modes.plan._repair_loop import RepairBudgetExceeded
from molexp.agent.modes.plan.errors import SkeletonCompileError
from molexp.agent.modes.plan.gates import PromptGatePolicy
from molexp.agent.modes.plan.handoff import PlanRunHandoff
from molexp.agent.modes.plan.policy import (
    PLAN_NODE_NAMES,
    STANDARD_PLAN_POLICY,
    PlanModelPolicy,
)
from molexp.agent.modes.plan.protocols import PlanGatePolicy
from molexp.agent.modes.plan.workspace_layout import (
    AGENT_PLAN_EXPERIMENTS_KIND,
    CheckResult,
    PlanManifest,
    PlanStatus,
    PlanWorkspaceHandle,
    RepairIterationRecord,
    ValidationReport,
)

__all__ = [
    "AGENT_PLAN_EXPERIMENTS_KIND",
    "PLAN_NODE_NAMES",
    "PLAN_WORKFLOW",
    "STANDARD_PLAN_POLICY",
    "CheckResult",
    "PlanGatePolicy",
    "PlanManifest",
    "PlanMode",
    "PlanModeConfig",
    "PlanModelPolicy",
    "PlanResult",
    "PlanRunHandoff",
    "PlanStatus",
    "PlanWorkspaceHandle",
    "PromptGatePolicy",
    "RepairBudgetExceeded",
    "RepairIterationRecord",
    "SkeletonCompileError",
    "ValidationReport",
    "build_plan_workflow",
]
