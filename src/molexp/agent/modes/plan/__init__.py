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
"""

from molexp.agent.modes.plan._mode import (
    PlanMode,
    PlanModeConfig,
    PlanResult,
)
from molexp.agent.modes.plan._pipeline import PLAN_WORKFLOW, build_plan_workflow
from molexp.agent.modes.plan.errors import SkeletonCompileError
from molexp.agent.modes.plan.policy import (
    PLAN_NODE_NAMES,
    STANDARD_PLAN_POLICY,
    PlanModelPolicy,
)
from molexp.agent.modes.plan.workspace_layout import (
    AGENT_PLAN_EXPERIMENTS_KIND,
    CheckResult,
    PlanManifest,
    PlanWorkspaceHandle,
    ValidationReport,
)

__all__ = [
    "AGENT_PLAN_EXPERIMENTS_KIND",
    "PLAN_NODE_NAMES",
    "PLAN_WORKFLOW",
    "STANDARD_PLAN_POLICY",
    "CheckResult",
    "PlanManifest",
    "PlanMode",
    "PlanModeConfig",
    "PlanModelPolicy",
    "PlanResult",
    "PlanWorkspaceHandle",
    "SkeletonCompileError",
    "ValidationReport",
    "build_plan_workflow",
]
