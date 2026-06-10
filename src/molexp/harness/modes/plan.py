"""``PlanMode`` — natural-language draft → runnable molexp.workflow source.

The first concrete :class:`~molexp.harness.mode.Mode`. It declares the
canonical planning-and-codegen stage sequence; the ``Mode`` base owns eager
task-by-task execution, the per-run completion ledger (verified caching +
resume), ``workspace.Run`` persistence, and audit. ``PlanMode`` owns **no**
LLM logic — the LLM-driven stages dispatch through the injected gateway.

Stage sequence (each stage resolves its upstream input by artifact kind):

    SaveUserPlan -> GenerateExperimentReport -> ExtractWorkflowIR
    -> ValidateWorkflowIR -> BindMolcraftsTasks -> ValidateBoundWorkflow
    -> GenerateWorkflowSource -> ValidateWorkflowSource -> ApprovalGate

``user_input`` is the short natural-language experiment draft (a ``str``).
The terminal :class:`ApprovalGate` resolves its decision **at run time**
through the gate's default auto-grant approver (the pipeline is
non-interactive by design) — the grant is a real decision recorded on the
event log with its actual ``decided_at``, not a pre-baked construction-time
value. A future interactive flow passes its own approver via
``ApprovalGate(requests, approve=...)`` without touching this mode.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

from molexp.harness.core.stage import Stage
from molexp.harness.mode import Mode
from molexp.harness.schemas import ApprovalRequest
from molexp.harness.stages import (
    ApprovalGate,
    BindMolcraftsTasks,
    ExtractWorkflowIR,
    GenerateExperimentReport,
    GenerateWorkflowSource,
    SaveUserPlan,
    ValidateBoundWorkflow,
    ValidateWorkflowIR,
    ValidateWorkflowSource,
)

__all__ = ["PlanMode"]


class PlanMode(Mode):
    """Idea → experiment plan → WorkflowIR → runnable molexp.workflow source."""

    name: ClassVar[str] = "plan"

    def stages(self, user_input: Any) -> list[Stage]:  # noqa: ANN401 — the NL draft
        request = ApprovalRequest(
            id="approve-plan",
            intent="final_report",
            reason="gate the plan-mode output before it is considered final",
            triggered_by_policy="PlanMode",
            created_at=datetime.now(tz=UTC),
        )
        return [
            SaveUserPlan(user_text=str(user_input)),
            GenerateExperimentReport(),
            ExtractWorkflowIR(),
            ValidateWorkflowIR(),
            BindMolcraftsTasks(),
            ValidateBoundWorkflow(),
            GenerateWorkflowSource(),
            ValidateWorkflowSource(),
            ApprovalGate(requests=[request]),
        ]
