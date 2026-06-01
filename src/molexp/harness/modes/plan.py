"""``PlanMode`` — natural-language draft → runnable molexp.workflow source.

The first concrete :class:`~molexp.harness.mode.Mode`. It declares the
canonical planning-and-codegen stage sequence; the ``Mode`` base owns eager
task-by-task execution, the per-run completion ledger (caching + resume),
``workspace.Run`` persistence, and audit. ``PlanMode`` owns **no** LLM logic —
the LLM-driven stages dispatch through the injected gateway.

Stage sequence (each stage resolves its upstream input by artifact kind):

    SaveUserPlan -> GenerateExperimentReport -> ExtractWorkflowIR
    -> ValidateWorkflowIR -> BindMolcraftsTasks -> ValidateBoundWorkflow
    -> GenerateWorkflowSource -> ValidateWorkflowSource -> ApprovalGate

``user_input`` is the short natural-language experiment draft (a ``str``).
Approval auto-grants for non-interactive runs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

from molexp.harness.core.stage import Stage
from molexp.harness.mode import Mode
from molexp.harness.schemas import ApprovalDecision, ApprovalRequest
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
        now = datetime.now(tz=UTC)
        request = ApprovalRequest(
            id="approve-plan",
            intent="final_report",
            reason="auto-approve plan-mode output for a non-interactive run",
            triggered_by_policy="PlanMode",
            created_at=now,
        )
        decision = ApprovalDecision(
            request_id="approve-plan",
            granted=True,
            decided_by="PlanMode",
            decided_at=now,
            reason="auto-grant (non-interactive)",
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
            ApprovalGate(decisions=[(request, decision)]),
        ]
