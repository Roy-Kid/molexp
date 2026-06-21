"""``PlanMode`` — natural-language draft → runnable molexp.workflow source.

The first concrete :class:`~molexp.harness.mode.Mode`. It declares the
canonical planning-and-codegen stage sequence; the ``Mode`` base owns eager
task-by-task execution, the per-run completion ledger (verified caching +
resume), ``workspace.Run`` persistence, and audit. ``PlanMode`` owns **no**
LLM logic — the LLM-driven stages dispatch through the injected gateway.

Stage sequence (each stage resolves its upstream input by artifact kind):

    SaveUserPlan -> GenerateExperimentReport -> ApprovalGate(experiment_spec)
    -> ExtractWorkflowIR -> ValidateWorkflowIR -> BindMolcraftsTasks
    -> ValidateBoundWorkflow -> GenerateWorkflowSource -> ValidateWorkflowSource
    -> ApprovalGate(final_report)

``user_input`` is the short natural-language experiment draft (a ``str``).
There are **two** gates. The early ``experiment_spec`` gate (named
``approve_experiment_spec``) sits right after the experiment report so a
human can review it **before** the plan compiles to WorkflowIR/source; it
takes the optional ``approver`` injected into ``PlanMode(approver=...)``. A
rejection there raises ``StageExecutionError`` before ``ExtractWorkflowIR``
runs, so no ``workflow_ir`` artifact is ever produced. The terminal
``final_report`` gate stays on the default auto-grant approver (the pipeline
is non-interactive by default). Both decisions are real decisions recorded on
the event log with their actual ``decided_at``. ``PlanMode()`` with no
argument keeps today's fully-auto behavior.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

from molexp.harness.core.stage import Stage
from molexp.harness.mode import Mode
from molexp.harness.schemas import ApprovalRequest
from molexp.harness.stages import (
    ApprovalGate,
    Approver,
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
    """Idea → experiment plan → WorkflowIR → runnable molexp.workflow source.

    ``approver`` (optional) gates the early ``experiment_spec`` checkpoint —
    a single async callback mirroring ``RunMode(executor=...)``; ``None``
    leaves that gate on the auto-grant default. It is not a config object and
    introduces no factory function.
    """

    name: ClassVar[str] = "plan"

    def __init__(self, approver: Approver | None = None) -> None:
        self._approver = approver

    def stages(self, user_input: Any) -> list[Stage]:  # noqa: ANN401 — the NL draft
        review_request = ApprovalRequest(
            id="approve-experiment-spec",
            intent="experiment_spec",
            reason="review the experiment report before the plan compiles to a workflow",
            triggered_by_policy="PlanMode",
            created_at=datetime.now(tz=UTC),
        )
        final_request = ApprovalRequest(
            id="approve-plan",
            intent="final_report",
            reason="gate the plan-mode output before it is considered final",
            triggered_by_policy="PlanMode",
            created_at=datetime.now(tz=UTC),
        )
        return [
            SaveUserPlan(user_text=str(user_input)),
            GenerateExperimentReport(),
            ApprovalGate(
                requests=[review_request],
                approve=self._approver,
                name="approve_experiment_spec",
            ),
            ExtractWorkflowIR(),
            ValidateWorkflowIR(),
            BindMolcraftsTasks(),
            ValidateBoundWorkflow(),
            GenerateWorkflowSource(),
            ValidateWorkflowSource(),
            ApprovalGate(requests=[final_request]),
        ]
