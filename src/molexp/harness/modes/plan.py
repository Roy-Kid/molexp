"""``PlanMode`` — the single end-to-end plan pipeline (9 steps).

The one concrete :class:`~molexp.harness.mode.Mode`. It turns a short
natural-language experiment draft into a fully-verified, reviewed plan plus
a descriptive execution report, in nine visible steps:

    1. Draft proposal   SaveUserPlan -> GenerateExperimentReport
    2. Draft spec        GenerateExperimentSpec -> ValidateExperimentSpec
                         -> ApprovalGate(experiment_spec)   # human approves the
                            concrete spec BEFORE it is compiled into a workflow
    3. Resolve caps      ResolveCapabilities
    4. Workflow IR       ExtractWorkflowIR -> ValidateWorkflowIR
    5. Tasks + tests     BindMolcraftsTasks -> ValidateBoundWorkflow
                         -> GenerateWorkflowSource -> ValidateWorkflowSource/ReviewPlan
                         -> GenerateTestSpec -> ValidateTestSpec
                         -> GenerateTestCode -> ValidateTestSource
    6. Input set         GenerateInputSet -> ValidateInputSet
    7. Compile/dry-run   MaterializeExecution -> ExecuteTests -> CompileWorkflow
    8. Review            ApprovalGate(final_report)
    9. Execution report  GenerateExecutionReport

The earlier two-class PlanMode/RunMode split is retired: real scientific
execution is **not** one of the nine steps. It is an explicit opt-in tail
(``PlanMode(execute=True)``), gated by the step-8 review, that runs the real
workflow and writes the final + audit reports — the surviving RunMode
stages, folded in. The plan-only default stops at the execution report and
never submits compute (north-star §2.2 forbids auto-submission).

``user_input`` is the short natural-language draft (a ``str``). The base
``Mode`` owns eager task-by-task execution, the per-run completion ledger
(verified caching + resume), and audit; ``PlanMode`` owns no LLM logic —
the LLM-driven stages dispatch through the injected gateway.

Injection (all optional, mirroring the old ``PlanMode(approver=...)`` /
``RunMode(executor=...)``):

* ``approver`` — the async callback for the step-8 review gate (default
  auto-grant; the pipeline is non-interactive by default).
* ``executor`` — the :class:`Executor` ``ExecuteTests`` / ``CompileWorkflow``
  (and the ``--execute`` tail) drive their subprocesses through; default
  :class:`LocalExecutor`. Compile + per-task tests are cheap and run no
  science; inject :class:`DryRunExecutor` to skip them entirely.
* ``execute`` — when True, append the real-execution tail.
* ``compute_target`` — the workspace ``ComputeTarget`` the step-9 execution
  report describes (``None`` → a local descriptive default).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from molexp.harness.core.stage import Stage
from molexp.harness.executors import Executor, LocalExecutor
from molexp.harness.mode import Mode
from molexp.harness.schemas import ApprovalRequest
from molexp.harness.stages import (
    ApprovalGate,
    Approver,
    BindMolcraftsTasks,
    CompileWorkflow,
    ExecuteTests,
    ExecuteWorkflow,
    ExtractWorkflowIR,
    GenerateAuditReport,
    GenerateExecutionReport,
    GenerateExperimentReport,
    GenerateExperimentSpec,
    GenerateFinalReport,
    GenerateInputSet,
    GenerateTestCode,
    GenerateTestSpec,
    GenerateWorkflowSource,
    MaterializeExecution,
    RepairLoop,
    ResolveCapabilities,
    ReviewPlan,
    SaveUserPlan,
    ValidateBoundWorkflow,
    ValidateExperimentSpec,
    ValidateInputSet,
    ValidateTestSource,
    ValidateTestSpec,
    ValidateWorkflowIR,
    ValidateWorkflowSource,
)

if TYPE_CHECKING:
    from molexp.workspace.models import ComputeTarget

__all__ = ["PlanMode"]


class PlanMode(Mode):
    """Idea → verified plan → execution report (9 steps), optionally executed."""

    name: ClassVar[str] = "plan"

    def __init__(
        self,
        approver: Approver | None = None,
        executor: Executor | None = None,
        *,
        execute: bool = False,
        compute_target: ComputeTarget | None = None,
    ) -> None:
        self._approver = approver
        self._executor: Executor = executor if executor is not None else LocalExecutor()
        self._execute = execute
        self._compute_target = compute_target

    def stages(self, user_input: Any) -> list[Stage]:  # noqa: ANN401 — the NL draft
        stages: list[Stage] = [
            # 1. Draft proposal — capture the request, draft a human-readable report.
            SaveUserPlan(user_text=str(user_input)),
            GenerateExperimentReport(),
            # 2. Draft spec — concretize every parameter, resolve open questions.
            RepairLoop(
                name="generate_experiment_spec",
                generate=GenerateExperimentSpec(),
                validators=[ValidateExperimentSpec()],
                feedback_kind="experiment_spec_feedback",
            ),
            # Spec approval — the human approves the concrete spec BEFORE it is
            # fed to the LLM to build the workflow. A rejection stops here: no
            # capability discovery, no IR, no source ever runs. `PlanMode(
            # approver=…)` wires this gate (the meaningful pre-compile checkpoint).
            ApprovalGate(
                requests=[self._spec_request()],
                approve=self._approver,
                name="approve_experiment_spec",
                result_kind="spec_approval",
            ),
            # 3. Resolve capabilities — discover the molcrafts toolchain.
            ResolveCapabilities(),
            # 4. Workflow IR — lift the concrete spec into a flow + topology.
            RepairLoop(
                name="extract_workflow_ir",
                generate=ExtractWorkflowIR(),
                validators=[ValidateWorkflowIR()],
                feedback_kind="workflow_ir_feedback",
            ),
            # 5. Tasks + per-task tests — bind, codegen, and a unit test per task.
            RepairLoop(
                name="bind_molcrafts_tasks",
                generate=BindMolcraftsTasks(),
                validators=[ValidateBoundWorkflow()],
                feedback_kind="bound_workflow_feedback",
            ),
            RepairLoop(
                name="generate_workflow_source",
                generate=GenerateWorkflowSource(),
                validators=[ValidateWorkflowSource(), ReviewPlan()],
                feedback_kind="workflow_source_feedback",
                attempts=4,
            ),
            GenerateTestSpec(),
            ValidateTestSpec(),
            GenerateTestCode(),
            ValidateTestSource(),
            # 6. Input set — the parameter-space sweep the workflow runs over.
            RepairLoop(
                name="generate_input_set",
                generate=GenerateInputSet(),
                validators=[ValidateInputSet()],
                feedback_kind="input_set_feedback",
            ),
            # 7. Compile / dry-run — materialize, run per-task tests, compile (no science).
            MaterializeExecution(),
            ExecuteTests(self._executor),
            CompileWorkflow(self._executor),
            # 8. Review — gate the whole verified plan before it is final.
            ApprovalGate(
                requests=[self._final_request()], approve=self._approver, name="approve_plan"
            ),
            # 9. Execution report — describe where & how it would run (no submission).
            GenerateExecutionReport(self._compute_target),
        ]
        if self._execute:
            stages.extend(self._execution_tail())
        return stages

    def _execution_tail(self) -> list[Stage]:
        """The opt-in real-execution stages (the folded-in RunMode back half)."""
        return [
            ExecuteWorkflow(self._executor),
            GenerateFinalReport(),
            ApprovalGate(requests=[self._execution_request()], name="approve_execution"),
            GenerateAuditReport(),
        ]

    @staticmethod
    def _spec_request() -> ApprovalRequest:
        return ApprovalRequest(
            id="approve-experiment-spec",
            intent="experiment_spec",
            reason="approve the concrete experiment spec before it compiles to a workflow",
            triggered_by_policy="PlanMode",
            created_at=datetime.now(tz=UTC),
        )

    @staticmethod
    def _final_request() -> ApprovalRequest:
        return ApprovalRequest(
            id="approve-plan",
            intent="final_report",
            reason="review the full verified plan before it is considered final",
            triggered_by_policy="PlanMode",
            created_at=datetime.now(tz=UTC),
        )

    @staticmethod
    def _execution_request() -> ApprovalRequest:
        return ApprovalRequest(
            id="approve-execution",
            intent="final_report",
            reason="review the executed result before it is considered final",
            triggered_by_policy="PlanMode",
            created_at=datetime.now(tz=UTC),
        )
