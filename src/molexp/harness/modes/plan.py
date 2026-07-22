"""``PlanMode`` — the single end-to-end plan pipeline (9 steps).

The one concrete :class:`~molexp.harness.mode.Mode`. It turns a short
natural-language experiment draft into a fully-verified, reviewed plan plus
a descriptive execution report, in nine visible steps:

    1. Draft proposal   SaveUserPlan -> AssembleKnowledgeContext
                         -> GenerateExperimentReport
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
    7. Compile/dry-run   MaterializeAndExecuteTests (repair) -> CompileWorkflow
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
    AssembleKnowledgeContext,
    BindMolcraftsTasks,
    CompileWorkflow,
    ExecuteWorkflow,
    ExtractWorkflowIR,
    GenerateAuditReport,
    GenerateExecutionReport,
    GenerateExperimentReport,
    GenerateExperimentSpec,
    GenerateFinalReport,
    GenerateInputSet,
    RepairLoop,
    ResolveCapabilities,
    ReviewPlan,
    SaveUserPlan,
    StepAuditLoop,
    ValidateBoundWorkflow,
    ValidateExperimentSpec,
    ValidateInputSet,
    ValidateWorkflowIR,
    ValidateWorkflowSource,
)
from molexp.harness.stages.review_pack_builders import (
    build_experiment_spec_review_pack,
    build_plan_review_pack,
)
from molexp.harness.stages.sequential_task_build import SequentialTaskBuild

if TYPE_CHECKING:
    from molexp.workspace.models import ComputeTarget

__all__ = ["DEFAULT_REPAIR_ATTEMPTS", "PlanMode", "PlanStep"]

#: Outer repair budget for every author generate→check loop. Raising this is
#: not a quality strategy — fix diagnosis / evidence (molmcp) instead of
#: lottery-retrying. Operators may override via PlanMode(repair_attempts=…).
DEFAULT_REPAIR_ATTEMPTS = 3


class PlanStep:
    """One *visible* step of the plan pipeline — a titled group of stages.

    The nine step titles are the documented user-facing vocabulary
    (``docs/guide/plan-mode.md`` / ``docs/architecture/plan-mode.md``); the
    stage objects inside are internal execution units. ``tail=True`` marks
    the opt-in ``--execute`` group, which is *not* one of the nine steps.
    A runtime container (holds live ``Stage`` instances), so a plain class.
    """

    def __init__(self, title: str, stages: list[Stage], *, tail: bool = False) -> None:
        self.title = title
        self.stages = stages
        self.tail = tail


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
        repair_attempts: int = DEFAULT_REPAIR_ATTEMPTS,
    ) -> None:
        if repair_attempts < 1:
            raise ValueError("PlanMode repair_attempts must be >= 1")
        self._approver = approver
        self._executor: Executor = executor if executor is not None else LocalExecutor()
        self._execute = execute
        self._compute_target = compute_target
        self._repair_attempts = repair_attempts

    def step_groups(self, user_input: Any) -> list[PlanStep]:  # noqa: ANN401 — the NL draft
        """The nine documented steps (plus the opt-in ``--execute`` tail).

        The single source of truth for step ↔ stage grouping: :meth:`stages`
        flattens this structure, and the CLI banner/progress renders it, so
        the user-facing "nine steps" and the executed stage sequence can
        never drift apart.
        """
        n = self._repair_attempts
        # Shared so StepAuditLoop can re-run the generator after reject/revise.
        spec_generate_loop = RepairLoop(
            name="generate_experiment_spec",
            generate=GenerateExperimentSpec(),
            validators=[ValidateExperimentSpec()],
            feedback_kind="experiment_spec_feedback",
            attempts=n,
        )
        groups: list[PlanStep] = [
            # 1. Draft proposal — capture the request, draft a human-readable report.
            PlanStep(
                "Draft proposal",
                [
                    SaveUserPlan(user_text=str(user_input)),
                    # Prior knowledge becomes a first-class artifact BEFORE the
                    # writers run — the knowledge→plan lineage edge is real.
                    AssembleKnowledgeContext(),
                    GenerateExperimentReport(),
                ],
            ),
            # 2. Draft spec — concretize every parameter, resolve open questions.
            # StepAuditLoop (hard) lets the human approve the concrete spec
            # BEFORE it is fed to the LLM to build the workflow. A rejection
            # stops here: no capability discovery, no IR, no source ever runs.
            PlanStep(
                "Draft spec",
                [
                    spec_generate_loop,
                    StepAuditLoop(
                        name="approve_experiment_spec",
                        subject_kind="experiment_spec",
                        pack_builder=build_experiment_spec_review_pack,
                        policy="hard",
                        request=self._spec_request(),
                        approve=self._approver,
                        regenerate=spec_generate_loop,
                        feedback_kind="experiment_spec_feedback",
                        result_kind="spec_approval",
                    ),
                ],
            ),
            # 3. Resolve capabilities — discover the molcrafts toolchain.
            PlanStep("Resolve capabilities", [ResolveCapabilities()]),
            # 4. Workflow IR — lift the concrete spec into a flow + topology.
            PlanStep(
                "Workflow IR",
                [
                    RepairLoop(
                        name="extract_workflow_ir",
                        generate=ExtractWorkflowIR(),
                        validators=[ValidateWorkflowIR()],
                        feedback_kind="workflow_ir_feedback",
                        attempts=n,
                    ),
                ],
            ),
            # 5. Bind + sequential task build: ONE task at a time
            #    (codegen → unit test → pytest) so dry-run failures cannot
            #    wait until the whole suite is written. ReviewPlan still
            #    gates the assembled source after all tasks are green.
            PlanStep(
                "Tasks + per-task tests",
                [
                    RepairLoop(
                        name="bind_molcrafts_tasks",
                        generate=BindMolcraftsTasks(),
                        validators=[ValidateBoundWorkflow()],
                        feedback_kind="bound_workflow_feedback",
                        attempts=n,
                    ),
                    SequentialTaskBuild(
                        self._executor, attempts=self._repair_attempts
                    ),
                    # Assembly already green under per-task pytest; structural
                    # + semantic gates only (no full rewrite of sequential work).
                    ValidateWorkflowSource(),
                    ReviewPlan(),
                ],
            ),
            # 6. Input set — the parameter-space sweep the workflow runs over.
            PlanStep(
                "Input set",
                [
                    RepairLoop(
                        name="generate_input_set",
                        generate=GenerateInputSet(),
                        validators=[ValidateInputSet()],
                        feedback_kind="input_set_feedback",
                        attempts=self._repair_attempts,
                    ),
                ],
            ),
            # 7. Compile-only dry-run (per-task pytest already passed in step 5).
            PlanStep(
                "Compile / dry-run",
                [
                    CompileWorkflow(self._executor),
                ],
            ),
            # 8. Review — StepAuditLoop hard-gates the whole verified plan.
            PlanStep(
                "Review",
                [
                    StepAuditLoop(
                        name="approve_plan",
                        subject_kind="execution_result",
                        pack_builder=build_plan_review_pack,
                        policy="hard",
                        request=self._final_request(),
                        approve=self._approver,
                        feedback_kind="plan_review_feedback",
                        result_kind="analysis_result",
                    ),
                ],
            ),
            # 9. Execution report — describe where & how it would run (no submission).
            PlanStep("Execution report", [GenerateExecutionReport(self._compute_target)]),
        ]
        if self._execute:
            groups.append(PlanStep("Execute", self._execution_tail(), tail=True))
        return groups

    def stages(self, user_input: Any) -> list[Stage]:  # noqa: ANN401 — the NL draft
        return [stage for group in self.step_groups(user_input) for stage in group.stages]

    def _execution_tail(self) -> list[Stage]:
        """The opt-in real-execution stages (the folded-in RunMode back half)."""
        return [
            ExecuteWorkflow(self._executor),
            GenerateFinalReport(),
            # The tail gate rides the same approver as the other two gates.
            ApprovalGate(
                requests=[self._execution_request()],
                approve=self._approver,
                name="approve_execution",
            ),
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
