"""``RunMode`` — planned workflow → tested, executed, reported experiment.

The second concrete :class:`~molexp.harness.mode.Mode`, the back half of the
harness north star: it consumes the artifacts :class:`PlanMode` left in the
**same** ``workspace.Run``'s artifact store (resolved by kind via each
stage's ``require_latest``) and carries them through test generation, real
execution, and reporting:

    GenerateTestSpec -> ValidateTestSpec -> GenerateTestCode
    -> ValidateTestSource -> MaterializeExecution -> ExecuteTests
    -> ExecuteWorkflow -> GenerateFinalReport -> ApprovalGate
    -> GenerateAuditReport

``user_input`` is the **same** natural-language draft string PlanMode ran
with. The base ledger keys on ``{mode.name}-{sha(user_input)}``, so the two
modes keep separate ledgers on one Run, and re-running RunMode on the same
draft resumes/skips completed stages by the base-class law.

Execution is subprocess-only through the injected
:class:`~molexp.harness.executors.Executor` (default
:class:`LocalExecutor`): the generated pytest module and the materialized
driver run as child processes, so the real ``molexp.workflow`` engine never
loads in the harness process. Injecting :class:`DryRunExecutor` yields a
dry-run flavored RunMode without a second mode class. Failing generated
tests raise at ``execute_tests`` (persist-then-raise), unconditionally
blocking ``execute_workflow``.

Plan-artifact pre-guard: ``require_latest`` raises ``StageExecutionError``,
which the CLI maps to a "ledger resume" hint — misleading when PlanMode was
simply never run. :meth:`RunMode.run` therefore checks
:data:`_REQUIRED_PLAN_KINDS` against the shared store first and raises
:class:`ArtifactNotFoundError` naming the missing kinds and the remedy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from molexp.harness.core.stage import Stage
from molexp.harness.errors import ArtifactNotFoundError
from molexp.harness.executors import Executor, LocalExecutor
from molexp.harness.mode import Mode
from molexp.harness.policy import make_final_report_approval_request
from molexp.harness.schemas import ApprovalPolicy, ApprovalRequest, ModeResult
from molexp.harness.stages import (
    ApprovalGate,
    ExecuteTests,
    ExecuteWorkflow,
    GenerateAuditReport,
    GenerateFinalReport,
    GenerateTestCode,
    GenerateTestSpec,
    MaterializeExecution,
    ValidateTestSource,
    ValidateTestSpec,
)
from molexp.harness.store.file_artifact_store import FileArtifactStore

if TYPE_CHECKING:
    from molexp.harness.gateways.gateway import AgentGateway

__all__ = ["RunMode"]

# PlanMode artifacts RunMode consumes; checked up front for a clear error.
_REQUIRED_PLAN_KINDS = ("experiment_report", "workflow_ir", "bound_workflow", "workflow_source")


class RunMode(Mode):
    """Planned workflow → generated tests → real execution → final report."""

    name: ClassVar[str] = "run"

    def __init__(self, executor: Executor | None = None) -> None:
        self._executor: Executor = executor if executor is not None else LocalExecutor()

    def stages(self, user_input: Any) -> list[Stage]:  # noqa: ANN401, ARG002 — ledger key only; the stage list is draft-independent
        request = self._final_report_request()
        return [
            GenerateTestSpec(),
            ValidateTestSpec(),
            GenerateTestCode(),
            ValidateTestSource(),
            MaterializeExecution(),
            ExecuteTests(self._executor),
            ExecuteWorkflow(self._executor),
            GenerateFinalReport(),
            ApprovalGate(requests=[request]),
            GenerateAuditReport(),
        ]

    async def run(
        self,
        *,
        run: Any,  # noqa: ANN401 — workspace.Run (duck-typed run_dir/id, as in Mode)
        user_input: Any,  # noqa: ANN401
        gateway: AgentGateway | None = None,
    ) -> ModeResult:
        """Guard for PlanMode artifacts, then run the stage sequence.

        Raises:
            ArtifactNotFoundError: If any :data:`_REQUIRED_PLAN_KINDS`
                artifact is missing from the run's store — PlanMode has not
                run on this Run yet.
        """
        store = FileArtifactStore(root=run.run_dir / "artifacts")
        missing = [kind for kind in _REQUIRED_PLAN_KINDS if store.latest_by_kind(kind) is None]
        if missing:
            raise ArtifactNotFoundError(
                f"run {run.id!r} has no {', '.join(repr(k) for k in missing)} artifact(s); "
                "run PlanMode first (molexp plan) to produce the planning artifacts "
                "RunMode executes"
            )
        return await super().run(run=run, user_input=user_input, gateway=gateway)

    @staticmethod
    def _final_report_request() -> ApprovalRequest:
        """Build the final-report approval request, narrowing the Optional.

        ``make_final_report_approval_request`` returns ``None`` only when the
        policy disables final-report approval; the default ``ApprovalPolicy``
        requires it, so the ``None`` branch is unreachable here.
        """
        request = make_final_report_approval_request(ApprovalPolicy())
        if request is None:  # pragma: no cover — default policy always requires it
            raise AssertionError(
                "unreachable: default ApprovalPolicy requires final-report approval"
            )
        return request
