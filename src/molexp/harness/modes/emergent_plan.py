"""``EmergentPlanOrchestrator`` — the emergent-planning pipeline (05c).

This module composes the phases 01-05b parts into one runnable planning
pipeline that coexists with the nine-step :class:`~molexp.harness.modes.PlanMode`.
It deliberately does **not** subclass :class:`~molexp.harness.mode.Mode` and
holds no completion ledger: the emergent driver has no linear stage-ledger
semantics — its resume correctness rides entirely on the hard gate's
store-first replay (see below), not on a per-stage cache.

Coexistence with ``PlanMode`` (spec ``plan-emergent-05c-orchestrator``):
    The orchestrator lives beside ``PlanMode`` in ``harness.modes`` and is
    reached by the subpackage path ``harness.modes.emergent_plan`` only. This
    module is **not** added to ``harness/modes/__init__.py`` or
    ``molexp.harness.__all__`` (still locked at 21 symbols) — swapping the
    default driver and re-pointing ``drive_plan_mode`` / CLI / server is
    phase-08's job. ``EmergentPlanOrchestrator.run`` matches the shared
    :func:`~molexp.services.plan_runtime.drive_plan_mode` ``_ModeLike`` shape
    (``async run(*, run, user_input, gateway, capability_registry=None)``), so
    08 can re-point onto it without changing that call site.

Store-first suspend / replay contract:
    The plan's human review is a ``StepAuditLoop`` **hard** gate with a
    *stable* per-run request id. With no approver and no stored grant the gate
    records the request **pending** in the run's ``harness.sqlite`` and raises
    :class:`~molexp.harness.errors.ApprovalPendingError` — a suspension, never
    a failure. A later re-run with a stored grant for that same id replays
    store-first straight through the gate into
    :func:`~molexp.harness.plan.freeze_experiment_plan` (content-addressed: the
    same board freezes to the same id) and the ``plan_report_renderer`` render.
    A **malformed** final board never reaches the gate at all: a deterministic
    :class:`~molexp.harness.validators.EmergentPlanFormValidator` guard raises
    before the ``experiment_plan`` subject is even persisted, so no review pack
    and no pending request are recorded.

Widened harness→agent edge:
    Loop construction is the driver's job (phase-02 design), so the private
    :class:`InteractiveLoopPlanRunner` — the **only** harness site importing
    ``molexp.agent.loops`` — builds a phase-02 ``InteractiveLoop`` on the
    harness side. It imports ``molexp.agent.loops`` **lazily inside**
    :meth:`InteractiveLoopPlanRunner.run_planning` (mirroring ``Mode.run``
    deferring ``molexp.workflow``), so ``import molexp.harness`` still pulls no
    ``pydantic_ai`` / ``pydantic_graph`` / ``molexp.workflow``. The loop /
    runtime / session / events / hook modules are on the sanctioned
    harness→agent allowlist (``tests/test_harness/test_import_guard.py``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mollog import get_logger

from molexp.agent.loops.hooks import HookOutcome, LoopHooks, LoopState
from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.errors import StageExecutionError
from molexp.harness.plan import (
    ExperimentPlan,
    TaskBoard,
    board_path,
    freeze_experiment_plan,
    read_board,
    write_board,
)
from molexp.harness.plan_tools import BOARD_TOOLS, as_loop_tool
from molexp.harness.schemas import AgentCallSpec, ApprovalRequest, ModeResult
from molexp.harness.stages.approval_gate import Approver
from molexp.harness.stages.plan_reachability_probe import PlanReachabilityProbe
from molexp.harness.stages.review_pack_builders import build_experiment_plan_review_pack
from molexp.harness.stages.step_audit_loop import StepAuditLoop
from molexp.harness.store.approval_store import SQLiteApprovalStore
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore
from molexp.harness.validators import EmergentPlanFormValidator
from molexp.workspace.utils import derive_execution_id

if TYPE_CHECKING:
    from molexp.agent.router import Router
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.registry.capability_registry import CapabilityRegistry

__all__ = ["EmergentPlanOrchestrator", "InteractiveLoopPlanRunner", "PlanLoopRunner"]

_LOG = get_logger(__name__)


@runtime_checkable
class PlanLoopRunner(Protocol):
    """Inversion seam: drive the emergent planning loop over a task board.

    The orchestrator depends on this Protocol rather than importing an agent
    loop directly, so tests inject a stub runner (writing a canned board) and
    production injects :class:`InteractiveLoopPlanRunner` (driving a real
    phase-02 Pi loop). A conforming object exposes one async method.
    """

    async def run_planning(
        self,
        *,
        ctx: HarnessRunContext,
        board: TaskBoard,
        tools: tuple[object, ...],
        hooks: LoopHooks,
        user_input: str,
    ) -> None:
        """Drive the planning loop; the board is reached on disk via ``ctx``."""
        ...


@runtime_checkable
class _RouterBackedGateway(Protocol):
    """A gateway that also exposes its underlying :class:`Router`.

    :class:`~molexp.harness.gateways.RouterBackedAgentGateway` satisfies this
    (its ``router`` property); the bare :class:`AgentGateway` Protocol does not.
    The loop runner needs the router to build an ``AgentRuntime``.
    """

    @property
    def router(self) -> Router: ...


class EmergentPlanOrchestrator:
    """Compose 01-05b into one runnable emergent-planning pipeline.

    Not a :class:`~molexp.harness.mode.Mode` subclass and holds no ledger — see
    the module docstring for the coexistence-with-``PlanMode`` boundary and the
    store-first suspend/replay contract.

    Args:
        loop_runner: The planning-loop driver seam; defaults to the production
            :class:`InteractiveLoopPlanRunner`.
        probe: The read-only feasibility annotator; defaults to a fresh
            :class:`~molexp.harness.stages.plan_reachability_probe.PlanReachabilityProbe`.
        approve: Optional approver for the plan-tool side-effect gate and the
            hard review gate. ``None`` means the gate suspends store-first.
    """

    name = "emergent_plan"

    def __init__(
        self,
        *,
        loop_runner: PlanLoopRunner | None = None,
        probe: PlanReachabilityProbe | None = None,
        approve: Approver | None = None,
    ) -> None:
        self._loop_runner: PlanLoopRunner = loop_runner or InteractiveLoopPlanRunner()
        self._probe = probe or PlanReachabilityProbe()
        self._approve = approve

    async def run(
        self,
        *,
        run: Any,  # noqa: ANN401 — workspace.Run, duck-typed (run_dir / id) like Mode.run
        user_input: str,
        gateway: AgentGateway,
        capability_registry: CapabilityRegistry | None = None,
    ) -> ModeResult:
        """Run the emergent-planning pipeline on ``run`` and return a result.

        Args:
            run: The ``workspace.Run`` the pipeline executes under (provides
                ``run_dir`` for the store bundle + task board, and ``id``).
            user_input: The plan draft — a JSON object with ``title`` /
                ``objective`` when parseable, else free text folded into a
                minimal spec.
            gateway: The agent gateway; the plan report is rendered through its
                ``call``, and (for the production loop runner) its ``router``
                drives the planning loop.
            capability_registry: Optional grounded capability catalog the
                feasibility probe annotates against; ``None`` grades every task
                unreachable.

        Returns:
            A :class:`ModeResult` whose ``final_artifact`` is the rendered
            ``plan_report``.

        Raises:
            StageExecutionError: The final board is malformed — the gate is not
                opened (a malformed plan never reaches the human).
            ApprovalPendingError: The hard review gate suspended store-first
                (no approver, no stored grant) — a suspension, not a failure.
        """
        store = FileArtifactStore(root=run.run_dir / "artifacts")
        ctx = self._build_ctx(run, store, gateway, capability_registry)
        spec = self._derive_spec(user_input)
        board_file = board_path(run.run_dir)

        # Adapt the phase-03 board tools into loop-callables (single-point
        # side-effect gating lives inside the adapter) and assemble the hook
        # bundle whose should_stop guard is the "malformed never surfaces"
        # first gate: it reads the CURRENT board and denies termination while
        # the plan is form-invalid, feeding the violations back to steer.
        tools = tuple(as_loop_tool(tool, ctx=ctx, approve=self._approve) for tool in BOARD_TOOLS)
        hooks = LoopHooks(should_stop=self._make_form_guard(spec, board_file))

        await self._loop_runner.run_planning(
            ctx=ctx,
            board=read_board(board_file),
            tools=tools,
            hooks=hooks,
            user_input=user_input,
        )

        # Read-only feasibility annotation, then re-persist the annotated board.
        board = self._probe.annotate(read_board(board_file), ctx.capability_registry)
        write_board(board_file, board)

        # Deterministic belt-and-suspenders guard: a malformed final board is
        # rejected here, BEFORE the subject is persisted or the gate opened, so
        # a malformed plan never reaches the human review gate.
        plan = ExperimentPlan(spec=spec, board=board)
        report = EmergentPlanFormValidator.validate(plan)
        if not report.passed:
            raise StageExecutionError(
                "EmergentPlanOrchestrator: final board is malformed; refusing to open the "
                "review gate — " + "; ".join(v.message for v in report.violations)
            )

        plan_ref = ctx.artifact_store.put_json(
            kind="experiment_plan",
            obj=plan.model_dump(mode="json"),
            created_by=self.name,
            parent_ids=[],
        )

        # Hard review gate with a STABLE per-run request id so a re-run replays
        # the same pending record / stored grant (store-first).
        audit = StepAuditLoop(
            name="review_plan",
            subject_kind="experiment_plan",
            pack_builder=build_experiment_plan_review_pack,
            policy="hard",
            request=ApprovalRequest(
                id=f"approve_experiment_plan-{run.id}",
                intent="approve_experiment_plan",
                reason="Approve the emergent experiment plan before it is frozen.",
                triggered_by_policy="hard",
                created_at=datetime.now(tz=UTC),
            ),
            approve=self._approve,
        )
        # No approver + no stored grant → ApprovalPendingError (suspend, not
        # failure); a stored grant / injected approver → proceed.
        audit_ref = await audit.run(ctx)

        # Approved: content-addressed freeze (same board ⇒ same frozen id).
        frozen_ref = freeze_experiment_plan(
            plan,
            store,
            created_by=self.name,
            parent_ids=(plan_ref.id,),
        )

        render = await gateway.call(
            AgentCallSpec(
                agent_name="plan_report_renderer",
                input_artifact_ids=[frozen_ref.id],
                output_schema={},
                call_mode="structured",
            )
        )

        return ModeResult(
            mode_name=self.name,
            run_id=run.id,
            execution_id=derive_execution_id(run.id, run.run_dir / "executions"),
            stage_artifacts=(plan_ref, audit_ref, frozen_ref, render.output_artifact),
            final_artifact=render.output_artifact,
        )

    # ----------------------------------------------------------- internals

    def _build_ctx(
        self,
        run: Any,  # noqa: ANN401 — workspace.Run, duck-typed like Mode._build_ctx
        artifact_store: FileArtifactStore,
        gateway: AgentGateway,
        capability_registry: CapabilityRegistry | None,
    ) -> HarnessRunContext:
        """Reproduce the ``Mode._build_ctx`` store bundle inline.

        Deliberately duplicated (not extracted to a shared helper): ``Mode`` is
        deleted in phase-08, so extracting now would create a short-lived
        cross-module coupling. The bundle is the same one every harness stage
        expects — artifacts under ``run_dir/artifacts`` and the event / lineage
        / approval stores all sharing ``run_dir/harness.sqlite``. The concrete
        ``artifact_store`` is threaded in (not built here) so the caller can
        also hand it to :func:`freeze_experiment_plan`, which needs the
        ``FileArtifactStore`` concrete type.
        """
        db_path = run.run_dir / "harness.sqlite"
        return HarnessRunContext(
            run_id=run.id,
            workspace_root=run.run_dir,
            artifact_store=artifact_store,
            event_log=SQLiteEventLog(path=db_path),
            lineage_store=SQLiteArtifactLineageStore(path=db_path, artifact_store=artifact_store),
            agent_gateway=gateway,
            capability_registry=capability_registry,
            approval_store=SQLiteApprovalStore(path=db_path),
        )

    def _make_form_guard(self, spec: dict[str, Any], board_file: Any):  # noqa: ANN401,ANN202
        """Build the should_stop guard that vetoes terminating on a bad board.

        The returned guard reads the CURRENT on-disk board each time it is
        consulted and grades ``ExperimentPlan(spec, board)``: a passing report
        allows termination (``proceed``); a failing one denies it, feeding the
        violations back so the planning loop steers rather than surfaces a
        malformed plan.
        """

        async def should_stop(*, state: LoopState) -> HookOutcome:
            del state
            report = EmergentPlanFormValidator.validate(
                ExperimentPlan(spec=spec, board=read_board(board_file))
            )
            if report.passed:
                return HookOutcome.proceed()
            return HookOutcome.deny("; ".join(v.message for v in report.violations))

        return should_stop

    @staticmethod
    def _derive_spec(user_input: str) -> dict[str, Any]:
        """Derive the experiment spec dict from ``user_input``.

        A JSON object is used verbatim; any other text folds into a minimal
        ``{"title": …, "objective": …}`` so the spec half stays form-valid and
        only the task board drives valid/invalid.
        """
        try:
            parsed = json.loads(user_input)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        text = user_input.strip() or "experiment"
        return {"title": text, "objective": text}


class InteractiveLoopPlanRunner:
    """Production :class:`PlanLoopRunner` driving a phase-02 ``InteractiveLoop``.

    The **only** harness site importing ``molexp.agent.loops``; it does so
    **lazily** inside :meth:`run_planning` (mirroring ``Mode.run`` deferring
    ``molexp.workflow``) so ``import molexp.harness`` pulls no
    ``pydantic_ai`` / ``pydantic_graph``. It builds the loop on the router the
    05a gateway already owns (``ctx.agent_gateway.router``), injecting the plan
    tools and the ``EmergentPlanFormValidator``-backed ``should_stop`` guard the
    orchestrator hands in.
    """

    async def run_planning(
        self,
        *,
        ctx: HarnessRunContext,
        board: TaskBoard,
        tools: tuple[object, ...],
        hooks: LoopHooks,
        user_input: str,
    ) -> None:
        """Drive one emergent planning turn over the run's task board.

        Args:
            ctx: The harness run context; its ``agent_gateway`` supplies the
                :class:`Router`, and ``workspace_root`` roots the loop.
            board: The current board (unused directly — the loop mutates the
                board through the injected tools, which reach it via ``ctx``).
            tools: The ``as_loop_tool``-adapted plan tools handed to the loop.
            hooks: The hook bundle whose ``should_stop`` steers on a malformed
                board.
            user_input: The plan draft the planning turn opens on.

        Raises:
            StageExecutionError: ``ctx.agent_gateway`` exposes no ``router``.
        """
        del board  # the loop drives the board through the injected tools

        # Lazy import — keeps molexp.agent.loops out of `import molexp.harness`.
        from molexp.agent.events import AsyncIteratorEventSink
        from molexp.agent.execution_env import LocalExecutionEnv
        from molexp.agent.loops import InteractiveLoop, InteractiveLoopConfig
        from molexp.agent.runtime import AgentRuntime
        from molexp.agent.session import InMemorySessionStorage, Session

        gateway = ctx.agent_gateway
        if not isinstance(gateway, _RouterBackedGateway):
            raise StageExecutionError(
                "InteractiveLoopPlanRunner requires ctx.agent_gateway to expose a public "
                ".router accessor (RouterBackedAgentGateway or a fake exposing .router)"
            )

        runtime = AgentRuntime(
            session=Session(storage=InMemorySessionStorage()),
            router=gateway.router,
            execution_env=LocalExecutionEnv(scratch_dir=ctx.workspace_root / ".plan_scratch"),
        )
        loop = InteractiveLoop(
            config=InteractiveLoopConfig(workspace_root=ctx.workspace_root),
            hooks=hooks,
            tools=tuple(tools),
        )
        sink = AsyncIteratorEventSink()
        try:
            await loop.run(runtime=runtime, sink=sink, user_input=user_input)
        finally:
            await sink.close()
