"""``PlanOrchestrator`` — two-phase planning pipeline.

**Phase 1 — interactive planning:** an agent loop drives a task board with
board tools; a form guard blocks stop while the board is incomplete; a hard
review gate freezes the approved plan (store-first suspend when no grant).

**Phase 2 — deterministic realization:** the frozen board is projected to a
``bound_workflow`` + ``experiment_spec``, then :class:`RealizeBoard` map→reduce
→compile produces ``workflow_source`` / ``test_source`` / compile result.

Resume correctness for the human gate rides on store-first replay, not a
linear stage ledger.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mollog import get_logger

from molexp.agent.loops.hooks import HookOutcome, LoopHooks, LoopState
from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.errors import StageExecutionError
from molexp.harness.executors.local import LocalExecutor
from molexp.harness.plan import (
    ExperimentPlan,
    board_path,
    freeze_experiment_plan,
    read_board,
    write_board,
)
from molexp.harness.plan.bind_board import materialize_plan_for_realization
from molexp.harness.plan.disk_board import DiskTaskBoard
from molexp.harness.plan_tools import BOARD_TOOLS, as_loop_tool
from molexp.harness.schemas import AgentCallSpec, ApprovalRequest, ModeResult
from molexp.harness.stages.approval_gate import Approver
from molexp.harness.stages.plan_reachability_probe import PlanReachabilityProbe
from molexp.harness.stages.realize_board import RealizeBoard
from molexp.harness.stages.review_pack_builders import build_experiment_plan_review_pack
from molexp.harness.stages.step_audit_loop import StepAuditLoop
from molexp.harness.store.approval_store import SQLiteApprovalStore
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore
from molexp.harness.validators import PlanFormValidator
from molexp.workspace.utils import derive_execution_id

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from molexp.agent.router import Router
    from molexp.harness.executors import Executor
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.plan import TaskBoard
    from molexp.harness.registry.capability_registry import CapabilityRegistry
    from molexp.harness.schemas import PlanArtifactRef

    #: Observer for the planning loop's event stream. Typed as ``object`` on
    #: purpose: the concrete event is an agent-layer ``AgentEvent``, and harness
    #: must not widen its single sanctioned agent edge (``agent.router.Router``)
    #: just to name it. Services inject the real projector.
    LoopEventObserver = Callable[[object], Awaitable[None]]

__all__ = ["InteractiveLoopPlanRunner", "PlanLoopRunner", "PlanOrchestrator"]

_LOG = get_logger(__name__)

# System guidance for the planning InteractiveLoop (board tools, not free prose).
_PLAN_LOOP_PREAMBLE = (
    "You are a computational-chemistry experiment PLANNER. "
    "Your job is to decompose the user's research intent into a concrete task board "
    "using the board tools.\n\n"
    "Required method:\n"
    "1. Call list_tasks to see the current board.\n"
    "2. Use place_task for each ordered step (build system → parameterize → "
    "simulate/analyze → measure). Every task MUST have non-empty acceptance "
    "criteria (strings the realization phase can test).\n"
    "3. Prefer propose_plan_patch with a tasks list when placing several tasks.\n"
    "4. Use molmcp tools (molcrafts_packages / outline / open / compose) to ground "
    "each step on a real API — never invent symbols (ok=false means do not use it).\n"
    "5. When the board is complete and every task has acceptance criteria, stop.\n"
    "Do not write workflow Python yourself; realization codegen happens after approval."
)


@runtime_checkable
class PlanLoopRunner(Protocol):
    """Inversion seam: drive the planning loop over a task board."""

    async def run_planning(
        self,
        *,
        ctx: HarnessRunContext,
        board: TaskBoard,
        tools: tuple[object, ...],
        hooks: LoopHooks,
        user_input: str,
    ) -> None: ...


@runtime_checkable
class _RouterBackedGateway(Protocol):
    @property
    def router(self) -> Router: ...


class PlanOrchestrator:
    """Two-phase plan pipeline: interactive board → freeze → realize.

    Args:
        loop_runner: Planning-loop driver; defaults to
            :class:`InteractiveLoopPlanRunner`.
        probe: Feasibility annotator after the planning loop.
        approve: Optional approver for the hard review gate.
        realize: When True (default), run Phase 2 after freeze.
        executor: Executor for Phase 2 compile/tests; defaults to
            :class:`LocalExecutor`.
        realize_attempts: Per-task codegen self-repair budget.
    """

    name = "plan"

    def __init__(
        self,
        *,
        loop_runner: PlanLoopRunner | None = None,
        probe: PlanReachabilityProbe | None = None,
        approve: Approver | None = None,
        realize: bool = True,
        executor: Executor | None = None,
        realize_attempts: int = 3,
        on_loop_event: LoopEventObserver | None = None,
    ) -> None:
        # ``on_loop_event`` wires live thinking/tool stream for UI (services inject).
        self._loop_runner: PlanLoopRunner = loop_runner or InteractiveLoopPlanRunner(
            on_event=on_loop_event
        )
        self._probe = probe or PlanReachabilityProbe()
        self._approve = approve
        self._realize = realize
        self._executor = executor
        self._realize_attempts = realize_attempts

    async def run(
        self,
        *,
        run: Any,  # noqa: ANN401
        user_input: str,
        gateway: AgentGateway,
        capability_registry: CapabilityRegistry | None = None,
    ) -> ModeResult:
        """Run the plan pipeline on ``run`` and return a result."""
        store = FileArtifactStore(root=run.run_dir / "artifacts")
        ctx = self._build_ctx(run, store, gateway, capability_registry)
        spec = self._derive_spec(user_input)
        board_file = board_path(run.run_dir)
        disk_board = DiskTaskBoard(board_file, artifact_store=store)

        tools = tuple(
            as_loop_tool(tool, ctx=ctx, board=disk_board, approve=self._approve)
            for tool in BOARD_TOOLS
        )
        hooks = LoopHooks(should_stop=self._make_form_guard(spec, board_file))

        await self._loop_runner.run_planning(
            ctx=ctx,
            board=read_board(board_file),
            tools=tools,
            hooks=hooks,
            user_input=user_input,
        )

        board = self._probe.annotate(read_board(board_file), ctx.capability_registry)
        write_board(board_file, board)

        plan = ExperimentPlan(spec=spec, board=board)
        report = PlanFormValidator.validate(plan, require_feasibility=True)
        if not report.passed:
            raise StageExecutionError(
                "PlanOrchestrator: final board is malformed; refusing to open the "
                "review gate — " + "; ".join(v.message for v in report.violations)
            )

        plan_ref = ctx.artifact_store.put_json(
            kind="experiment_plan",
            obj=plan.model_dump(mode="json"),
            created_by=self.name,
            parent_ids=[],
        )

        # Render the operator-facing plan book *before* the review gate so the
        # agent answer is a filled document (not an empty 12-section shell).
        # Input is the mutable experiment_plan; freeze still happens after grant.
        render = await gateway.call(
            AgentCallSpec(
                agent_name="plan_report_renderer",
                input_artifact_ids=[plan_ref.id],
                output_schema={},
                call_mode="structured",
            )
        )

        audit = StepAuditLoop(
            name="review_plan",
            subject_kind="experiment_plan",
            pack_builder=build_experiment_plan_review_pack,
            policy="hard",
            request=ApprovalRequest(
                id=f"approve_experiment_plan-{run.id}",
                intent="approve_experiment_plan",
                reason="Approve the experiment plan before it is frozen and realized.",
                triggered_by_policy="hard",
                created_at=datetime.now(tz=UTC),
            ),
            approve=self._approve,
        )
        audit_ref = await audit.run(ctx)

        frozen_ref = freeze_experiment_plan(
            plan,
            store,
            created_by=self.name,
            parent_ids=(plan_ref.id, render.output_artifact.id),
        )

        stage_artifacts: list[Any] = [plan_ref, render.output_artifact, audit_ref, frozen_ref]
        final = render.output_artifact

        if self._realize:
            _LOG.info(f"[plan {run.id}] phase-2 realization starting")
            exec_ref = await self._run_realization(ctx, plan, parent_ids=(frozen_ref.id,))
            stage_artifacts.append(exec_ref)
            final = exec_ref

        return ModeResult(
            mode_name=self.name,
            run_id=run.id,
            execution_id=derive_execution_id(run.id, run.run_dir / "executions"),
            stage_artifacts=tuple(stage_artifacts),
            final_artifact=final,
        )

    async def _run_realization(
        self,
        ctx: HarnessRunContext,
        plan: ExperimentPlan,
        *,
        parent_ids: tuple[str, ...],
    ) -> PlanArtifactRef:
        """Materialize bound artifacts then run :class:`RealizeBoard`."""
        materialize_plan_for_realization(
            plan,
            ctx.artifact_store,
            created_by=self.name,
            parent_ids=parent_ids,
        )
        executor = self._executor or LocalExecutor()
        stage = RealizeBoard(executor, attempts=self._realize_attempts)
        return await stage.run(ctx)

    def _build_ctx(
        self,
        run: Any,  # noqa: ANN401
        artifact_store: FileArtifactStore,
        gateway: AgentGateway,
        capability_registry: CapabilityRegistry | None,
    ) -> HarnessRunContext:
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
        """should_stop guard — feasibility not required mid-loop (probe is post-loop)."""

        async def should_stop(*, state: LoopState) -> HookOutcome:
            del state
            report = PlanFormValidator.validate(
                ExperimentPlan(spec=spec, board=read_board(board_file)),
                require_feasibility=False,
            )
            if report.passed:
                return HookOutcome.proceed()
            return HookOutcome.deny("; ".join(v.message for v in report.violations))

        return should_stop

    @staticmethod
    def _derive_spec(user_input: str) -> dict[str, Any]:
        """Seed the opaque plan ``spec`` from the operator draft.

        **Does not** invent science — only splits a short chrome title from the
        full objective. Previously ``title == objective == entire draft``, so
        the review UI looked like a finished plan book whose only content was
        the raw prompt repeated three times (title / blurb / Goal).
        """
        try:
            parsed = json.loads(user_input)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        text = user_input.strip() or "experiment"
        title = PlanOrchestrator._short_plan_title(text)
        return {
            "title": title,
            "objective": text,
            "raw_request": text,
        }

    #: Clause separators for the chrome title. The fullwidth twins of the ASCII
    #: marks are deliberate (research intent is routinely typed in Chinese), not
    #: homoglyph typos — hence the blanket RUF001 waiver on this one line.
    _TITLE_SEPARATORS = ("，", "。", "；", ",", ";", ".", "?", "？", "!", "：", ":")  # noqa: RUF001

    @staticmethod
    def _short_plan_title(text: str, *, max_len: int = 72) -> str:
        """Human chrome title: first clause / line, truncated — not the whole draft."""
        first = (text.splitlines()[0] if text else "").strip() or "experiment"
        for sep in PlanOrchestrator._TITLE_SEPARATORS:
            if sep in first:
                head = first.split(sep, 1)[0].strip()
                # Chinese clauses are often short (e.g. 「创建一个项目」); keep any
                # non-trivial head so the full draft is not used as the title.
                if len(head) >= 2 and head != first:
                    first = head
                    break
        if len(first) <= max_len:
            return first
        return first[: max_len - 1].rstrip() + "…"


class InteractiveLoopPlanRunner:
    """Production :class:`PlanLoopRunner` driving an :class:`InteractiveLoop`.

    Optional ``on_event`` is a best-effort observer for live UX (e.g. project
    thinking / tool events into an agent-task transcript). It must never raise
    into the planning loop — failures are logged and swallowed here.
    """

    def __init__(self, *, on_event: LoopEventObserver | None = None) -> None:
        self._on_event = on_event

    async def run_planning(
        self,
        *,
        ctx: HarnessRunContext,
        board: TaskBoard,
        tools: tuple[object, ...],
        hooks: LoopHooks,
        user_input: str,
    ) -> None:
        del board  # disk board is reached through injected tools

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
        # Plan board tools + chat ops surface (no default ensure/land).
        loop = InteractiveLoop(
            config=InteractiveLoopConfig(
                workspace_root=ctx.workspace_root,
                system_prompt=_PLAN_LOOP_PREAMBLE,
                operation_mode="chat",
            ),
            hooks=hooks,
            tools=tuple(tools),
        )
        observer = self._on_event

        async def _sink(event: Any) -> None:  # noqa: ANN401 — AgentEvent
            if observer is None:
                return
            try:
                await observer(event)
            except Exception as exc:
                _LOG.debug(f"[plan-loop] on_event failed: {exc!r}")

        await loop.run(runtime=runtime, sink=_sink, user_input=user_input)
