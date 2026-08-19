"""``PlanOrchestrator`` — compose the plan host and run the plan workflow.

The graph (including review / freeze / realize) lives in
:mod:`molexp.harness.modes.plan_workflow`. This module mounts plugins,
registers board tools on ``ctx.tools``, and re-raises a stored
:class:`ApprovalPendingError` after the graph returns.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from mollog import get_logger

from molexp.harness.errors import StageExecutionError
from molexp.harness.executors.local import LocalExecutor
from molexp.harness.host.compose import compose_plan
from molexp.harness.host.host import Host
from molexp.harness.host.keys import Keys
from molexp.harness.host.plugins.tools import ToolBelt
from molexp.harness.plan import FROZEN_PLAN_KIND, ExperimentPlan
from molexp.harness.plan.bind_board import materialize_plan_for_realization
from molexp.harness.plan.disk_board import DiskTaskBoard
from molexp.harness.plan_tools import BOARD_TOOLS, as_loop_tool
from molexp.harness.schemas import ModeResult
from molexp.harness.stages.approval_gate import Approver
from molexp.harness.stages.plan_reachability_probe import PlanReachabilityProbe
from molexp.harness.stages.realize_board import RealizeBoard
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.workspace.utils import derive_execution_id

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from molexp.harness.executors import Executor
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.modes.plan_workflow import PlanDraft
    from molexp.harness.registry.capability_registry import CapabilityRegistry
    from molexp.harness.schemas import PlanArtifactRef

    LoopEventObserver = Callable[[object], Awaitable[None]]

__all__ = [
    "PlanOrchestrator",
    "plan_loop_system_prompt",
]

_LOG = get_logger(__name__)


def _as_mapping(value: object) -> dict[str, Any]:
    """Coerce a workflow task output to a dict."""
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    return {}


def plan_loop_system_prompt(knowledge_digest: str | None = None) -> str:
    """Re-export — planning ReAct system prompt."""
    from molexp.harness.modes.plan_workflow import plan_loop_system_prompt as _prompt

    return _prompt(knowledge_digest)


class PlanOrchestrator:
    """Plan bundle driver: host + plan workflow.

    Args:
        draft: Optional test seam that writes a board instead of a ReAct call.
        probe: Feasibility annotator after the draft loop.
        approve: Optional approver for the hard review gate.
        realize: When True (default), run realization after freeze.
        executor: Executor for compile/tests; defaults to :class:`LocalExecutor`.
        realize_attempts: Per-task codegen self-repair budget.
        board_max_iters: ``wf.loop`` cap on draft ⟲ form_check.
    """

    name = "plan"

    def __init__(
        self,
        *,
        draft: PlanDraft | None = None,
        probe: PlanReachabilityProbe | None = None,
        approve: Approver | None = None,
        realize: bool = True,
        executor: Executor | None = None,
        realize_attempts: int = 3,
        on_loop_event: LoopEventObserver | None = None,
        board_max_iters: int = 8,
    ) -> None:
        self._draft = draft
        self._probe = probe or PlanReachabilityProbe()
        self._approve = approve
        self._realize = realize
        self._executor = executor
        self._realize_attempts = realize_attempts
        self._on_loop_event = on_loop_event
        self._board_max_iters = board_max_iters

    async def run(
        self,
        *,
        run: Any,  # noqa: ANN401
        user_input: str,
        gateway: AgentGateway,
        capability_registry: CapabilityRegistry | None = None,
    ) -> ModeResult:
        """Run the plan pipeline on ``run`` and return a result."""
        host = compose_plan(
            run_id=run.id,
            run_dir=run.run_dir,
            gateway=gateway,
            capability_registry=capability_registry,
        )
        try:
            return await self._run_on_host(host, run=run, user_input=user_input)
        finally:
            host.unload()

    async def _run_on_host(
        self,
        host: Host,
        *,
        run: Any,  # noqa: ANN401
        user_input: str,
    ) -> ModeResult:
        """Drive the mounted plan host. Caller owns ``unload``."""
        from molexp.harness.modes.plan_workflow import PlanBag, compile_plan_workflow
        from molexp.harness.plan import board_path

        ctx = host.as_run_context()
        store = ctx.artifact_store
        if not isinstance(store, FileArtifactStore):
            raise StageExecutionError("plan host did not publish a FileArtifactStore")
        if ctx.agent_gateway is None:
            raise StageExecutionError("plan host did not publish AgentCall")
        spec = self._derive_spec(user_input)
        board_file = board_path(run.run_dir)
        disk_board = DiskTaskBoard(board_file, artifact_store=store)
        belt = host.ctx.require(Keys.TOOLS)
        if not isinstance(belt, ToolBelt):
            raise StageExecutionError("plan host did not publish a ToolBelt")
        for tool in BOARD_TOOLS:
            belt.register(
                as_loop_tool(tool, ctx=ctx, board=disk_board, approve=self._approve),
                host.ctx,
            )
        bag = PlanBag(
            ctx=ctx,
            user_input=user_input,
            spec=spec,
            board_file=board_file,
            tools=belt.snapshot(),
            name=self.name,
            probe=self._probe,
            draft=self._draft,
            on_event=self._on_loop_event,
            board_max_iters=self._board_max_iters,
            do_realize=self._realize,
            approve=self._approve,
            executor=self._executor,
            realize_attempts=self._realize_attempts,
            realize_impl=self._run_realization,
            run_id=run.id,
        )
        compiled = compile_plan_workflow(bag)
        scratch: Path = run.run_dir / ".plan_scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        from molexp.workflow import WorkflowRuntime

        wf_result = await WorkflowRuntime().execute(
            compiled,
            persist=True,
            run_dir=run.run_dir,
            scratch_root=scratch,
        )
        if bag.approval_pending is not None:
            raise bag.approval_pending
        if wf_result.status != "succeeded":
            raise StageExecutionError(
                f"plan workflow failed: {getattr(wf_result, 'error', None) or wf_result.status}"
            )
        persist_out = _as_mapping(wf_result.outputs.get("persist_plan"))
        if not persist_out.get("ok"):
            raise StageExecutionError(
                "PlanOrchestrator: final board is malformed; refusing to open the "
                f"review gate — {persist_out.get('error', persist_out)}"
            )
        render_out = _as_mapping(wf_result.outputs.get("render_report"))
        if not render_out.get("ok"):
            raise StageExecutionError("plan report renderer did not produce a plan_report")

        def _ref(kind: str) -> PlanArtifactRef:
            latest = store.latest_by_kind(kind)
            if latest is None:
                raise StageExecutionError(f"missing artifact {kind}")
            return latest

        knowledge_ref = store.latest_by_kind("knowledge_context")
        plan_ref = _ref("experiment_plan")
        report_ref = _ref("plan_report")
        audit_ref = store.latest_by_kind("review_pack")
        frozen_ref = store.latest_by_kind(FROZEN_PLAN_KIND)
        realize_out = _as_mapping(wf_result.outputs.get("realize"))
        exec_ref = store.latest_by_kind("execution_result") if realize_out.get("ok") else None

        stage_artifacts: list[Any] = [
            *([knowledge_ref] if knowledge_ref is not None else []),
            plan_ref,
            report_ref,
            *([audit_ref] if audit_ref is not None else []),
            *([frozen_ref] if frozen_ref is not None else []),
            *([exec_ref] if exec_ref is not None else []),
        ]
        final = exec_ref or report_ref
        return ModeResult(
            mode_name=self.name,
            run_id=run.id,
            execution_id=derive_execution_id(run.id, run.run_dir / "executions"),
            stage_artifacts=tuple(a for a in stage_artifacts if a is not None),
            final_artifact=final,
        )

    async def _run_realization(
        self,
        ctx: Any,  # noqa: ANN401
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

    @staticmethod
    def _derive_spec(user_input: str) -> dict[str, Any]:
        """Seed the opaque plan ``spec`` from the operator draft."""
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

    _TITLE_SEPARATORS = ("，", "。", "；", ",", ";", ".", "?", "？", "!", "：", ":")  # noqa: RUF001

    @staticmethod
    def _short_plan_title(text: str, *, max_len: int = 72) -> str:
        """Human chrome title: first clause / line, truncated — not the whole draft."""
        first = (text.splitlines()[0] if text else "").strip() or "experiment"
        for sep in PlanOrchestrator._TITLE_SEPARATORS:
            if sep in first:
                head = first.split(sep, 1)[0].strip()
                if len(head) >= 2 and head != first:
                    first = head
                    break
        if len(first) <= max_len:
            return first
        return first[: max_len - 1].rstrip() + "…"
