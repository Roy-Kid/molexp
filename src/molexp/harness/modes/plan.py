"""Plan bundle — ``compose_plan`` + plan workflow.

:class:`Plan` is an object that composes plugins. It is not a Mode kernel
and does not subclass an Agent.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from molexp.harness.errors import StageExecutionError
from molexp.harness.host.compose import compose_plan
from molexp.harness.host.keys import Keys
from molexp.harness.host.plugins.tools import ToolBelt
from molexp.harness.plan import FROZEN_PLAN_KIND
from molexp.harness.plan.disk_board import DiskTaskBoard
from molexp.harness.plan_tools import BOARD_TOOLS, as_loop_tool
from molexp.harness.schemas import ModeResult
from molexp.harness.stages.plan_reachability_probe import PlanReachabilityProbe
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.workspace.utils import derive_execution_id

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from molexp.harness.executors import Executor
    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.host.host import Host
    from molexp.harness.host.plugin import Plugin
    from molexp.harness.modes.plan_workflow import PlanDraft
    from molexp.harness.registry.capability_registry import CapabilityRegistry
    from molexp.harness.schemas import PlanArtifactRef
    from molexp.harness.stages.approval_gate import Approver

    LoopEventObserver = Callable[[object], Awaitable[None]]

__all__ = ["Plan", "plan_loop_system_prompt"]

_PLAN_NAME = "plan"
_TITLE_SEPARATORS = ("，", "。", "；", ",", ";", ".", "?", "？", "!", "：", ":")  # noqa: RUF001


def plan_loop_system_prompt(knowledge_digest: str | None = None) -> str:
    """Re-export — planning ReAct system prompt."""
    from molexp.harness.modes.plan_workflow import plan_loop_system_prompt as _prompt

    return _prompt(knowledge_digest)


def _as_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    return {}


def _short_plan_title(text: str, *, max_len: int = 72) -> str:
    first = (text.splitlines()[0] if text else "").strip() or "experiment"
    for sep in _TITLE_SEPARATORS:
        if sep in first:
            head = first.split(sep, 1)[0].strip()
            if len(head) >= 2 and head != first:
                first = head
                break
    if len(first) <= max_len:
        return first
    return first[: max_len - 1].rstrip() + "…"


def derive_plan_spec(user_input: str) -> dict[str, Any]:
    """Seed the opaque plan ``spec`` from the operator draft."""
    try:
        parsed = json.loads(user_input)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    text = user_input.strip() or "experiment"
    return {
        "title": _short_plan_title(text),
        "objective": text,
        "raw_request": text,
    }


class Plan:
    """Plan bundle: mount the plan host and run the plan workflow.

    Options live on the instance. ``run`` takes the workspace Run and the
    model gateway. No Agent / Mode inheritance.
    """

    name = _PLAN_NAME

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
        plugins: tuple[Plugin, ...] = (),
    ) -> None:
        self.draft = draft
        self.probe = probe or PlanReachabilityProbe()
        self.approve = approve
        self.realize = realize
        self.executor = executor
        self.realize_attempts = realize_attempts
        self.on_loop_event = on_loop_event
        self.board_max_iters = board_max_iters
        self.plugins = plugins

    async def run(
        self,
        *,
        run: Any,  # noqa: ANN401 — workspace Run
        user_input: str,
        gateway: AgentGateway,
        capability_registry: CapabilityRegistry | None = None,
    ) -> ModeResult:
        """Run this bundle on *run* and return a :class:`ModeResult`."""
        host = compose_plan(
            run_id=run.id,
            run_dir=run.run_dir,
            gateway=gateway,
            capability_registry=capability_registry,
            extra=self.plugins,
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
        from molexp.harness.host.plugins.workflow import WorkflowHandle
        from molexp.harness.modes.plan_workflow import PlanBag, compile_plan_workflow
        from molexp.harness.plan import board_path
        from molexp.workflow import WorkflowRuntime
        from molexp.workflow.types import WorkflowResult

        ctx = host.as_run_context()
        store = ctx.artifact_store
        if not isinstance(store, FileArtifactStore):
            raise StageExecutionError("plan host did not publish a FileArtifactStore")
        if ctx.agent_gateway is None:
            raise StageExecutionError("plan host did not publish ctx.llm")
        spec = derive_plan_spec(user_input)
        board_file = board_path(run.run_dir)
        disk_board = DiskTaskBoard(board_file, artifact_store=store)
        belt = host.ctx.require(Keys.TOOLS)
        if not isinstance(belt, ToolBelt):
            raise StageExecutionError("plan host did not publish a ToolBelt")
        for tool in BOARD_TOOLS:
            belt.register(
                as_loop_tool(tool, ctx=ctx, board=disk_board, approve=self.approve),
                host.ctx,
            )
        bag = PlanBag(
            ctx=ctx,
            user_input=user_input,
            spec=spec,
            board_file=board_file,
            tools=belt.snapshot(),
            name=self.name,
            probe=self.probe,
            draft=self.draft,
            on_event=self.on_loop_event,
            board_max_iters=self.board_max_iters,
            do_realize=self.realize,
            approve=self.approve,
            executor=self.executor,
            realize_attempts=self.realize_attempts,
            run_id=run.id,
        )
        compiled = compile_plan_workflow(bag)
        scratch = run.run_dir / ".plan_scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        wf_handle = host.ctx.get(Keys.WORKFLOW)
        if isinstance(wf_handle, WorkflowHandle):
            raw = await wf_handle.execute(
                compiled,
                persist=True,
                run_dir=run.run_dir,
                scratch_root=scratch,
            )
        else:
            raw = await WorkflowRuntime().execute(
                compiled,
                persist=True,
                run_dir=run.run_dir,
                scratch_root=scratch,
            )
        if not isinstance(raw, WorkflowResult):
            raise StageExecutionError("plan workflow execute did not return a WorkflowResult")
        wf_result = raw
        if bag.approval_pending is not None:
            raise bag.approval_pending
        if wf_result.status != "succeeded":
            raise StageExecutionError(
                f"plan workflow failed: {getattr(wf_result, 'error', None) or wf_result.status}"
            )
        persist_out = _as_mapping(wf_result.outputs.get("persist_plan"))
        if not persist_out.get("ok"):
            raise StageExecutionError(
                "plan bundle: final board is malformed; refusing to open the "
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
