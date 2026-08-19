"""Plan as a :class:`~molexp.workflow.WorkflowCompiler` graph.

Each LLM node is one :meth:`AgentGateway.call` (ReAct or structured).
Form failure is ``wf.loop`` / ``Next("continue")``, not an agent outer loop.

Imported lazily from :class:`PlanOrchestrator` so ``import molexp.harness``
does not load :mod:`molexp.workflow`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mollog import get_logger

from molexp.harness.errors import ApprovalPendingError, StageExecutionError
from molexp.harness.executors.local import LocalExecutor
from molexp.harness.plan import ExperimentPlan, freeze_experiment_plan, read_board, write_board
from molexp.harness.plan.bind_board import materialize_plan_for_realization
from molexp.harness.schemas import AgentCallSpec, ApprovalRequest
from molexp.harness.stages.assemble_knowledge_context import AssembleKnowledgeContext
from molexp.harness.stages.plan_reachability_probe import PlanReachabilityProbe
from molexp.harness.stages.realize_board import RealizeBoard
from molexp.harness.stages.review_pack_builders import build_experiment_plan_review_pack
from molexp.harness.stages.step_audit_loop import StepAuditLoop
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.validators import PlanFormValidator
from molexp.workflow import Next, WorkflowCompiler
from molexp.workflow.compiled import CompiledWorkflow

if TYPE_CHECKING:
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.executors import Executor
    from molexp.harness.gateways.call_runtime import AgentCallRuntime
    from molexp.harness.plan import TaskBoard
    from molexp.harness.stages.approval_gate import Approver

_LOG = get_logger(__name__)

__all__ = ["PlanBag", "PlanDraft", "compile_plan_workflow", "plan_loop_system_prompt"]

PlanDraft = Callable[..., Awaitable[None]]

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

_KNOWLEDGE_DIGEST_GUIDANCE = (
    "\n\n## Prior knowledge (workspace digest)\n"
    "A deterministic digest of this workspace's prior knowledge may follow. "
    "Ground the board in it: treat FailureAnalysis items as known pitfalls to avoid, "
    "Findings as established results not to re-derive, and Decisions/Constraints as "
    "standing choices to respect. When a prior item shapes a task, cite its path in "
    "the task description or acceptance criteria.\n\n"
)


def plan_loop_system_prompt(knowledge_digest: str | None = None) -> str:
    """Build the planning ReAct system prompt, optionally with a digest."""
    body = (knowledge_digest or "").strip()
    if not body:
        return _PLAN_LOOP_PREAMBLE
    return _PLAN_LOOP_PREAMBLE + _KNOWLEDGE_DIGEST_GUIDANCE + body


@dataclass
class PlanBag:
    """Closed-over state for one plan-workflow execution."""

    ctx: HarnessRunContext
    user_input: str
    spec: dict[str, Any]
    board_file: Path
    tools: tuple[object, ...]
    name: str
    probe: PlanReachabilityProbe
    draft: PlanDraft | None = None
    on_event: Callable[[object], Awaitable[None]] | None = None
    board_max_iters: int = 8
    do_realize: bool = True
    approve: Approver | None = None
    executor: Executor | None = None
    realize_attempts: int = 3
    realize_impl: Callable[..., Awaitable[Any]] | None = None
    run_id: str = ""
    steer_message: str = ""
    knowledge_id: str = ""
    plan_artifact_id: str = ""
    render_artifact_id: str = ""
    approval_pending: ApprovalPendingError | None = None
    stage_ids: list[str] = field(default_factory=list)


def compile_plan_workflow(bag: PlanBag) -> CompiledWorkflow:
    """Compile the plan graph: assemble → (draft ⟲ form) → probe → persist → render."""
    wf = WorkflowCompiler(name="plan", entry="assemble_knowledge")

    @wf.task
    async def assemble_knowledge() -> str:
        ref = await AssembleKnowledgeContext().run(bag.ctx)
        bag.knowledge_id = ref.id
        bag.stage_ids.append(ref.id)
        return ref.id

    @wf.task
    async def draft_board(value: str | None = None) -> str:
        del value
        if bag.draft is not None:
            await bag.draft(ctx=bag.ctx, user_input=bag.user_input)
            return "ok"
        return await _draft_via_agent_call(bag)

    @wf.task(depends_on=["draft_board"])
    async def form_check(draft_board: str = "") -> tuple[str, Next]:
        del draft_board
        report = PlanFormValidator.validate(
            ExperimentPlan(spec=bag.spec, board=read_board(bag.board_file)),
            require_feasibility=False,
        )
        if report.passed:
            return "ok", Next("exit")
        bag.steer_message = "; ".join(v.message for v in report.violations)
        return bag.steer_message, Next("continue")

    @wf.task
    async def probe() -> str:
        board: TaskBoard = bag.probe.annotate(
            read_board(bag.board_file), bag.ctx.capability_registry
        )
        write_board(bag.board_file, board)
        return "ok"

    @wf.task(depends_on=["probe"])
    async def persist_plan(probe: str = "") -> dict[str, Any]:
        del probe
        plan = ExperimentPlan(spec=bag.spec, board=read_board(bag.board_file))
        report = PlanFormValidator.validate(plan, require_feasibility=True)
        if not report.passed:
            return {
                "ok": False,
                "error": "final board is malformed; refusing to open the review gate — "
                + "; ".join(v.message for v in report.violations),
            }
        plan_ref = bag.ctx.artifact_store.put_json(
            kind="experiment_plan",
            obj=plan.model_dump(mode="json"),
            created_by=bag.name,
            parent_ids=[bag.knowledge_id] if bag.knowledge_id else [],
        )
        bag.stage_ids.append(plan_ref.id)
        bag.plan_artifact_id = plan_ref.id
        return {"ok": True, "id": plan_ref.id}

    @wf.task(depends_on=["persist_plan"])
    async def render_report(ok: bool = False, id: str = "") -> dict[str, Any]:
        if not ok or not id:
            return {"ok": False, "error": "experiment_plan was not persisted"}
        gateway = bag.ctx.agent_gateway
        if gateway is None:
            raise StageExecutionError("plan host did not publish AgentCall")
        render = await gateway.call(
            AgentCallSpec(
                agent_name="plan_report_renderer",
                input_artifact_ids=[id],
                output_schema={},
                call_mode="structured",
            )
        )
        bag.stage_ids.append(render.output_artifact.id)
        bag.render_artifact_id = render.output_artifact.id
        return {"ok": True, "id": render.output_artifact.id, "kind": render.output_artifact.kind}

    @wf.task(depends_on=["render_report"])
    async def review_gate(ok: bool = False, id: str = "") -> dict[str, Any]:
        if not ok or not id:
            return {"ok": False, "pending": False}
        from datetime import UTC, datetime

        audit = StepAuditLoop(
            name="review_plan",
            subject_kind="experiment_plan",
            pack_builder=build_experiment_plan_review_pack,
            policy="hard",
            request=ApprovalRequest(
                id=f"approve_experiment_plan-{bag.run_id}",
                intent="approve_experiment_plan",
                reason="Approve the experiment plan before it is frozen and realized.",
                triggered_by_policy="hard",
                created_at=datetime.now(tz=UTC),
            ),
            approve=bag.approve,
        )
        try:
            audit_ref = await audit.run(bag.ctx)
        except ApprovalPendingError as exc:
            bag.approval_pending = exc
            return {"ok": False, "pending": True}
        bag.stage_ids.append(audit_ref.id)
        return {"ok": True, "id": audit_ref.id, "pending": False}

    @wf.task(depends_on=["review_gate"])
    async def freeze(ok: bool = False, id: str = "", pending: bool = False) -> dict[str, Any]:
        del id
        if pending or not ok:
            return {"ok": False, "pending": pending}
        plan = ExperimentPlan(spec=bag.spec, board=read_board(bag.board_file))
        store = bag.ctx.artifact_store
        if not isinstance(store, FileArtifactStore):
            raise StageExecutionError("freeze requires a FileArtifactStore")
        parents = tuple(p for p in (bag.plan_artifact_id, bag.render_artifact_id) if p)
        frozen = freeze_experiment_plan(
            plan,
            store,
            created_by=bag.name,
            parent_ids=parents,
        )
        bag.stage_ids.append(frozen.id)
        return {"ok": True, "id": frozen.id, "pending": False}

    @wf.task(depends_on=["freeze"])
    async def realize(ok: bool = False, id: str = "", pending: bool = False) -> dict[str, Any]:
        if pending:
            return {"ok": False, "pending": True, "skipped": True}
        if not bag.do_realize:
            return {"ok": True, "skipped": True, "pending": False}
        if not ok or not id:
            return {"ok": False, "pending": False, "skipped": True}
        plan = ExperimentPlan(spec=bag.spec, board=read_board(bag.board_file))
        if bag.realize_impl is not None:
            ref = await bag.realize_impl(bag.ctx, plan, parent_ids=(id,))
        else:
            materialize_plan_for_realization(
                plan,
                bag.ctx.artifact_store,
                created_by=bag.name,
                parent_ids=(id,),
            )
            executor = bag.executor or LocalExecutor()
            ref = await RealizeBoard(executor, attempts=bag.realize_attempts).run(bag.ctx)
        bag.stage_ids.append(ref.id)
        return {"ok": True, "id": ref.id, "skipped": False, "pending": False}

    wf.control("assemble_knowledge", "draft_board")
    wf.loop(
        body=["draft_board"], until="form_check", max_iters=bag.board_max_iters, on_exit="probe"
    )
    return wf.compile()


async def _draft_via_agent_call(bag: PlanBag) -> str:
    """One ReAct AgentCall that mutates the disk board through injected tools."""
    from molexp.harness.gateways.call_runtime import AgentCallRuntime

    gateway = bag.ctx.agent_gateway
    if gateway is None:
        raise StageExecutionError("plan draft requires ctx.agent_gateway")

    digest: str | None = None
    knowledge_ref = bag.ctx.artifact_store.latest_by_kind("knowledge_context")
    if knowledge_ref is not None:
        try:
            digest = bag.ctx.artifact_store.get(knowledge_ref.id).decode("utf-8")
        except Exception as exc:
            _LOG.debug(f"[plan] knowledge_context read failed: {exc!r}")

    prompt_text = bag.user_input
    if bag.steer_message:
        prompt_text = (
            f"{bag.user_input}\n\nThe previous board was incomplete: {bag.steer_message}\n"
            "Fix the board with the board tools, then stop."
        )
    prompt_ref = bag.ctx.artifact_store.put_text(
        kind="prompt",
        text=prompt_text,
        created_by="plan_board",
        parent_ids=[],
    )

    observer = bag.on_event

    async def _on_event(event: object) -> None:
        if observer is None:
            return
        try:
            await observer(event)
        except Exception as exc:
            _LOG.debug(f"[plan] on_event failed: {exc!r}")

    runtime: AgentCallRuntime = AgentCallRuntime(
        tools=bag.tools,
        on_event=_on_event,
        system_prompt=plan_loop_system_prompt(digest),
        workspace_root=bag.ctx.workspace_root,
        operation_mode="chat",
    )
    await gateway.call(
        AgentCallSpec(
            agent_name="plan_board",
            input_artifact_ids=[prompt_ref.id],
            output_schema={},
            call_mode="agentic",
        ),
        runtime=runtime,
    )
    return "ok"
