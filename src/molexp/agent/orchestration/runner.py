"""AgentRunner — per-turn pipeline.

One ``run_turn`` builds a :class:`ContextPacket`, calls
:meth:`ModelClient.complete`, dispatches any tool calls, and loops
until a final text response or the tool-loop ceiling is hit.

In :attr:`AgentMode.PLAN` a text-only response is treated as the
proposed plan; the runner emits :class:`PlanCreated` and parks on
:class:`PlanStateMachine` until the user resolves it.
"""

from __future__ import annotations

import dataclasses
import json
import secrets
from dataclasses import dataclass, field
from typing import Any

from molexp.agent._serialize import to_jsonable as _jsonable
from molexp.agent.context.manager import ContextManager, DefaultContextManager
from molexp.agent.context.packet import ContextBuildRequest, ContextPacket
from molexp.agent.model import ModelClient, ModelRequest, ModelResponse
from molexp.agent.orchestration.chat import ChatGateway
from molexp.agent.orchestration.events import (
    ContextBuilt,
    FailureRecorded,
    ModelRequested,
    ModelResponded,
    PlanCreated,
    PlanDecided,
    SessionStarted,
    ToolCallCompleted,
    ToolCallRequested,
    TurnStarted,
)
from molexp.agent.orchestration.gates import (
    SessionApprovalGate,
    SessionChatGateway,
)
from molexp.agent.orchestration.plan import (
    PlanState,
    render_reject_feedback,
)
from molexp.agent.observability.evals import Evaluator, NoopEvaluator
from molexp.agent.orchestration.session import AgentSession
from molexp.agent.recovery.constraints import ConstraintSet
from molexp.agent.recovery.retry import (
    NoRetryPolicy,
    RecoveryPolicy,
    SimpleRetryPolicy,
)
from molexp.agent.state.sessions import SessionMetadata, SessionStore
from molexp.agent.tools.dispatcher import ToolDispatcher
from molexp.agent.tools.policy import PERMISSIVE_POLICY, ToolPolicy
from molexp.agent.tools.registry import ToolRegistry
from molexp.agent.tools.spec import ToolContext, ToolResult
from molexp.agent.types import (
    AgentFailure,
    AgentMode,
    FailureKind,
    Goal,
    Message,
    SessionStatus,
    WorkflowPreview,
    utc_now,
)


_MISSING: Any = object()


@dataclass
class AgentRunner:
    """Run one turn end-to-end: context → model → tools → state."""

    model: ModelClient
    registry: ToolRegistry
    store: SessionStore
    workspace: Any = None
    dispatcher: ToolDispatcher | None = None
    context_manager: ContextManager = field(default_factory=DefaultContextManager)
    policy: ToolPolicy = PERMISSIVE_POLICY
    constraints: ConstraintSet = field(default_factory=ConstraintSet)
    recovery: RecoveryPolicy = field(default_factory=SimpleRetryPolicy)
    evaluator: Evaluator = field(default_factory=NoopEvaluator)
    base_system_prompt: str = ""
    workspace_addendum: str = ""

    def __post_init__(self) -> None:
        if self.dispatcher is None:
            self.dispatcher = ToolDispatcher(self.registry)

    # ---------------------------------------------------------------- driver

    async def drive_session(self, session: AgentSession) -> None:
        """Pull inbound user messages and run turns until cancelled.

        Spawned by :class:`AgentService` as a background task.
        """

        await self._publish_and_persist(session, SessionStarted(
            session_id=session.session_id,
            goal_description=session.goal.description,
        ))
        turn_count = await self._drive_loop(session)
        await self._finalize_session(session, turn_count)

    async def _drive_loop(self, session: AgentSession) -> int:
        """Run turns until cancelled / failed; return the turn count."""

        history: list[Message] = []
        if session.goal.description:
            history.append(Message(role="user", content=session.goal.description))
            self.store.append_messages(session.session_id, [history[0]])

        turn_count = 0
        while True:
            if not history:
                inbound = await session.next_inbound()
                if inbound is None:  # session cancelled
                    return turn_count
                history.append(Message(role="user", content=inbound.content))
                self.store.append_messages(session.session_id, [history[-1]])
            if turn_count >= self.constraints.max_turns:
                await self._publish_and_persist(
                    session,
                    FailureRecorded(
                        turn_id="",
                        failure=AgentFailure(
                            kind=FailureKind.INTERNAL_ERROR,
                            message=(
                                f"Session exceeded {self.constraints.max_turns} turns"
                            ),
                        ),
                    ),
                )
                session.status = SessionStatus.FAILED
                self._flush_metadata(session, summary="failed: turn cap exceeded")
                return turn_count
            turn_count += 1
            try:
                history = await self.run_turn(session, history)
            except _SessionCancelled:
                return turn_count
            except Exception as exc:  # noqa: BLE001 — normalize at boundary
                await self._publish_and_persist(
                    session,
                    FailureRecorded(
                        turn_id="",
                        failure=AgentFailure(
                            kind=FailureKind.INTERNAL_ERROR,
                            message=f"{type(exc).__name__}: {exc}",
                        ),
                    ),
                )
                session.status = SessionStatus.FAILED
                self._flush_metadata(session, summary=f"failed: {exc!r}")
                return turn_count
            inbound = await session.next_inbound()
            if inbound is None:
                return turn_count
            history.append(Message(role="user", content=inbound.content))
            self.store.append_messages(session.session_id, [history[-1]])

    # ------------------------------------------------------------------ turn

    async def run_turn(
        self,
        session: AgentSession,
        history: list[Message],
    ) -> list[Message]:
        """Execute one full turn (model + optional tool loop)."""

        turn_id = _new_turn_id()
        session.status = SessionStatus.RUNNING
        await self._publish_and_persist(
            session,
            TurnStarted(session_id=session.session_id, turn_id=turn_id),
        )
        chat_gateway: ChatGateway = SessionChatGateway(session)
        approval_gate = SessionApprovalGate(session, turn_id=turn_id)
        dispatcher = self.dispatcher
        assert dispatcher is not None
        tool_schemas = await dispatcher.discover(self.workspace, self.policy)

        for _ in range(self.constraints.max_tool_calls_per_turn):
            packet = await self._build_context(session, turn_id, history)
            await self._publish_and_persist(
                session,
                ContextBuilt(
                    turn_id=turn_id,
                    used_chars=packet.budget.used_chars,
                    diagnostics=tuple(packet.diagnostics),
                ),
            )
            await self._publish_and_persist(
                session,
                ModelRequested(turn_id=turn_id, model_name=self.model.name),
            )
            request = ModelRequest(
                session_id=session.session_id,
                turn_id=turn_id,
                system=packet.system,
                messages=tuple(packet.messages),
                tools=tool_schemas,
            )
            response = await self._complete_with_retry(request)
            await self._publish_and_persist(
                session,
                ModelResponded(
                    turn_id=turn_id,
                    finish_reason=response.finish_reason,
                    usage=response.usage,
                ),
            )

            if response.text:
                msg = Message(role="assistant", content=response.text)
                history.append(msg)
                self.store.append_messages(session.session_id, [msg])

            if response.tool_calls:
                history = await self._run_tool_calls(
                    session=session,
                    turn_id=turn_id,
                    response=response,
                    history=history,
                    chat_gateway=chat_gateway,
                    dispatcher=dispatcher,
                    gate=approval_gate,
                )
                continue

            # No tool calls — either plan emission or final response.
            if session.goal.mode is AgentMode.PLAN and response.text:
                history = await self._handle_plan_emission(
                    session=session,
                    turn_id=turn_id,
                    plan_text=response.text,
                    history=history,
                )
                if session.plan.state is PlanState.PLAN_REJECTED:
                    # Reset to PLAN_REQUESTED so the next iteration
                    # asks the model to revise.
                    session.plan = session.plan.request_plan()
                    continue
                # Approved or unsupported preview — fall through.
            # Final assistant response; turn ends.
            session.status = SessionStatus.RUNNING
            self._flush_metadata(session, summary=response.text[:200])
            return history

        # Exceeded loop ceiling — record failure but don't crash session.
        await self._publish_and_persist(
            session,
            FailureRecorded(
                turn_id=turn_id,
                failure=AgentFailure(
                    kind=FailureKind.INTERNAL_ERROR,
                    message=(
                        f"Tool-call loop exceeded "
                        f"{self.constraints.max_tool_calls_per_turn} rounds"
                    ),
                ),
            ),
        )
        session.status = SessionStatus.FAILED
        self._flush_metadata(session, summary="failed: tool loop exceeded")
        return history

    # ---------------------------------------------------------------- helpers

    async def _complete_with_retry(self, request: ModelRequest) -> ModelResponse:
        """Call ``model.complete`` with :class:`RecoveryPolicy` retries.

        The default :class:`SimpleRetryPolicy` retries one transient
        ``MODEL_ERROR`` after a short delay; other failure kinds give
        up immediately. Re-raises the last exception when the policy
        declines to retry, so the outer ``drive_session`` catch
        records it as a typed failure.
        """

        attempt = 0
        while True:
            try:
                return await self.model.complete(request)
            except Exception as exc:  # noqa: BLE001 — policy decides what's transient
                failure = AgentFailure(
                    kind=FailureKind.MODEL_ERROR,
                    message=f"{type(exc).__name__}: {exc}",
                )
                decision = self.recovery.on_failure(failure, attempt)
                if not decision.retry:
                    raise
                if decision.delay_seconds > 0:
                    import asyncio

                    await asyncio.sleep(decision.delay_seconds)
                attempt += 1

    async def _build_context(
        self,
        session: AgentSession,
        turn_id: str,
        history: list[Message],
    ) -> ContextPacket:
        request = ContextBuildRequest(
            session_id=session.session_id,
            turn_id=turn_id,
            base_system=self.base_system_prompt,
            workspace_addendum=self.workspace_addendum,
            skill_addendum="",
            instructions_override=session.goal.instructions_override,
            history=tuple(history),
        )
        return await self.context_manager.build(request)

    async def _run_tool_calls(
        self,
        *,
        session: AgentSession,
        turn_id: str,
        response: ModelResponse,
        history: list[Message],
        chat_gateway: ChatGateway,
        dispatcher: ToolDispatcher,
        gate: SessionApprovalGate,
    ) -> list[Message]:
        new_msgs: list[Message] = []
        for call in response.tool_calls:
            await self._publish_and_persist(
                session,
                ToolCallRequested(
                    turn_id=turn_id,
                    call_id=call.id,
                    tool_name=call.name,
                    arguments=dict(call.arguments),
                ),
            )
            ctx = ToolContext(
                workspace=self.workspace,
                session_id=session.session_id,
                turn_id=turn_id,
                chat=chat_gateway,
            )
            result = await dispatcher.dispatch(call, ctx, self.policy, gate=gate)
            value_jsonable = _jsonable(result.value)
            await self._publish_and_persist(
                session,
                ToolCallCompleted(
                    turn_id=turn_id,
                    call_id=call.id,
                    tool_name=call.name,
                    ok=result.ok,
                    value=value_jsonable,
                    error=result.error,
                    artifacts=tuple(result.artifacts),
                    metadata=dict(result.metadata),
                ),
            )
            new_msgs.append(
                Message(
                    role="tool",
                    name=call.name,
                    content=_render_tool_payload(result, value_jsonable),
                    metadata={"call_id": call.id, "ok": result.ok},
                )
            )
        history.extend(new_msgs)
        if new_msgs:
            self.store.append_messages(session.session_id, new_msgs)
        return history

    async def _handle_plan_emission(
        self,
        *,
        session: AgentSession,
        turn_id: str,
        plan_text: str,
        history: list[Message],
    ) -> list[Message]:
        request_id = _new_request_id()
        plan_markdown, preview = _extract_plan(plan_text)
        session.plan = session.plan.request_plan().emit_plan(
            request_id=request_id,
            plan_markdown=plan_markdown,
            preview=preview,
        )
        session.status = SessionStatus.AWAITING_PLAN_DECISION
        await self._publish_and_persist(
            session,
            PlanCreated(
                turn_id=turn_id,
                request_id=request_id,
                plan_markdown=plan_markdown,
                workflow_preview=preview,
            ),
        )
        # Park until respond_plan() flips the plan state machine.
        await session.wait_plan_decision()
        if session.status is SessionStatus.CANCELLED:
            raise _SessionCancelled()
        plan_state = session.plan
        approved = plan_state.state is PlanState.PLAN_APPROVED
        decided = PlanDecided(
            request_id=plan_state.last_request_id or request_id,
            approved=approved,
            feedback=plan_state.last_feedback,
            edited_plan=plan_state.edited_plan,
            edited_workflow_ir=plan_state.edited_workflow_ir,
        )
        await self._publish_and_persist(session, decided)
        self._write_plan_checkpoint(session, turn_id, plan_state, decided)
        if approved:
            session.goal = _goal_in_mode(session.goal, AgentMode.CHAT)
            session.plan = session.plan.reset_to_chat()
        else:
            synthetic = Message(
                role="user",
                content=render_reject_feedback(plan_state.last_feedback),
            )
            history.append(synthetic)
            self.store.append_messages(session.session_id, [synthetic])
        session.status = SessionStatus.RUNNING
        return history

    async def _publish_and_persist(
        self, session: AgentSession, event: Any
    ) -> None:
        await session.bus.publish(event)
        self.store.append_event(session.session_id, event)

    def _flush_metadata(self, session: AgentSession, summary: str = "") -> None:
        meta = SessionMetadata(
            session_id=session.session_id,
            goal=session.goal,
            status=session.status,
            updated_at=utc_now(),
            summary=summary,
        )
        self.store.write_metadata(meta)

    async def _finalize_session(
        self, session: AgentSession, turn_count: int
    ) -> None:
        """Hand the terminal session to the configured :class:`Evaluator`.

        The eval result is persisted as a per-session checkpoint so
        replay/audit tooling can reconstruct outcome scoring without
        re-running the model. Default :class:`NoopEvaluator` records
        nothing of interest but keeps the call site uniform.
        """

        result = await self.evaluator.evaluate(
            session.session_id,
            {
                "status": session.status.value,
                "turns": turn_count,
                "goal": session.goal.description,
            },
        )
        self.store.write_checkpoint(
            session.session_id,
            "eval",
            {
                "evaluator": result.name,
                "score": result.score,
                "passed": result.passed,
                "details": dict(result.details),
                "ts": utc_now().isoformat(),
            },
        )

    def _write_plan_checkpoint(
        self,
        session: AgentSession,
        turn_id: str,
        plan_state,
        decided: PlanDecided,
    ) -> None:
        self.store.write_checkpoint(
            session.session_id,
            turn_id,
            {
                "turn_id": turn_id,
                "kind": "plan_decision",
                "state": plan_state.state.value,
                "request_id": plan_state.last_request_id,
                "plan_markdown": plan_state.last_plan_markdown,
                "approved": decided.approved,
                "feedback": decided.feedback,
                "edited_plan": decided.edited_plan,
                "edited_workflow_ir": decided.edited_workflow_ir,
                "ts": decided.ts.isoformat(),
            },
        )


# ---------------------------------------------------------------- internals


class _SessionCancelled(Exception):
    """Internal signal: the session was cancelled while a turn was running."""


def _new_turn_id() -> str:
    return f"turn_{secrets.token_hex(4)}"


def _new_request_id() -> str:
    return f"plan_{secrets.token_hex(4)}"


def _render_tool_payload(result: ToolResult, value_jsonable: Any = _MISSING) -> str:
    """Render a :class:`ToolResult` into the ``content`` of a tool message.

    Pass ``value_jsonable`` when the caller has already coerced
    ``result.value`` via :func:`_jsonable`, so it isn't recomputed.
    """

    if result.ok:
        coerced = _jsonable(result.value) if value_jsonable is _MISSING else value_jsonable
        return json.dumps(coerced, ensure_ascii=False)
    error = result.error
    if error is None:
        return json.dumps({"ok": False})
    return json.dumps(
        {
            "ok": False,
            "error": {
                "kind": error.kind.value,
                "message": error.message,
                "detail": _jsonable(error.detail),
            },
        },
        ensure_ascii=False,
    )


def _extract_plan(text: str) -> tuple[str, WorkflowPreview]:
    """Pull plan markdown + workflow preview from a model response.

    The model is expected to return either pure markdown (empty
    preview) or markdown plus a fenced ``json`` block carrying
    ``{"workflow_ir", "python_script", "mermaid", "intervention_points"}``.
    A malformed JSON block raises :class:`json.JSONDecodeError`; the
    runner surfaces that as a ``FailureRecorded`` for the turn.
    """

    plan = text.strip()
    if "```json" not in plan:
        return plan, WorkflowPreview(workflow_ir={})
    head, _, rest = plan.partition("```json")
    body, _, tail = rest.partition("```")
    data = json.loads(body)
    preview = WorkflowPreview(workflow_ir={})
    if isinstance(data, dict):
        preview = WorkflowPreview(
            workflow_ir=data.get("workflow_ir", {}) or {},
            python_script=str(data.get("python_script", "")),
            mermaid=str(data.get("mermaid", "")),
            intervention_points=list(data.get("intervention_points", []) or []),
        )
    return (head + tail).strip(), preview


def _goal_in_mode(goal: Goal, mode: AgentMode) -> Goal:
    return dataclasses.replace(goal, mode=mode)
