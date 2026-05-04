"""PydanticAISession: concrete AgentSession implementation.

Design:
- Agent run executes as an asyncio.Task (background coroutine)
- Events are forwarded through an asyncio.Queue to stream_events()
- Approval requests use asyncio.Future for suspend/resume
- Session state is persisted to workspace after completion
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator

from mollog import get_logger
from pydantic_ai import Agent
from pydantic_ai.messages import AgentStreamEvent

from ..types import (
    AgentSession,
    Goal,
    PlanCreatedEvent,
    ResultArtifactEvent,
    SessionCompletedEvent,
    SessionEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
    UserMessageRequestEvent,
    WorkflowPreview,
)
from molexp.workflow.compiler import default_compiler

from .events import map_stream_event

if TYPE_CHECKING:
    from .deps import MolexpDeps

logger = get_logger(__name__)

_DONE = object()  # Sentinel to signal stream_events() to stop

_ARTIFACT_KINDS = {"plot", "table", "text"}


def _maybe_artifact(result: Any) -> ResultArtifactEvent | None:
    """Detect the artifact convention in a tool result.

    Tools (including agent-generated code-execution returns) may
    produce a dict shaped ``{"kind": "plot"|"table"|"text", ...}``.
    When detected, the runtime emits a :class:`ResultArtifactEvent`
    so the UI can render the artifact inline.
    """
    if not isinstance(result, dict):
        return None
    kind = result.get("kind")
    if kind not in _ARTIFACT_KINDS:
        return None
    payload = {k: v for k, v in result.items() if k not in {"kind", "title"}}
    return ResultArtifactEvent(
        kind=kind,
        title=str(result.get("title", "")),
        payload=payload,
    )


class PydanticAISession(AgentSession):
    """AgentSession backed by a pydantic-ai Agent run.

    The agent run is launched as an asyncio.Task immediately after
    construction. Events flow from the agent run → asyncio.Queue →
    stream_events() async iterator.

    Approval flow:
        1. ApprovalRequiredToolset raises ApprovalRequired
        2. The agent run gets DeferredToolRequests
        3. We emit ApprovalRequestEvent to the queue
        4. respond_approval() resolves the pending Future
        5. The agent run resumes with approved=True/False
    """

    def __init__(
        self,
        session_id: str,
        goal: Goal,
        workspace: Any,
    ) -> None:
        super().__init__(session_id=session_id, goal=goal)
        self._workspace = workspace
        self._event_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._approval_futures: dict[str, asyncio.Future[bool]] = {}
        self._user_message_futures: dict[str, asyncio.Future[str]] = {}
        self._plan_decision_futures: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._run_task: asyncio.Task | None = None
        self._message_history: list[Any] = []
        self._agent: Agent[Any, str] | None = None
        self._deps: Any = None
        self._followup_lock = asyncio.Lock()
        # Hook the runtime sets so it can rebuild the agent (and thus its
        # toolset + system prompt) when plan mode is exited via a structured
        # decision. Receives the freshly-mutated :class:`Goal` and returns
        # a new pydantic-ai Agent.
        self._rebuild_agent: Any = None
        # Optional hook fired after status transitions to a terminal state
        # ("completed" / "failed"). The runtime sets this so it can re-flush
        # metadata to disk; tests may leave it None.
        self._on_terminal: Any = None
        # The composed system prompt active for this session — set by the
        # runtime right before launching the agent so the inspector route
        # can show what the agent actually saw without re-running the
        # composition (which could drift from live edits to provider config).
        self._system_prompt: str = ""

    @property
    def system_prompt(self) -> str:
        """Return the composed system prompt the agent was started with."""
        return self._system_prompt

    def set_system_prompt(self, prompt: str) -> None:
        """Record the prompt that was passed to the underlying Agent."""
        self._system_prompt = prompt

    def _launch(self, agent: Agent[MolexpDeps, str], prompt: str, deps: MolexpDeps) -> None:
        """Start the agent run as a background asyncio.Task."""
        self.stats.started_at = datetime.now(timezone.utc)
        self._agent = agent
        self._deps = deps
        self._run_task = asyncio.create_task(
            self._run_agent(agent, prompt, deps),
            name=f"molexp-session-{self.session_id}",
        )

    async def _run_agent(
        self, agent: Agent[MolexpDeps, str], prompt: str, deps: MolexpDeps
    ) -> None:
        """Background coroutine: runs the agent and forwards events."""
        try:
            event_queue = self._event_queue
            stats = self.stats

            async def handle_events(ctx: Any, events: AsyncIterable[AgentStreamEvent]) -> None:
                async for raw_event in events:
                    molexp_event = map_stream_event(raw_event)
                    if molexp_event is None:
                        continue
                    stats.events += 1
                    if isinstance(molexp_event, ToolCallEvent):
                        stats.tool_calls += 1
                    await event_queue.put(molexp_event)
                    # Convention: tools that return a dict shaped like
                    # {"kind": "plot"|"table"|"text", ...} produce an
                    # additional inline artifact event for the UI.
                    if isinstance(molexp_event, ToolResultEvent):
                        artifact = _maybe_artifact(molexp_event.result)
                        if artifact is not None:
                            stats.events += 1
                            await event_queue.put(artifact)
                            self.artifacts.append(
                                {
                                    "kind": artifact.kind,
                                    "title": artifact.title,
                                    "payload": artifact.payload,
                                }
                            )

            result = await agent.run(
                prompt,
                deps=deps,
                event_stream_handler=handle_events,
                message_history=self._message_history or None,
            )

            # Persist message history for potential resumption
            self._message_history = result.all_messages()
            self._absorb_usage(result)

            self.status = "completed"
            stats.completed_at = datetime.now(timezone.utc)
            completed_event = SessionCompletedEvent(
                summary=str(result.output),
                produced_runs=list(self.produced_runs),
                artifacts=list(self.artifacts),
            )
            await event_queue.put(completed_event)

        except Exception as exc:
            logger.exception(f"Agent session {self.session_id} failed")
            self.status = "failed"
            self.stats.completed_at = datetime.now(timezone.utc)
            await event_queue.put(SessionCompletedEvent(summary=f"Session failed: {exc}"))
        finally:
            self._fire_terminal_hook()
            await self._event_queue.put(_DONE)

    def _fire_terminal_hook(self) -> None:
        """Best-effort callback after the run loop exits — never raises."""
        hook = self._on_terminal
        if hook is None:
            return
        try:
            hook(self)
        except Exception:
            logger.exception(f"Terminal hook failed for session {self.session_id}")

    def _absorb_usage(self, result: Any) -> None:
        """Merge a pydantic-ai run result's usage into the session stats."""
        getter = getattr(result, "usage", None)
        if getter is None:
            return
        try:
            usage = getter() if callable(getter) else getter
        except Exception:
            logger.debug("Failed to read usage from agent result", exc_info=True)
            return
        if usage is None:
            return
        stats = self.stats
        stats.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        stats.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        stats.cache_read_tokens += int(getattr(usage, "cache_read_tokens", 0) or 0)
        stats.cache_write_tokens += int(getattr(usage, "cache_write_tokens", 0) or 0)
        stats.requests += int(getattr(usage, "requests", 0) or 0)
        # Prefer pydantic-ai's tool_calls count (authoritative) when present.
        usage_tool_calls = int(getattr(usage, "tool_calls", 0) or 0)
        if usage_tool_calls:
            stats.tool_calls = max(stats.tool_calls, usage_tool_calls)
        stats.total_tokens = stats.input_tokens + stats.output_tokens

    async def stream_events(self) -> AsyncIterator[SessionEvent]:
        """Yield session events until the (current) run completes."""
        while True:
            item = await self._event_queue.get()
            if item is _DONE:
                return
            yield item

    def drain_pending_events(self) -> list[SessionEvent]:
        """Non-blocking drain of buffered events.

        Returns every event currently sitting in the queue. ``_DONE``
        sentinels are filtered out so chat follow-ups remain consumable
        across multiple agent runs.
        """
        out: list[SessionEvent] = []
        while True:
            try:
                item = self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                return out
            if item is _DONE:
                continue
            out.append(item)

    async def respond_approval(self, request_id: str, approved: bool) -> None:
        """Resolve a pending approval request.

        Args:
            request_id: ID from the ApprovalRequestEvent
            approved: True to allow, False to deny
        """
        future = self._approval_futures.get(request_id)
        if future is not None and not future.done():
            future.set_result(approved)
        else:
            logger.warning(f"No pending approval request for id={request_id}")

    def get_message_history(self) -> list[Any]:
        """Return the accumulated message history for persistence/resumption."""
        return list(self._message_history)

    def restore_message_history(self, history: list[Any]) -> None:
        """Restore message history for a resumed session."""
        self._message_history = list(history)

    async def await_user_message(self, prompt: str) -> str:
        """Pause the run and wait for a user reply.

        Emits a :class:`UserMessageRequestEvent` and parks on a future
        until :meth:`respond_user_message` resolves it.
        """
        request_id = uuid.uuid4().hex
        future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        self._user_message_futures[request_id] = future
        await self._event_queue.put(
            UserMessageRequestEvent(request_id=request_id, prompt=prompt)
        )
        try:
            return await future
        finally:
            self._user_message_futures.pop(request_id, None)

    async def respond_user_message(
        self, content: str, request_id: str | None = None
    ) -> None:
        """Deliver a user reply to the session.

        If ``request_id`` matches a pending :meth:`await_user_message`,
        resolves the future. Otherwise treats the message as an
        unsolicited follow-up and re-runs the agent with the message
        appended to history (keeps the session alive for chat).
        """
        await self._event_queue.put(
            UserMessageEvent(content=content, request_id=request_id)
        )
        if request_id is not None:
            future = self._user_message_futures.get(request_id)
            if future is not None and not future.done():
                future.set_result(content)
                return
            logger.warning(
                f"No pending user-message request for id={request_id}; "
                "treating as unsolicited follow-up."
            )
        await self._kick_followup(content)

    async def _kick_followup(self, content: str) -> None:
        """Re-run the agent with a follow-up user prompt.

        Serialized via ``_followup_lock`` so concurrent messages queue
        rather than racing the agent run.
        """
        async with self._followup_lock:
            if self._agent is None or self._deps is None:
                logger.warning(
                    "Cannot dispatch follow-up: session has no agent/deps yet."
                )
                return
            # Wait for any in-flight run to finish so message history is consistent.
            if self._run_task is not None and not self._run_task.done():
                try:
                    await self._run_task
                except Exception:
                    pass
            self.status = "running"
            self.stats.completed_at = None
            self._run_task = asyncio.create_task(
                self._run_agent(self._agent, content, self._deps),
                name=f"molexp-session-{self.session_id}-followup",
            )

    # ── Plan-mode handoff ────────────────────────────────────────────────

    async def await_plan_decision(
        self,
        plan_markdown: str,
        workflow_preview: dict[str, Any],
    ) -> dict[str, Any]:
        """Pause the run on a finalized plan and wait for user decision.

        Every plan is a workflow: the prose ``plan_markdown`` and the
        structured ``workflow_preview.workflow_ir`` are two views of the
        same set of nodes. Emits a :class:`PlanCreatedEvent` to the
        stream and parks on a future. Resolved by :meth:`respond_plan`
        with the user's approve / reject + any edits. The dict returned
        to the agent is what the agent reads as the tool result of
        ``exit_plan_mode``.

        Re-validates the workflow_preview shape so the session can never
        park on a malformed plan even if the tool wrapper is bypassed.
        """
        if not isinstance(workflow_preview, dict):
            raise ValueError(
                "await_plan_decision: workflow_preview must be a dict with a "
                "non-empty workflow_ir.task_configs list."
            )
        ir = workflow_preview.get("workflow_ir")
        if not isinstance(ir, dict) or not ir.get("task_configs"):
            raise ValueError(
                "await_plan_decision: workflow_preview.workflow_ir.task_configs "
                "must contain at least one node."
            )

        request_id = uuid.uuid4().hex
        future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_event_loop().create_future()
        )
        self._plan_decision_futures[request_id] = future

        # Auto-render the matching Python script if the agent did not
        # supply one. The IR is the source of truth; the script is the
        # human-readable view derived from it via the bidirectional
        # codegen module. When the agent does provide a script, preserve
        # its text verbatim (the agent may have authored
        # intentionally-readable comments / formatting we do not want
        # to normalize away).
        ir_dict = dict(ir)
        agent_script_raw = str(workflow_preview.get("python_script", ""))
        if agent_script_raw.strip():
            python_script = agent_script_raw
        else:
            try:
                python_script = default_compiler.ir_to_python(ir_dict)
            except ValueError:
                logger.exception(
                    "Failed to render python_script for workflow IR; "
                    "falling back to empty string."
                )
                python_script = ""
        preview_obj = WorkflowPreview(
            workflow_ir=ir_dict,
            python_script=python_script,
            mermaid=str(workflow_preview.get("mermaid", "")),
            intervention_points=list(workflow_preview.get("intervention_points") or []),
        )

        await self._event_queue.put(
            PlanCreatedEvent(
                request_id=request_id,
                plan_markdown=plan_markdown,
                workflow_preview=preview_obj,
            )
        )
        try:
            return await future
        finally:
            self._plan_decision_futures.pop(request_id, None)

    async def respond_plan(
        self,
        request_id: str,
        approved: bool,
        edited_plan: str | None = None,
        edited_workflow_ir: dict[str, Any] | None = None,
        feedback: str = "",
    ) -> None:
        """Resolve a pending plan decision from the chat client.

        On approval the goal flips out of plan mode and the agent is
        rebuilt without the plan-mode system prompt; the agent's
        ``exit_plan_mode`` tool returns the user's decision dict so it
        can proceed with binding and executing the workflow. On
        rejection the agent receives the feedback and can revise +
        call ``exit_plan_mode`` again in the same session.
        """
        future = self._plan_decision_futures.get(request_id)
        if future is None:
            logger.warning(f"No pending plan decision for id={request_id}")
            return
        if future.done():
            logger.warning(f"Plan decision for id={request_id} already resolved")
            return
        if approved:
            self.goal.plan_mode = False
            if self._rebuild_agent is not None:
                try:
                    self._agent = self._rebuild_agent(self.goal)
                except Exception:
                    logger.exception(
                        "Failed to rebuild agent after plan approval"
                    )
            future.set_result(
                {
                    "approved": True,
                    "edited_plan": edited_plan,
                    "edited_workflow_ir": edited_workflow_ir,
                }
            )
        else:
            future.set_result(
                {
                    "approved": False,
                    "feedback": feedback or "User rejected the plan.",
                }
            )
