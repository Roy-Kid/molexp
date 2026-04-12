"""PydanticAISession: concrete AgentSession implementation.

Design:
- Agent run executes as an asyncio.Task (background coroutine)
- Events are forwarded through an asyncio.Queue to stream_events()
- Approval requests use asyncio.Future for suspend/resume
- Session state is persisted to workspace after completion
"""

from __future__ import annotations

import asyncio
from mollog import get_logger
from typing import TYPE_CHECKING, Any, AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.messages import AgentStreamEvent

from ..types import (
    AgentSession,
    Goal,
    SessionCompletedEvent,
    SessionEvent,
)
from .events import map_stream_event

if TYPE_CHECKING:
    from .deps import MolexpDeps

logger = get_logger(__name__)

_DONE = object()  # Sentinel to signal stream_events() to stop


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
        self._run_task: asyncio.Task | None = None
        self._message_history: list[Any] = []

    def _launch(self, agent: Agent, prompt: str, deps: MolexpDeps) -> None:
        """Start the agent run as a background asyncio.Task."""
        self._run_task = asyncio.create_task(
            self._run_agent(agent, prompt, deps),
            name=f"molexp-session-{self.session_id}",
        )

    async def _run_agent(self, agent: Agent, prompt: str, deps: MolexpDeps) -> None:
        """Background coroutine: runs the agent and forwards events."""
        try:
            event_queue = self._event_queue

            async def handle_events(
                ctx: Any, events: AsyncIterator[AgentStreamEvent]
            ) -> None:
                async for raw_event in events:
                    molexp_event = map_stream_event(raw_event)
                    if molexp_event is not None:
                        await event_queue.put(molexp_event)

            result = await agent.run(
                prompt,
                deps=deps,
                event_stream_handler=handle_events,
                message_history=self._message_history or None,
            )

            # Persist message history for potential resumption
            self._message_history = result.all_messages()

            self.status = "completed"
            completed_event = SessionCompletedEvent(
                summary=str(result.output),
                produced_runs=list(self.produced_runs),
                artifacts=list(self.artifacts),
            )
            await event_queue.put(completed_event)

        except Exception as exc:
            logger.exception(f"Agent session {self.session_id} failed")
            self.status = "failed"
            await event_queue.put(
                SessionCompletedEvent(summary=f"Session failed: {exc}")
            )
        finally:
            await self._event_queue.put(_DONE)

    async def stream_events(self) -> AsyncIterator[SessionEvent]:
        """Yield session events until the run completes."""
        while True:
            item = await self._event_queue.get()
            if item is _DONE:
                return
            yield item

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
