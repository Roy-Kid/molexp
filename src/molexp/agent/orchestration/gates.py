"""Session-driven approval and chat gateways (spec §6.2, §6.3).

The dispatcher is model-agnostic; it only knows about an
:class:`ApprovalGate` protocol. The runner installs a
:class:`SessionApprovalGate` that bridges the dispatcher to the live
session: emit :class:`ToolApprovalRequested`, park on
:class:`ApprovalRegistry`, and return the resolved decision.

Likewise :class:`SessionChatGateway` bridges ``native:ask_user`` style
tools to :class:`UserMessageRegistry` + :class:`UserMessageRequested`.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

from molexp.agent.model import ModelToolCall
from molexp.agent.orchestration.events import (
    ToolApprovalRequested,
    UserMessageRequested,
)
from molexp.agent.tools.policy import ApprovalDecision
from molexp.agent.tools.spec import ToolContext, ToolSpec
from molexp.agent.types import SessionStatus, utc_now

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from molexp.agent.orchestration.session import AgentSession


class SessionApprovalGate:
    """Bridge :class:`ToolDispatcher` to the live :class:`AgentSession`.

    The dispatcher calls :meth:`request` for every approval-required
    tool; the gate emits :class:`ToolApprovalRequested`, opens a
    pending entry on the session's :class:`ApprovalRegistry`, flips the
    session into :attr:`SessionStatus.AWAITING_APPROVAL`, and awaits
    the resolution future. Server routes resolve the future via
    :meth:`AgentSession.respond_approval`.
    """

    def __init__(self, session: "AgentSession", turn_id: str) -> None:
        self._session = session
        self._turn_id = turn_id

    async def request(
        self,
        call: ModelToolCall,
        spec: ToolSpec,
        ctx: ToolContext,
    ) -> ApprovalDecision:
        request_id = call.id or secrets.token_hex(6)
        pending = self._session.approvals.open(
            request_id=request_id,
            tool_name=spec.name,
            arguments=dict(call.arguments),
        )
        prior_status = self._session.status
        self._session.status = SessionStatus.AWAITING_APPROVAL
        await self._session.bus.publish(
            ToolApprovalRequested(
                turn_id=self._turn_id,
                request_id=pending.request_id,
                tool_name=spec.name,
                arguments=dict(call.arguments),
                ts=utc_now(),
            )
        )
        decision = await pending.future
        # Restore pre-approval status only if it wasn't independently
        # advanced (e.g. cancelled) while parked.
        if self._session.status is SessionStatus.AWAITING_APPROVAL:
            self._session.status = prior_status
        return decision


class SessionChatGateway:
    """Bridge ``ask_user``-style tools to the live session inbox.

    Emits :class:`UserMessageRequested`, opens a pending entry on the
    session's :class:`UserMessageRegistry`, flips status to
    :attr:`SessionStatus.AWAITING_USER`, and awaits the user's reply.
    """

    def __init__(self, session: "AgentSession") -> None:
        self._session = session

    async def ask(self, prompt: str) -> str:
        pending = self._session.user_inbox.open(prompt)
        prior_status = self._session.status
        self._session.status = SessionStatus.AWAITING_USER
        await self._session.bus.publish(
            UserMessageRequested(
                request_id=pending.request_id,
                prompt=prompt,
                ts=utc_now(),
            )
        )
        reply = await pending.future
        if self._session.status is SessionStatus.AWAITING_USER:
            self._session.status = prior_status
        return reply
