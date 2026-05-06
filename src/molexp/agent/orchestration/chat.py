"""User-message wait/resume primitives.

The harness exposes an ``ask_user`` tool to model plugins. When a tool
calls the chat gateway, the runner emits :class:`UserMessageRequested`
and parks until the session receives an inbound user message tagged
with the matching ``request_id``.

Mirrors :mod:`approvals` but for free-form user replies rather than
yes/no decisions.
"""

from __future__ import annotations

import asyncio
import secrets
from typing import Protocol, runtime_checkable


class PendingUserMessage:
    """Outstanding ``ask_user`` prompt the session is parked on.

    Plain class because it carries a live ``asyncio.Future`` runtime ref.
    """

    __slots__ = ("request_id", "prompt", "future")

    def __init__(
        self,
        request_id: str,
        prompt: str,
        future: "asyncio.Future[str]",
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.future = future


class UserMessageRegistry:
    """Per-session map of ``request_id`` -> outstanding chat request."""

    def __init__(self) -> None:
        self._pending: dict[str, PendingUserMessage] = {}

    def open(self, prompt: str, request_id: str | None = None) -> PendingUserMessage:
        loop = asyncio.get_running_loop()
        rid = request_id or secrets.token_hex(6)
        record = PendingUserMessage(
            request_id=rid,
            prompt=prompt,
            future=loop.create_future(),
        )
        self._pending[rid] = record
        return record

    def resolve(self, request_id: str, content: str) -> bool:
        record = self._pending.pop(request_id, None)
        if record is None or record.future.done():
            return False
        record.future.set_result(content)
        return True

    def has(self, request_id: str) -> bool:
        return request_id in self._pending

    def list(self) -> tuple[PendingUserMessage, ...]:
        return tuple(self._pending.values())

    def pop_oldest(self) -> PendingUserMessage | None:
        """Resolve the oldest outstanding request when an unsolicited
        user message arrives. Returns the popped record, or None.
        """

        if not self._pending:
            return None
        first_id = next(iter(self._pending))
        return self._pending.pop(first_id)

    def cancel_all(self) -> None:
        """Drop every outstanding request and cancel its future.

        Called when the session is cancelled, so awaiting tools wake up
        instead of leaking the pending dict.
        """

        for record in self._pending.values():
            if not record.future.done():
                record.future.cancel()
        self._pending.clear()


@runtime_checkable
class ChatGateway(Protocol):
    """Tool-facing handle for prompting the user during a turn."""

    async def ask(self, prompt: str) -> str: ...
