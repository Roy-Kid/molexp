"""Approval wait/resume primitives.

The dispatcher owns the gate; the session-side handle lives here so
route layers can resolve a pending approval by id.
"""

from __future__ import annotations

import asyncio

from molexp.agent.tools.policy import ApprovalDecision


class PendingApproval:
    """Outstanding approval request the session is parked on.

    Plain class because it carries a live ``asyncio.Future`` runtime ref.
    """

    __slots__ = ("request_id", "tool_name", "arguments", "future")

    def __init__(
        self,
        request_id: str,
        tool_name: str,
        arguments: dict,
        future: "asyncio.Future[ApprovalDecision]",
    ) -> None:
        self.request_id = request_id
        self.tool_name = tool_name
        self.arguments = arguments
        self.future = future


class ApprovalRegistry:
    """Per-session map of ``request_id`` -> outstanding approval."""

    def __init__(self) -> None:
        self._pending: dict[str, PendingApproval] = {}

    def open(
        self,
        request_id: str,
        tool_name: str,
        arguments: dict,
    ) -> PendingApproval:
        loop = asyncio.get_running_loop()
        record = PendingApproval(
            request_id=request_id,
            tool_name=tool_name,
            arguments=arguments,
            future=loop.create_future(),
        )
        self._pending[request_id] = record
        return record

    def resolve(self, decision: ApprovalDecision) -> bool:
        record = self._pending.pop(decision.request_id, None)
        if record is None or record.future.done():
            return False
        record.future.set_result(decision)
        return True

    def has(self, request_id: str) -> bool:
        return request_id in self._pending

    def list(self) -> tuple[PendingApproval, ...]:
        return tuple(self._pending.values())
