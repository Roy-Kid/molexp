"""ToolDispatcher: validation + policy + execution.

The dispatcher is the only code path that turns a ``ModelToolCall``
into a ``ToolResult``. It looks up the tool, applies the policy
filter, awaits human approval when required, calls the tool, and
normalizes exceptions into typed :class:`AgentFailure` results.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Protocol, runtime_checkable

from molexp.agent.model import ModelToolCall
from molexp.agent.tools.policy import ApprovalDecision, ToolPolicy
from molexp.agent.tools.registry import ToolRegistry
from molexp.agent.tools.spec import ToolContext, ToolResult, ToolSpec
from molexp.agent.types import AgentFailure, FailureKind


@runtime_checkable
class ApprovalGate(Protocol):
    """Yields approval decisions for tool calls flagged by policy."""

    async def request(
        self,
        call: ModelToolCall,
        spec: ToolSpec,
        ctx: ToolContext,
    ) -> ApprovalDecision: ...


class AutoApproveGate:
    """Default gate: every request is approved without human input.

    Used in tests; production deployments must override.
    """

    async def request(
        self,
        call: ModelToolCall,
        spec: ToolSpec,
        ctx: ToolContext,
    ) -> ApprovalDecision:
        return ApprovalDecision(request_id=call.id, approved=True)


class DenyAllGate:
    """Test gate: every approval-required request is denied."""

    async def request(
        self,
        call: ModelToolCall,
        spec: ToolSpec,
        ctx: ToolContext,
    ) -> ApprovalDecision:
        return ApprovalDecision(
            request_id=call.id,
            approved=False,
            reason="DenyAllGate: approvals disabled in this context",
        )


ToolEventCallback = Callable[[str, dict[str, object]], Awaitable[None]]


class ToolDispatcher:
    """Translate model tool calls into normalized results."""

    def __init__(
        self,
        registry: ToolRegistry,
        gate: ApprovalGate | None = None,
        on_event: ToolEventCallback | None = None,
    ) -> None:
        self._registry = registry
        self._gate = gate or AutoApproveGate()
        self._on_event = on_event

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    async def dispatch(
        self,
        call: ModelToolCall,
        ctx: ToolContext,
        policy: ToolPolicy,
        gate: ApprovalGate | None = None,
    ) -> ToolResult:
        registered = self._registry.get(call.name)
        if registered is None or not policy.visible(registered.spec):
            return ToolResult(
                ok=False,
                error=AgentFailure(
                    kind=FailureKind.TOOL_NOT_FOUND,
                    message=f"Tool '{call.name}' is not visible under the active policy",
                ),
            )

        active_gate = gate or self._gate
        spec = registered.spec
        if policy.needs_approval(spec):
            decision = await active_gate.request(call, spec, ctx)
            await self._emit("tool.approval", {
                "tool": call.name,
                "approved": decision.approved,
                "request_id": decision.request_id,
            })
            if not decision.approved:
                return ToolResult(
                    ok=False,
                    error=AgentFailure(
                        kind=FailureKind.APPROVAL_DENIED,
                        message=decision.reason or "Approval denied",
                    ),
                )

        await self._emit("tool.call", {"tool": call.name, "id": call.id})
        try:
            result = await registered.fn(call.arguments, ctx)
        except Exception as exc:  # noqa: BLE001 — normalize all callable errors
            return ToolResult(
                ok=False,
                error=AgentFailure(
                    kind=FailureKind.TOOL_ERROR,
                    message=f"{type(exc).__name__}: {exc}",
                    detail={"tool": call.name},
                ),
            )

        await self._emit("tool.result", {"tool": call.name, "ok": result.ok})
        return result

    async def _emit(self, kind: str, payload: dict[str, object]) -> None:
        if self._on_event is None:
            return
        await self._on_event(kind, payload)
