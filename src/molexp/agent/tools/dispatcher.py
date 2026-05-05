"""ToolDispatcher: validation + policy + execution.

The dispatcher is the only code path that turns a ``ModelToolCall``
into a ``ToolResult``. It looks up the tool — first against the
service-owned native :class:`ToolRegistry`, then against any
registered :class:`ToolSource` — applies the policy filter, awaits
human approval when required, calls the tool, and normalizes
exceptions into typed :class:`AgentFailure` results.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Protocol, Sequence, runtime_checkable

from molexp.agent.model import ModelToolCall, ToolSchema
from molexp.agent.tools.policy import ApprovalDecision, ToolPolicy
from molexp.agent.tools.registry import ToolRegistry
from molexp.agent.tools.source import ToolSource
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
        sources: Sequence[ToolSource] = (),
    ) -> None:
        self._registry = registry
        self._gate = gate or AutoApproveGate()
        self._on_event = on_event
        self._sources: tuple[ToolSource, ...] = tuple(sources)
        self._source_index: dict[str, tuple[ToolSource, ToolSpec]] = {}

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def sources(self) -> tuple[ToolSource, ...]:
        return self._sources

    async def discover(self, workspace: object, policy: ToolPolicy) -> tuple[ToolSchema, ...]:
        """Return every tool schema visible under ``policy``.

        Native tools come from the registry; source-contributed tools
        are listed via :meth:`ToolSource.list_tools`. The discovered
        source specs are cached on the dispatcher so a subsequent
        :meth:`dispatch` can resolve the call without re-listing.
        Native names take precedence on collision.
        """

        schemas: list[ToolSchema] = list(self._registry.schemas(policy))
        seen: set[str] = {schema.name for schema in schemas}
        index: dict[str, tuple[ToolSource, ToolSpec]] = {}
        for source in self._sources:
            for spec in await source.list_tools(workspace):
                if not policy.visible(spec):
                    continue
                if spec.name in seen:
                    continue
                seen.add(spec.name)
                index[spec.name] = (source, spec)
                schemas.append(
                    ToolSchema(
                        name=spec.name,
                        description=spec.description,
                        input_schema=spec.input_schema,
                    )
                )
        self._source_index = index
        return tuple(schemas)

    async def dispatch(
        self,
        call: ModelToolCall,
        ctx: ToolContext,
        policy: ToolPolicy,
        gate: ApprovalGate | None = None,
    ) -> ToolResult:
        registered = self._registry.get(call.name)
        if registered is not None:
            if not policy.visible(registered.spec):
                return _not_found(call.name)
            return await self._dispatch_native(registered, call, ctx, policy, gate)

        bound = self._source_index.get(call.name)
        if bound is None:
            return _not_found(call.name)
        source, spec = bound
        if not policy.visible(spec):
            return _not_found(call.name)
        return await self._dispatch_source(source, spec, call, ctx, policy, gate)

    async def _dispatch_native(
        self,
        registered,
        call: ModelToolCall,
        ctx: ToolContext,
        policy: ToolPolicy,
        gate: ApprovalGate | None,
    ) -> ToolResult:
        spec = registered.spec
        denial = await self._approve_or_deny(call, spec, ctx, policy, gate)
        if denial is not None:
            return denial

        await self._emit("tool.call", {"tool": call.name, "id": call.id})
        try:
            result = await registered.fn(call.arguments, ctx)
        except Exception as exc:  # noqa: BLE001 — normalize all callable errors
            return _tool_error(call.name, exc)
        await self._emit("tool.result", {"tool": call.name, "ok": result.ok})
        return result

    async def _dispatch_source(
        self,
        source: ToolSource,
        spec: ToolSpec,
        call: ModelToolCall,
        ctx: ToolContext,
        policy: ToolPolicy,
        gate: ApprovalGate | None,
    ) -> ToolResult:
        denial = await self._approve_or_deny(call, spec, ctx, policy, gate)
        if denial is not None:
            return denial

        await self._emit(
            "tool.call",
            {"tool": call.name, "id": call.id, "source": source.source_name},
        )
        try:
            result = await source.call(call.name, dict(call.arguments), ctx)
        except Exception as exc:  # noqa: BLE001 — normalize all callable errors
            return _tool_error(call.name, exc, source=source.source_name)
        await self._emit(
            "tool.result",
            {"tool": call.name, "ok": result.ok, "source": source.source_name},
        )
        return result

    async def _approve_or_deny(
        self,
        call: ModelToolCall,
        spec: ToolSpec,
        ctx: ToolContext,
        policy: ToolPolicy,
        gate: ApprovalGate | None,
    ) -> ToolResult | None:
        if not policy.needs_approval(spec):
            return None
        active_gate = gate or self._gate
        decision = await active_gate.request(call, spec, ctx)
        await self._emit(
            "tool.approval",
            {
                "tool": call.name,
                "approved": decision.approved,
                "request_id": decision.request_id,
            },
        )
        if decision.approved:
            return None
        return ToolResult(
            ok=False,
            error=AgentFailure(
                kind=FailureKind.APPROVAL_DENIED,
                message=decision.reason or "Approval denied",
            ),
        )

    async def _emit(self, kind: str, payload: dict[str, object]) -> None:
        if self._on_event is None:
            return
        await self._on_event(kind, payload)


def _not_found(name: str) -> ToolResult:
    return ToolResult(
        ok=False,
        error=AgentFailure(
            kind=FailureKind.TOOL_NOT_FOUND,
            message=f"Tool '{name}' is not visible under the active policy",
        ),
    )


def _tool_error(name: str, exc: BaseException, **detail: object) -> ToolResult:
    payload: dict[str, object] = {"tool": name}
    payload.update(detail)
    return ToolResult(
        ok=False,
        error=AgentFailure(
            kind=FailureKind.TOOL_ERROR,
            message=f"{type(exc).__name__}: {exc}",
            detail=payload,
        ),
    )
