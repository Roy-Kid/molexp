"""Phase 1a: ToolDispatcher validation, approval, and failure normalization."""

from __future__ import annotations

import pytest

from molexp.agent import (
    FailureKind,
    ModelToolCall,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from molexp.agent.tools import (
    AutoApproveGate,
    DenyAllGate,
    PERMISSIVE_POLICY,
    ToolDispatcher,
    ToolPolicy,
)


def _ctx() -> ToolContext:
    return ToolContext(workspace=None, session_id="s", turn_id="t1")


def _spec(name: str, **kwargs) -> ToolSpec:
    base = {
        "name": name,
        "description": "",
        "input_schema": {"type": "object", "properties": {}},
    }
    base.update(kwargs)
    return ToolSpec(**base)


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_returns_typed_failure() -> None:
    registry = ToolRegistry()
    dispatcher = ToolDispatcher(registry)
    result = await dispatcher.dispatch(
        ModelToolCall(id="1", name="native:missing", arguments={}),
        _ctx(),
        PERMISSIVE_POLICY,
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is FailureKind.TOOL_NOT_FOUND


@pytest.mark.asyncio
async def test_dispatch_invokes_callable_and_returns_result() -> None:
    registry = ToolRegistry()

    captured: dict = {}

    async def stub(args: dict, ctx: ToolContext) -> ToolResult:
        captured["args"] = args
        captured["session"] = ctx.session_id
        return ToolResult(ok=True, value={"echo": args})

    registry.register(_spec("native:echo"), stub)
    dispatcher = ToolDispatcher(registry)
    result = await dispatcher.dispatch(
        ModelToolCall(id="1", name="native:echo", arguments={"x": 1}),
        _ctx(),
        PERMISSIVE_POLICY,
    )
    assert result.ok is True
    assert result.value == {"echo": {"x": 1}}
    assert captured["args"] == {"x": 1}
    assert captured["session"] == "s"


@pytest.mark.asyncio
async def test_dispatch_normalizes_callable_exceptions() -> None:
    registry = ToolRegistry()

    async def boom(args: dict, ctx: ToolContext) -> ToolResult:
        raise RuntimeError("kaboom")

    registry.register(_spec("native:boom"), boom)
    dispatcher = ToolDispatcher(registry)
    result = await dispatcher.dispatch(
        ModelToolCall(id="1", name="native:boom", arguments={}),
        _ctx(),
        PERMISSIVE_POLICY,
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is FailureKind.TOOL_ERROR
    assert "kaboom" in result.error.message


@pytest.mark.asyncio
async def test_dispatch_blocked_by_policy_returns_tool_not_found() -> None:
    registry = ToolRegistry()

    async def stub(args: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult(ok=True)

    registry.register(_spec("native:write_file", mutates=True), stub)
    dispatcher = ToolDispatcher(registry)
    policy = ToolPolicy(deny=("native:write_*",))
    result = await dispatcher.dispatch(
        ModelToolCall(id="1", name="native:write_file", arguments={}),
        _ctx(),
        policy,
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is FailureKind.TOOL_NOT_FOUND


@pytest.mark.asyncio
async def test_dispatch_denied_approval_returns_typed_failure() -> None:
    registry = ToolRegistry()

    async def stub(args: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult(ok=True)

    registry.register(_spec("native:run", mutates=True), stub)
    dispatcher = ToolDispatcher(registry, gate=DenyAllGate())
    result = await dispatcher.dispatch(
        ModelToolCall(id="1", name="native:run", arguments={}),
        _ctx(),
        PERMISSIVE_POLICY,
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is FailureKind.APPROVAL_DENIED


@pytest.mark.asyncio
async def test_auto_approve_gate_lets_mutating_tools_through() -> None:
    registry = ToolRegistry()

    async def stub(args: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult(ok=True, value="done")

    registry.register(_spec("native:run", mutates=True), stub)
    dispatcher = ToolDispatcher(registry, gate=AutoApproveGate())
    result = await dispatcher.dispatch(
        ModelToolCall(id="1", name="native:run", arguments={}),
        _ctx(),
        PERMISSIVE_POLICY,
    )
    assert result.ok is True
    assert result.value == "done"
