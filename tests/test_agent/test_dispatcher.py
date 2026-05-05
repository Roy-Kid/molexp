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
    PERMISSIVE_POLICY,
    AutoApproveGate,
    DenyAllGate,
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


class _StubToolSource:
    """Minimal :class:`ToolSource` stand-in for dispatcher tests."""

    source_name = "stub"

    def __init__(self, specs: list[ToolSpec]) -> None:
        self.specs = specs
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self, workspace):  # noqa: ANN001 — workspace is opaque
        return list(self.specs)

    async def call(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        self.calls.append((name, args))
        return ToolResult(ok=True, value={"name": name, "args": args})


@pytest.mark.asyncio
async def test_discover_merges_native_and_source_schemas() -> None:
    registry = ToolRegistry()
    registry.register(
        _spec("native:list"),
        lambda a, c: ToolResult(ok=True),  # type: ignore[arg-type]
    )
    source = _StubToolSource([_spec("mcp:srv.echo"), _spec("mcp:srv.ping")])
    dispatcher = ToolDispatcher(registry, sources=(source,))

    schemas = await dispatcher.discover(workspace=None, policy=PERMISSIVE_POLICY)
    names = {s.name for s in schemas}

    assert names == {"native:list", "mcp:srv.echo", "mcp:srv.ping"}


@pytest.mark.asyncio
async def test_dispatch_routes_source_calls_to_source_dot_call() -> None:
    registry = ToolRegistry()
    source = _StubToolSource([_spec("mcp:srv.echo")])
    dispatcher = ToolDispatcher(registry, sources=(source,))
    await dispatcher.discover(workspace=None, policy=PERMISSIVE_POLICY)

    result = await dispatcher.dispatch(
        ModelToolCall(id="1", name="mcp:srv.echo", arguments={"x": 1}),
        _ctx(),
        PERMISSIVE_POLICY,
    )

    assert result.ok is True
    assert result.value == {"name": "mcp:srv.echo", "args": {"x": 1}}
    assert source.calls == [("mcp:srv.echo", {"x": 1})]


@pytest.mark.asyncio
async def test_dispatch_returns_not_found_when_source_unknown_until_discover() -> None:
    """Source tools must be discovered before they become callable."""

    registry = ToolRegistry()
    source = _StubToolSource([_spec("mcp:srv.echo")])
    dispatcher = ToolDispatcher(registry, sources=(source,))

    result = await dispatcher.dispatch(
        ModelToolCall(id="1", name="mcp:srv.echo", arguments={}),
        _ctx(),
        PERMISSIVE_POLICY,
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is FailureKind.TOOL_NOT_FOUND


@pytest.mark.asyncio
async def test_native_name_shadows_source_with_same_name() -> None:
    """When both register the same name, native wins."""

    registry = ToolRegistry()

    async def native_stub(args: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult(ok=True, value="native")

    registry.register(_spec("collide"), native_stub)
    source = _StubToolSource([_spec("collide")])
    dispatcher = ToolDispatcher(registry, sources=(source,))

    schemas = await dispatcher.discover(workspace=None, policy=PERMISSIVE_POLICY)
    assert sum(1 for s in schemas if s.name == "collide") == 1

    result = await dispatcher.dispatch(
        ModelToolCall(id="1", name="collide", arguments={}),
        _ctx(),
        PERMISSIVE_POLICY,
    )
    assert result.value == "native"
    assert source.calls == []


@pytest.mark.asyncio
async def test_source_call_normalizes_exceptions_into_tool_error() -> None:
    registry = ToolRegistry()

    class _Boom:
        source_name = "boom"

        async def list_tools(self, workspace):  # noqa: ANN001
            return [_spec("boom:fail")]

        async def call(self, name, args, ctx):  # noqa: ANN001
            raise RuntimeError("kaboom")

    dispatcher = ToolDispatcher(registry, sources=(_Boom(),))
    await dispatcher.discover(workspace=None, policy=PERMISSIVE_POLICY)

    result = await dispatcher.dispatch(
        ModelToolCall(id="1", name="boom:fail", arguments={}),
        _ctx(),
        PERMISSIVE_POLICY,
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind is FailureKind.TOOL_ERROR
    assert "kaboom" in result.error.message
    assert result.error.detail.get("source") == "boom"
