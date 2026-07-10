"""Runtime MCP catalog listing (no hard-coded tool name tables)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from molexp.agent.execution_env import LocalExecutionEnv
from molexp.agent.loops.interactive.mcp_toolsets import list_mcp_tool_specs
from molexp.agent.ops import build_session_context
from molexp.agent.ops.protocols import ToolSpec


class _FakeMCPToolset:
    def __init__(self, tools: list[SimpleNamespace]) -> None:
        self._tools = tools
        self.list_calls = 0

    async def list_tools(self) -> list[SimpleNamespace]:
        self.list_calls += 1
        return list(self._tools)


class _FakePrefixed:
    def __init__(self, wrapped: _FakeMCPToolset, prefix: str) -> None:
        self.wrapped = wrapped
        self.prefix = prefix


@pytest.mark.asyncio
async def test_list_mcp_tool_specs_prefixes_names() -> None:
    inner = _FakeMCPToolset(
        [
            SimpleNamespace(name="add_project", description="create project"),
            SimpleNamespace(name="search", description="search symbols"),
        ]
    )
    prefixed = _FakePrefixed(inner, "molmcp")
    specs = await list_mcp_tool_specs((prefixed,))
    names = {s.name for s in specs}
    assert names == {"molmcp_add_project", "molmcp_search"}
    assert inner.list_calls == 1
    by_name = {s.name: s for s in specs}
    assert by_name["molmcp_add_project"].description == "create project"
    assert by_name["molmcp_add_project"].source == "molmcp"


@pytest.mark.asyncio
async def test_list_mcp_tool_specs_swallows_list_failure() -> None:
    class _Boom:
        prefix = "x"

        @property
        def wrapped(self) -> _Boom:
            return self

        async def list_tools(self) -> list[object]:
            raise RuntimeError("server down")

    specs = await list_mcp_tool_specs((_Boom(),))
    assert specs == ()


def test_session_context_surfaces_mcp_specs_in_discover(tmp_path) -> None:
    specs = (
        ToolSpec(name="molmcp_foo", description="foo tool", source="molmcp"),
        ToolSpec(name="molmcp_bar", description="bar tool", source="molmcp"),
    )
    ctx = build_session_context(
        workspace_root=tmp_path,
        execution_env=LocalExecutionEnv(scratch_dir=tmp_path / "s"),
        mcp_tool_specs=specs,
    )
    hits = ctx.discovery.search("foo", kind="mcp_tool")
    assert any(h.ref == "molmcp_foo" for h in hits)
    assert "foo tool" in ctx.discovery.describe("molmcp_foo")
