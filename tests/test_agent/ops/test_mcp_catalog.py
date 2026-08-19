"""Runtime MCP catalog listing (auto-discovery law: no hard-coded tool tables).

Covers ``molexp.agent.mcp.catalog.list_mcp_tool_specs`` and
the ``CatalogDiscovery`` surfacing of runtime-discovered tool specs through
``build_session_context``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from molexp.agent.execution_env import LocalExecutionEnv
from molexp.agent.mcp.catalog import (
    McpCatalog,
    filter_toolsets,
    list_mcp_tool_specs,
)
from molexp.agent.ops import build_session_context
from molexp.agent.ops.builtins import declared_requirements
from molexp.agent.ops.protocols import ToolSpec
from molexp.agent.ops.surface import CHAT_SURFACE, required_keys


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


class TestListMcpToolSpecs:
    @pytest.mark.asyncio
    async def test_prefixes_names_and_captures_description_and_source(self) -> None:
        inner = _FakeMCPToolset(
            [
                SimpleNamespace(name="add_project", description="create project"),
                SimpleNamespace(name="search", description="search symbols"),
            ]
        )
        prefixed = _FakePrefixed(inner, "molmcp")
        specs = await list_mcp_tool_specs((prefixed,))
        by_name = {s.name: s for s in specs}
        assert set(by_name) == {"molmcp_add_project", "molmcp_search"}
        assert inner.list_calls == 1
        assert by_name["molmcp_add_project"].description == "create project"
        assert by_name["molmcp_add_project"].source == "molmcp"

    @pytest.mark.asyncio
    async def test_swallows_list_failure_instead_of_aborting_turn(self) -> None:
        """A toolset whose list_tools raises is skipped, never aborts the turn."""

        class _Boom:
            prefix = "x"

            @property
            def wrapped(self) -> _Boom:
                return self

            async def list_tools(self) -> list[object]:
                raise RuntimeError("server down")

        specs = await list_mcp_tool_specs((_Boom(),))
        assert specs == ()


class _EnteredToolset:
    """Records enter/exit so the catalog's turn-scoped dispose is observable."""

    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self._log = log

    async def __aenter__(self) -> _EnteredToolset:
        self._log.append(f"enter:{self.name}")
        return self

    async def __aexit__(self, *args: object) -> None:
        del args
        self._log.append(f"exit:{self.name}")


class TestMcpCatalogLifecycle:
    @pytest.mark.asyncio
    async def test_open_enters_and_aclose_exits_lifo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log: list[str] = []

        def _fake_build(**kwargs: object) -> _EnteredToolset:
            return _EnteredToolset(str(kwargs.get("name", "")), log)

        monkeypatch.setattr("molexp.agent._pydanticai.mcp.build_mcp_server", _fake_build)
        monkeypatch.setattr("molexp.agent.mcp.store.USER_DIR", tmp_path / "user_home")
        monkeypatch.setattr("molexp.agent.mcp.defaults.seed_user_defaults", lambda *_a, **_k: False)
        (tmp_path / "mcp.json").write_text(
            '{"mcpServers": {'
            '"a": {"type": "stdio", "command": "true"},'
            '"b": {"type": "stdio", "command": "true"}'
            "}}",
            encoding="utf-8",
        )
        catalog = McpCatalog(tmp_path)
        await catalog.open()
        assert {ts.name for ts in catalog.toolsets} == {"a", "b"}  # type: ignore[attr-defined]
        assert log == ["enter:a", "enter:b"]
        await catalog.aclose()
        assert catalog.toolsets == ()
        assert log == ["enter:a", "enter:b", "exit:b", "exit:a"]

    @pytest.mark.asyncio
    async def test_failed_enter_is_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Boom:
            async def __aenter__(self) -> _Boom:
                raise RuntimeError("handshake failed")

        monkeypatch.setattr("molexp.agent._pydanticai.mcp.build_mcp_server", lambda **_k: _Boom())
        monkeypatch.setattr("molexp.agent.mcp.store.USER_DIR", tmp_path / "user_home")
        monkeypatch.setattr("molexp.agent.mcp.defaults.seed_user_defaults", lambda *_a, **_k: False)
        (tmp_path / "mcp.json").write_text(
            '{"mcpServers": {"x": {"type": "stdio", "command": "true"}}}',
            encoding="utf-8",
        )
        catalog = McpCatalog(tmp_path)
        await catalog.open()
        assert catalog.toolsets == ()
        await catalog.aclose()


class TestFilterToolsets:
    def test_drops_classified_mutator_names(self) -> None:
        seen: list[str] = []

        class _Def:
            def __init__(self, name: str) -> None:
                self.name = name

        class _Filterable:
            def filtered(self, fn: object) -> object:
                declared = declared_requirements()

                def _allow(name: str) -> bool:
                    return CHAT_SURFACE.allows(required_keys(name, declared=declared))

                for name in ("molmcp_add_project", "molmcp_search"):
                    kept = fn(None, _Def(name))  # type: ignore[operator]
                    seen.append(name if kept else f"drop:{name}")
                    assert kept is _allow(name)
                return "wrapped"

        out = filter_toolsets(
            (_Filterable(),),
            allow=lambda n: CHAT_SURFACE.allows(required_keys(n, declared=declared_requirements())),
        )
        assert out == ("wrapped",)
        assert "drop:molmcp_add_project" in seen
        assert "molmcp_search" in seen


class TestCatalogDiscoverySurface:
    def test_chat_builtin_search_omits_archive_tools(self, tmp_path: Path) -> None:
        ctx = build_session_context(
            workspace_root=tmp_path,
            execution_env=LocalExecutionEnv(scratch_dir=tmp_path / "s"),
            surface="chat",
        )
        hits = ctx.discovery.search("ensure", kind="builtin")
        assert all(h.ref != "workspace_ensure" for h in hits)
        detail = ctx.discovery.describe("workspace_ensure")
        assert not detail.startswith("# tool workspace_ensure")


class TestCatalogDiscovery:
    def test_surfaces_mcp_specs_via_search_and_describe(self, tmp_path: Path) -> None:
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
