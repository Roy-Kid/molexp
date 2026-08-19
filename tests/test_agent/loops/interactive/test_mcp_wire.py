"""ReAct wires McpStore entries into ``stream_agentic(toolsets=...)``.

Owns the loop↔MCP seam only: a valid store entry becomes an opened toolset
handed to the router (happy path), and a build failure is best-effort
non-fatal (boundary). Tool-mounting and chunk translation are owned by
``test_loop.py``; scratch confinement by ``tests/test_agent/ops/test_ops_surface.py``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from molexp.agent.events import LoopCompletedEvent
from molexp.agent.router import (
    AgenticChunk,
    FinalChunk,
    ModelTier,
    TextDeltaChunk,
)
from molexp.agent.runner import AgentRunner
from molexp.agent.session import Session
from molexp.agent.session_storage import InMemorySessionStorage
from molexp.agent.types import UsageBreakdown


class _CaptureToolsetsRouter:
    """Scripted router that records the ``toolsets`` kwarg it was handed."""

    def __init__(self) -> None:
        self.last_toolsets: tuple[Any, ...] = ()
        self.stream_agentic_calls = 0

    async def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        toolsets: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
    ) -> AsyncIterator[AgenticChunk]:
        del prompt, system, tools, tier, message_history
        self.stream_agentic_calls += 1
        self.last_toolsets = toolsets
        yield TextDeltaChunk(text="ok")
        yield FinalChunk(text="ok")

    async def complete_text(self, **_: object) -> object:
        raise AssertionError("unused")

    async def complete_structured(self, **_: object) -> object:
        raise AssertionError("unused")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def _write_workspace_mcp(workspace: Path, *, name: str = "demo") -> None:
    payload = {"mcpServers": {name: {"type": "stdio", "command": "python", "args": ["-c", "pass"]}}}
    (workspace / "mcp.json").write_text(json.dumps(payload), encoding="utf-8")


def _isolate_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the store at an empty user home so no platform seed leaks in."""
    monkeypatch.setattr("molexp.agent.mcp.store.USER_DIR", tmp_path / "user_home")
    monkeypatch.setattr("molexp.agent.mcp.defaults.seed_user_defaults", lambda *_a, **_k: False)


class TestReactMcpWiring:
    """The loop's best-effort MCP-toolset seam."""

    @pytest.mark.asyncio
    async def test_valid_store_entry_is_opened_and_passed_to_stream_agentic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A valid, unshadowed entry builds one toolset that reaches the router."""
        built: list[str] = []
        sentinel = object()

        def _fake_build_mcp_server(**kwargs: Any) -> object:
            built.append(str(kwargs.get("name", "")))
            return sentinel

        monkeypatch.setattr("molexp.agent._pydanticai.mcp.build_mcp_server", _fake_build_mcp_server)
        _isolate_store(monkeypatch, tmp_path)
        _write_workspace_mcp(tmp_path)

        router = _CaptureToolsetsRouter()
        runner = AgentRunner(router=router, workspace=tmp_path, mode="agentic")  # type: ignore[arg-type]
        session = Session(storage=InMemorySessionStorage(), session_id="mcp-wire")

        events = [ev async for ev in runner.run_events(session, "list tools")]

        assert built == ["demo"]
        assert router.last_toolsets == (sentinel,)
        assert isinstance(events[-1], LoopCompletedEvent)

    @pytest.mark.asyncio
    async def test_build_failure_is_non_fatal_and_yields_empty_toolsets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raising ``build_mcp_server`` is swallowed; the turn still completes."""

        def _boom(**_kwargs: Any) -> object:
            raise RuntimeError("mcp unavailable")

        monkeypatch.setattr("molexp.agent._pydanticai.mcp.build_mcp_server", _boom)
        _isolate_store(monkeypatch, tmp_path)
        _write_workspace_mcp(tmp_path)

        router = _CaptureToolsetsRouter()
        runner = AgentRunner(router=router, workspace=tmp_path, mode="agentic")  # type: ignore[arg-type]
        session = Session(storage=InMemorySessionStorage(), session_id="mcp-fail")

        events = [ev async for ev in runner.run_events(session, "still works")]

        assert router.last_toolsets == ()
        assert isinstance(events[-1], LoopCompletedEvent)
        assert events[-1].text == "ok"
