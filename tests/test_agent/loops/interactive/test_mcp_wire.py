"""InteractiveLoop opens McpStore entries and passes toolsets to stream_agentic."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from molexp.agent.events import LoopCompletedEvent
from molexp.agent.loops.interactive import InteractiveLoop, InteractiveLoopConfig
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
    """Scripted router that records ``toolsets`` / ``tools`` kwargs."""

    def __init__(self) -> None:
        self.last_toolsets: tuple[Any, ...] = ()
        self.last_tools: tuple[Any, ...] = ()
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
        del prompt, system, tier, message_history
        self.stream_agentic_calls += 1
        self.last_tools = tools
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
    payload = {
        "mcpServers": {
            name: {
                "type": "stdio",
                "command": "python",
                "args": ["-c", "pass"],
            }
        }
    }
    (workspace / "mcp.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.asyncio
async def test_interactive_loop_opens_mcp_store_toolsets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-003: valid unshadowed entry → build_mcp_server once, toolsets len 1."""
    built: list[str] = []
    sentinel = object()

    def _fake_build_mcp_server(**kwargs: Any) -> object:
        built.append(str(kwargs.get("name", "")))
        return sentinel

    monkeypatch.setattr(
        "molexp.agent._pydanticai.mcp.build_mcp_server",
        _fake_build_mcp_server,
    )
    # Isolate user home so platform seed does not add extra servers.
    monkeypatch.setattr(
        "molexp.agent.mcp.store.USER_DIR",
        tmp_path / "user_home",
    )
    # Prevent seed writing into real HOME if path still resolves oddly.
    monkeypatch.setattr(
        "molexp.agent.mcp.defaults.seed_user_defaults",
        lambda *_a, **_k: False,
    )

    _write_workspace_mcp(tmp_path)
    router = _CaptureToolsetsRouter()
    loop = InteractiveLoop(config=InteractiveLoopConfig(workspace_root=tmp_path))
    runner = AgentRunner(loop=loop, router=router)  # type: ignore[arg-type]
    session = Session(storage=InMemorySessionStorage(), session_id="mcp-wire")

    events = [ev async for ev in runner.run_events(session, "list tools")]

    assert router.stream_agentic_calls == 1
    assert built == ["demo"]
    assert router.last_toolsets == (sentinel,)
    assert any(t.__name__ == "code_write" for t in router.last_tools)
    assert isinstance(events[-1], LoopCompletedEvent)


@pytest.mark.asyncio
async def test_mcp_build_failure_is_non_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-004: build_mcp_server raises → turn still completes."""

    def _boom(**_kwargs: Any) -> object:
        raise RuntimeError("mcp unavailable")

    monkeypatch.setattr(
        "molexp.agent._pydanticai.mcp.build_mcp_server",
        _boom,
    )
    monkeypatch.setattr("molexp.agent.mcp.store.USER_DIR", tmp_path / "user_home")
    monkeypatch.setattr(
        "molexp.agent.mcp.defaults.seed_user_defaults",
        lambda *_a, **_k: False,
    )

    _write_workspace_mcp(tmp_path)
    router = _CaptureToolsetsRouter()
    loop = InteractiveLoop(config=InteractiveLoopConfig(workspace_root=tmp_path))
    runner = AgentRunner(loop=loop, router=router)  # type: ignore[arg-type]
    session = Session(storage=InMemorySessionStorage(), session_id="mcp-fail")

    events = [ev async for ev in runner.run_events(session, "still works")]

    assert router.stream_agentic_calls == 1
    assert router.last_toolsets == ()
    assert isinstance(events[-1], LoopCompletedEvent)
    assert events[-1].text == "ok"


@pytest.mark.asyncio
async def test_empty_mcp_store_passes_empty_toolsets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("molexp.agent.mcp.store.USER_DIR", tmp_path / "user_home")
    monkeypatch.setattr(
        "molexp.agent.mcp.defaults.seed_user_defaults",
        lambda *_a, **_k: False,
    )
    router = _CaptureToolsetsRouter()
    loop = InteractiveLoop(config=InteractiveLoopConfig(workspace_root=tmp_path))
    runner = AgentRunner(loop=loop, router=router)  # type: ignore[arg-type]
    session = Session(storage=InMemorySessionStorage(), session_id="mcp-empty")

    _ = [ev async for ev in runner.run_events(session, "hi")]

    assert router.last_toolsets == ()
