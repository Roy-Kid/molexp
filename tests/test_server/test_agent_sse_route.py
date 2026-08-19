"""Shell tests for live AgentEvent SSE streaming (routes/agent.py::stream_events
plus agent_runtime/serialize.py frame helpers).

The route is driven by consuming its ``StreamingResponse.body_iterator``
directly (deterministic — no TestClient background-turn race). Replay-then-tail
uses a gated Router so the subscribe seam is exercised mid-turn.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

import molexp.server.deps.agent_runtime as agent_runtime_deps
from molexp.agent.events import TokenDeltaEvent
from molexp.agent.router import (
    AgenticChunk,
    FinalChunk,
    ModelTier,
    RouterTextResult,
    TextDeltaChunk,
)
from molexp.agent.runner import AgentRunner
from molexp.agent.session import Session
from molexp.agent.session_storage import InMemorySessionStorage
from molexp.agent.types import UsageBreakdown
from molexp.server.agent_runtime import AgentSessionRegistry, serialize
from molexp.server.routes import agent as agent_routes


class _ScriptedRouter:
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
        yield TextDeltaChunk(text="hi")
        yield FinalChunk(text="hi")

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        return RouterTextResult(text="hi")

    async def complete_structured(self, **_: object) -> object:
        raise AssertionError

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


class _BoomRouter(_ScriptedRouter):
    async def stream_agentic(self, **_: object) -> AsyncIterator[AgenticChunk]:
        raise RuntimeError("kaboom")
        yield  # pragma: no cover - makes this an async generator


class _GatedRouter(_ScriptedRouter):
    def __init__(self, gate: asyncio.Event) -> None:
        self._gate = gate

    async def stream_agentic(self, **_: object) -> AsyncIterator[AgenticChunk]:
        yield TextDeltaChunk(text="a")
        yield TextDeltaChunk(text="b")
        await self._gate.wait()
        yield TextDeltaChunk(text="c")
        yield FinalChunk(text="abc")


def _runner(tmp: Path, router: object | None = None) -> AgentRunner:
    # Empty workspace root is fine; pin MCP open/list to no-ops so tests never
    # wait on a live molmcp stdio process during stream_agentic setup.
    return AgentRunner(
        router=router or _ScriptedRouter(),  # type: ignore[arg-type]
        workspace=tmp,
        mode="agentic",
    )


class _EmptyMcpCatalog:
    """Turn catalog that opens nothing — isolates tests from live MCP."""

    def __init__(self, _root: object) -> None:
        del _root

    @property
    def toolsets(self) -> tuple[object, ...]:
        return ()

    async def open(self) -> None:
        return None

    async def list_specs(self) -> tuple[()]:
        return ()

    async def aclose(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _stub_mcp_toolsets(monkeypatch: pytest.MonkeyPatch) -> None:
    """ReAct opens MCP toolsets before streaming; stub for speed.

    Patch the name bound in ``loop`` (not the source module) so the turn
    never waits on a live molmcp stdio process.
    """
    import molexp.agent.react as react_mod

    monkeypatch.setattr(react_mod, "McpCatalog", _EmptyMcpCatalog)


def _ws(tmp: Path) -> SimpleNamespace:
    return SimpleNamespace(root=str(tmp))


def _session(sid: str) -> Session:
    return Session(storage=InMemorySessionStorage(), session_id=sid)


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> AgentSessionRegistry:
    reg = AgentSessionRegistry()
    monkeypatch.setattr(agent_runtime_deps, "_agent_runtime_registry", reg)
    return reg


async def _drain(resp: Any) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    async for chunk in resp.body_iterator:
        text = chunk if isinstance(chunk, str) else chunk.decode()
        for part in text.split("\n\n"):
            part = part.strip()
            if part.startswith("data: "):
                frames.append(json.loads(part[len("data: ") :]))
    return frames


class TestSerialize:
    def test_frames_render_the_sse_data_envelope(self) -> None:
        frame = serialize.event_to_sse_frame(TokenDeltaEvent(text="x"))
        assert frame.startswith("data: ") and frame.endswith("\n\n")
        assert json.loads(frame[len("data: ") :])["kind"] == "token_delta"
        assert json.loads(serialize.done_frame()[len("data: ") :])["type"] == "done"
        err = json.loads(serialize.error_frame("boom")[len("data: ") :])
        assert err == {"type": "error", "message": "boom"}


class TestStreamEventsRoute:
    @pytest.mark.asyncio
    async def test_streams_ordered_frames_then_done(
        self, tmp_path: Path, registry: AgentSessionRegistry
    ) -> None:
        rt = registry.create(
            workspace_root=str(tmp_path),
            runner=_runner(tmp_path),
            session=_session("s1"),
            goal="hi",
            user_input="hi",
        )
        await rt.await_finished()

        resp = await agent_routes.stream_events("s1", workspace=_ws(tmp_path))
        assert resp.media_type == "text/event-stream"
        assert resp.headers["cache-control"] == "no-cache"
        assert resp.headers["x-accel-buffering"] == "no"

        frames = await _drain(resp)
        kinds = [f.get("kind") or f.get("type") for f in frames]
        assert kinds[0] == "loop_started"
        assert "token_delta" in kinds
        assert kinds[-2] == "loop_completed"
        assert kinds[-1] == "done"
        await registry.aclose()

    @pytest.mark.asyncio
    async def test_unknown_session_404_before_stream(
        self, tmp_path: Path, registry: AgentSessionRegistry
    ) -> None:
        with pytest.raises(HTTPException) as exc:
            await agent_routes.stream_events("nope", workspace=_ws(tmp_path))
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_failed_turn_yields_one_error_frame(
        self, tmp_path: Path, registry: AgentSessionRegistry
    ) -> None:
        rt = registry.create(
            workspace_root=str(tmp_path),
            runner=_runner(tmp_path, router=_BoomRouter()),
            session=_session("s1"),
            goal="x",
            user_input="x",
        )
        await rt.await_finished()
        assert rt.status() == "failed"

        frames = await _drain(await agent_routes.stream_events("s1", workspace=_ws(tmp_path)))
        error_frames = [f for f in frames if f.get("type") == "error"]
        assert len(error_frames) == 1
        assert not any(f.get("type") == "done" for f in frames)
        await registry.aclose()

    @pytest.mark.asyncio
    async def test_replay_then_tail_emits_each_event_exactly_once(
        self, tmp_path: Path, registry: AgentSessionRegistry
    ) -> None:
        gate = asyncio.Event()
        rt = registry.create(
            workspace_root=str(tmp_path),
            runner=_runner(tmp_path, router=_GatedRouter(gate)),
            session=_session("s1"),
            goal="x",
            user_input="x",
        )
        # wait until the pre-gate events (loop_started + 2 token deltas) are collected
        for _ in range(200):
            if len(rt.events()) >= 3:
                break
            await asyncio.sleep(0)
        assert len(rt.events()) >= 3

        sub = rt.subscribe_events()
        replay = [await sub.__anext__() for _ in range(3)]  # snapshot at subscribe time
        assert [e.kind for e in replay] == ["loop_started", "token_delta", "token_delta"]

        gate.set()  # release the rest of the turn → tail
        tail = [event async for event in sub]
        assert [e.kind for e in tail] == ["token_delta", "loop_completed"]

        # union is each event exactly once, in order
        all_kinds = [e.kind for e in replay + tail]
        assert all_kinds == [
            "loop_started",
            "token_delta",
            "token_delta",
            "token_delta",
            "loop_completed",
        ]
        await rt.await_finished()
        await registry.aclose()
