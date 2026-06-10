"""``server.agent_runtime`` core — registry / runtime / turn (spec 00a).

These exercise the runtime subsystem in isolation (no FastAPI): a scripted
:class:`~molexp.agent.router.Router` drives a real :class:`AgentRunner` +
:class:`InteractiveLoop` so a background turn produces a live ``AgentEvent``
stream the runtime collects — no LLM, no network.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from molexp.agent.loops.interactive import InteractiveLoop, InteractiveLoopConfig
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
from molexp.server.agent_runtime import (
    AgentSessionRegistry,
    AgentSessionRuntime,
    AgentTurn,
)

pytestmark = pytest.mark.asyncio


class _ScriptedRouter:
    """A fake Router whose ``stream_agentic`` replays a fixed short turn."""

    async def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
    ) -> AsyncIterator[AgenticChunk]:
        yield TextDeltaChunk(text="Hi ")
        yield TextDeltaChunk(text="there.")
        yield FinalChunk(text="Hi there.")

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        return RouterTextResult(text=f"echo:{prompt}")

    async def complete_structured(self, **_: object) -> object:
        raise AssertionError("not used")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


class _BlockingRouter(_ScriptedRouter):
    """A fake Router whose turn never finishes until cancelled."""

    async def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
    ) -> AsyncIterator[AgenticChunk]:
        yield TextDeltaChunk(text="working")
        await asyncio.Event().wait()  # never set → blocks until cancelled
        yield FinalChunk(text="never")


def _runner(tmp: Path, router: object | None = None) -> AgentRunner:
    loop = InteractiveLoop(config=InteractiveLoopConfig(workspace_root=tmp))
    return AgentRunner(loop=loop, router=router or _ScriptedRouter())  # type: ignore[arg-type]


def _session(session_id: str) -> Session:
    return Session(storage=InMemorySessionStorage(), session_id=session_id)


async def test_registry_create_get_list_is_per_workspace(tmp_path: Path) -> None:
    reg = AgentSessionRegistry()
    rt = reg.create(
        workspace_root="/ws/a",
        runner=_runner(tmp_path),
        session=_session("s1"),
        goal="inspect",
        user_input="inspect",
    )
    assert isinstance(rt, AgentSessionRuntime)
    assert reg.get("/ws/a", "s1") is rt
    assert reg.list_runtimes("/ws/a") == [rt]
    # a different workspace root is isolated
    assert reg.get("/ws/b", "s1") is None
    assert reg.list_runtimes("/ws/b") == []
    await reg.aclose()


async def test_turn_runs_and_collects_events(tmp_path: Path) -> None:
    reg = AgentSessionRegistry()
    rt = reg.create(
        workspace_root="/ws/a",
        runner=_runner(tmp_path),
        session=_session("s1"),
        goal="hi",
        user_input="hi",
    )
    await rt.await_finished()
    kinds = [e.kind for e in rt.events()]
    assert "loop_started" in kinds
    assert "token_delta" in kinds
    assert kinds[-1] == "loop_completed"
    assert rt.status() == "completed"
    await reg.aclose()


async def test_turn_records_failure(tmp_path: Path) -> None:
    class _BoomRouter(_ScriptedRouter):
        async def stream_agentic(self, **_: object) -> AsyncIterator[AgenticChunk]:
            raise RuntimeError("boom")
            yield  # pragma: no cover - unreachable, makes this an async generator

    turn = AgentTurn.start(
        runner=_runner(tmp_path, router=_BoomRouter()),
        session=_session("s1"),
        user_input="hi",
    )
    await turn.await_finished()
    assert turn.status == "failed"


async def test_aclose_cancels_in_flight_turns(tmp_path: Path) -> None:
    reg = AgentSessionRegistry()
    rt = reg.create(
        workspace_root="/ws/a",
        runner=_runner(tmp_path, router=_BlockingRouter()),
        session=_session("s1"),
        goal="work",
        user_input="work",
    )
    # let the turn start and emit its first event, then tear down
    await asyncio.sleep(0.01)
    assert rt.status() == "running"
    await reg.aclose()  # must cancel + await without raising
    assert rt.status() == "cancelled"
