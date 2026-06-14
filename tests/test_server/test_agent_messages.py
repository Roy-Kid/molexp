"""Message feedback — follow-up turns on a live session (spec 00b, re-scoped).

Calls the relit ``post_user_message`` against a registry we control (the
``deps`` singleton is monkeypatched), so a follow-up turn is asserted
deterministically without TestClient background-turn races. The approval
routes stay 503 (deferred).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

import molexp.server.deps.agent_runtime as agent_runtime_deps
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
from molexp.server.agent_runtime import AgentSessionRegistry
from molexp.server.routes import agent as agent_routes
from molexp.server.schemas import UserMessageCreateRequest

pytestmark = pytest.mark.asyncio


class _ScriptedRouter:
    async def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
    ) -> AsyncIterator[AgenticChunk]:
        yield TextDeltaChunk(text="ok")
        yield FinalChunk(text="ok")

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        return RouterTextResult(text="ok")

    async def complete_structured(self, **_: object) -> object:
        raise AssertionError

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


class _BlockingRouter(_ScriptedRouter):
    async def stream_agentic(self, **_: object) -> AsyncIterator[AgenticChunk]:
        yield TextDeltaChunk(text="working")
        await asyncio.Event().wait()
        yield FinalChunk(text="never")


def _runner(tmp: Path, router: object | None = None) -> AgentRunner:
    loop = InteractiveLoop(config=InteractiveLoopConfig(workspace_root=tmp))
    return AgentRunner(loop=loop, router=router or _ScriptedRouter())  # type: ignore[arg-type]


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> AgentSessionRegistry:
    reg = AgentSessionRegistry()
    monkeypatch.setattr(agent_runtime_deps, "_agent_runtime_registry", reg)
    return reg


def _ws(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(root=str(tmp_path))


def _new_session(session_id: str) -> Session:
    return Session(storage=InMemorySessionStorage(), session_id=session_id)


async def test_message_starts_followup_turn(tmp_path: Path, registry: AgentSessionRegistry) -> None:
    rt = registry.create(
        workspace_root=str(tmp_path),
        runner=_runner(tmp_path),
        session=_new_session("s1"),
        goal="hi",
        user_input="hi",
    )
    await rt.await_finished()
    assert rt.status() == "completed"

    resp = await agent_routes.post_user_message(
        "s1", UserMessageCreateRequest(content="and now this"), workspace=_ws(tmp_path)
    )
    assert resp.message  # a MessageResponse
    # a fresh turn was spawned on the same session
    assert rt.status() in ("running", "completed")
    await rt.await_finished()
    await registry.aclose()


async def test_message_unknown_session_404(tmp_path: Path, registry: AgentSessionRegistry) -> None:
    with pytest.raises(HTTPException) as exc:
        await agent_routes.post_user_message(
            "nope", UserMessageCreateRequest(content="hi"), workspace=_ws(tmp_path)
        )
    assert exc.value.status_code == 404


async def test_message_mid_turn_409(tmp_path: Path, registry: AgentSessionRegistry) -> None:
    rt = registry.create(
        workspace_root=str(tmp_path),
        runner=_runner(tmp_path, router=_BlockingRouter()),
        session=_new_session("s1"),
        goal="work",
        user_input="work",
    )
    await asyncio.sleep(0.01)
    assert rt.status() == "running"
    with pytest.raises(HTTPException) as exc:
        await agent_routes.post_user_message(
            "s1", UserMessageCreateRequest(content="interrupt"), workspace=_ws(tmp_path)
        )
    assert exc.value.status_code == 409
    await registry.aclose()
