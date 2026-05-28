"""``ChatMode`` unit tests — plain async + sink-driven (spec 03b)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from molexp.agent.events import (
    AsyncIteratorEventSink,
    ModeCompletedEvent,
    ModeStartedEvent,
)
from molexp.agent.modes import ChatMode, ChatModeConfig
from molexp.agent.router import AgenticChunk, ModelTier, RouterTextResult
from molexp.agent.runtime import AgentRuntime
from molexp.agent.session import Session
from molexp.agent.session_entry import MessageEntry
from molexp.agent.session_storage import InMemorySessionStorage
from molexp.agent.types import UsageBreakdown

# ── config ─────────────────────────────────────────────────────────────────


def test_chat_mode_carries_config() -> None:
    cfg = ChatModeConfig(system_prompt="you are helpful")
    mode = ChatMode(config=cfg)
    assert mode.name == "chat"
    assert mode.config is cfg


def test_chat_mode_config_is_frozen() -> None:
    from pydantic import ValidationError

    cfg = ChatModeConfig(system_prompt="x")
    with pytest.raises(ValidationError):
        cfg.system_prompt = "y"  # type: ignore[misc]


# ── test doubles ───────────────────────────────────────────────────────────


class _CapturingRouter:
    """Records every ``complete_text`` call and returns canned text."""

    def __init__(self, responses: Sequence[str] = ("ok",)) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        self.calls.append({"prompt": prompt, "system": system, "tier": tier})
        text = self._responses.pop(0) if self._responses else "ok"
        return RouterTextResult(text=text)

    async def complete_structured(self, **_: object) -> object:
        raise AssertionError("ChatMode never reaches complete_structured")

    def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
    ) -> AsyncIterator[AgenticChunk]:
        async def _empty() -> AsyncIterator[AgenticChunk]:
            if False:  # pragma: no cover
                yield  # type: ignore[unreachable]

        return _empty()

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def _runtime(router: object, tmp_path_factory: Any = None) -> tuple[AgentRuntime, Session]:
    """Build a minimal :class:`AgentRuntime` for a ChatMode run."""
    import tempfile
    from pathlib import Path

    from molexp.agent.execution_env import LocalExecutionEnv
    from molexp.agent.hooks import HookRegistry

    session = Session(storage=InMemorySessionStorage(), session_id="chat")
    scratch = Path(tempfile.mkdtemp(prefix="chat_test_"))
    runtime = AgentRuntime(
        session=session,
        router=router,  # type: ignore[arg-type]
        execution_env=LocalExecutionEnv(scratch_dir=scratch),
        hooks=HookRegistry(),
    )
    return runtime, session


# ── ChatMode emits the expected sink-event sequence ────────────────────────


@pytest.mark.asyncio
async def test_chat_mode_emits_started_then_completed() -> None:
    router = _CapturingRouter(responses=["the answer"])
    runtime, _ = _runtime(router)
    sink = AsyncIteratorEventSink()
    await ChatMode().run(runtime=runtime, sink=sink, user_input="ping")
    await sink.close()
    events = [ev async for ev in sink]
    assert isinstance(events[0], ModeStartedEvent)
    assert isinstance(events[-1], ModeCompletedEvent)
    assert events[-1].text == "the answer"


@pytest.mark.asyncio
async def test_chat_mode_records_turns_in_session_tree() -> None:
    router = _CapturingRouter(responses=["a reply"])
    runtime, session = _runtime(router)
    sink = AsyncIteratorEventSink()
    await ChatMode().run(runtime=runtime, sink=sink, user_input="a question")
    await sink.close()
    messages = [e.message for e in session.path_to_root() if isinstance(e, MessageEntry)]
    roles_contents = [(m.role, m.content) for m in messages]
    assert ("user", "a question") in roles_contents
    assert ("assistant", "a reply") in roles_contents


@pytest.mark.asyncio
async def test_chat_mode_passes_system_prompt() -> None:
    router = _CapturingRouter()
    runtime, _ = _runtime(router)
    sink = AsyncIteratorEventSink()
    mode = ChatMode(config=ChatModeConfig(system_prompt="be terse"))
    await mode.run(runtime=runtime, sink=sink, user_input="hi")
    await sink.close()
    assert router.calls[0]["system"] == "be terse"


# ── round trip via pydantic-ai TestModel ───────────────────────────────────


@pytest.mark.asyncio
async def test_chat_mode_run_returns_non_empty_text_via_test_model() -> None:
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    from molexp.agent._pydanticai.router import PydanticAIRouter

    test_model = TestModel()
    router = PydanticAIRouter(
        models={
            ModelTier.CHEAP: test_model,
            ModelTier.DEFAULT: test_model,
            ModelTier.HEAVY: test_model,
        },
    )
    runtime, _ = _runtime(router)  # type: ignore[arg-type]
    sink = AsyncIteratorEventSink()
    await ChatMode().run(runtime=runtime, sink=sink, user_input="ping")
    await sink.close()
    events = [ev async for ev in sink]
    completed = events[-1]
    assert isinstance(completed, ModeCompletedEvent)
    assert completed.text
