"""``ChatMode`` unit tests — harness-based reference mode (spec 02)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from molexp.agent.events import ModeCompletedEvent, ModeStartedEvent
from molexp.agent.modes import ChatMode, ChatModeConfig
from molexp.agent.router import ModelTier, RouterTextResult
from molexp.agent.runtime import AgentHarness
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
        message_history: tuple[object, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        self.calls.append({"prompt": prompt, "system": system, "tier": tier})
        text = self._responses.pop(0) if self._responses else "ok"
        return RouterTextResult(text=text)

    async def complete_structured(self, **_: object) -> object:
        raise AssertionError("ChatMode never reaches complete_structured")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def _harness(router: _CapturingRouter) -> tuple[AgentHarness, list[object]]:
    sink_events: list[object] = []

    async def sink(ev: object) -> None:
        sink_events.append(ev)

    session = Session(storage=InMemorySessionStorage(), session_id="chat")
    harness = AgentHarness(session=session, event_sink=sink, router=router)  # type: ignore[arg-type]
    return harness, sink_events


# ── run() yields an event stream ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_mode_run_yields_mode_completed() -> None:
    router = _CapturingRouter(responses=["the answer"])
    harness, _ = _harness(router)
    events = [ev async for ev in ChatMode().run(harness=harness, user_input="ping")]
    assert isinstance(events[-1], ModeCompletedEvent)
    assert events[-1].text == "the answer"


@pytest.mark.asyncio
async def test_chat_mode_emits_mode_started() -> None:
    router = _CapturingRouter()
    harness, sink_events = _harness(router)
    async for _ in ChatMode().run(harness=harness, user_input="ping"):
        pass
    assert any(isinstance(e, ModeStartedEvent) for e in sink_events)


@pytest.mark.asyncio
async def test_chat_mode_records_turns_in_session_tree() -> None:
    router = _CapturingRouter(responses=["a reply"])
    harness, _ = _harness(router)
    async for _ in ChatMode().run(harness=harness, user_input="a question"):
        pass
    messages = [e.message for e in harness.session.path_to_root() if isinstance(e, MessageEntry)]
    roles_contents = [(m.role, m.content) for m in messages]
    assert ("user", "a question") in roles_contents
    assert ("assistant", "a reply") in roles_contents


@pytest.mark.asyncio
async def test_chat_mode_threads_prior_context_into_prompt() -> None:
    """A second turn's system preamble carries the first turn's content."""
    router = _CapturingRouter(responses=["first-reply", "second-reply"])
    harness, _ = _harness(router)
    mode = ChatMode()
    async for _ in mode.run(harness=harness, user_input="first"):
        pass
    async for _ in mode.run(harness=harness, user_input="second"):
        pass
    # the second call's system prompt references the first conversation turn.
    second_call = router.calls[1]
    assert "first" in str(second_call["system"])


@pytest.mark.asyncio
async def test_chat_mode_passes_system_prompt() -> None:
    router = _CapturingRouter()
    harness, _ = _harness(router)
    mode = ChatMode(config=ChatModeConfig(system_prompt="be terse"))
    async for _ in mode.run(harness=harness, user_input="hi"):
        pass
    assert "be terse" in str(router.calls[0]["system"])


# ── round trip via TestModel ───────────────────────────────────────────────


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
    harness, _ = _harness_with_router(router)
    events = [ev async for ev in ChatMode().run(harness=harness, user_input="ping")]
    completed = events[-1]
    assert isinstance(completed, ModeCompletedEvent)
    assert completed.text


def _harness_with_router(router: object) -> tuple[AgentHarness, list[object]]:
    sink_events: list[object] = []

    async def sink(ev: object) -> None:
        sink_events.append(ev)

    session = Session(storage=InMemorySessionStorage(), session_id="chat")
    harness = AgentHarness(session=session, event_sink=sink, router=router)  # type: ignore[arg-type]
    return harness, sink_events
