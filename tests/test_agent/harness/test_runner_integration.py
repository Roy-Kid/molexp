"""Harness-based public contract — ``AgentMode`` / ``AgentRunner`` (ac-011, ac-012)."""

from __future__ import annotations

import inspect

import pytest

from molexp.agent.harness.events import (
    AgentEvent,
    ModeCompletedEvent,
    ModeStartedEvent,
    StageCompletedEvent,
    StageStartedEvent,
)
from molexp.agent.harness.harness import AgentHarness
from molexp.agent.harness.session import Session
from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.runner import AgentRunner
from molexp.agent.types import Message


class StubMode(AgentMode):
    """A minimal harness-based mode for integration tests.

    Emits a ``mode_started``, opens one stage, then yields a terminal
    ``mode_completed`` carrying the :class:`AgentRunResult`.
    """

    name = "stub"

    async def run(self, *, harness: AgentHarness, user_input: str):  # type: ignore[override]
        await harness.emit(ModeStartedEvent(mode_name=self.name, user_input=user_input))
        async with harness.stage("only-stage"):
            harness.session.append_message(Message(role="user", content=user_input))
        result = AgentRunResult(text=f"handled:{user_input}")
        yield ModeCompletedEvent(text=result.text, result=result.model_dump(mode="json"))


# ── AgentMode.run signature (ac-011) ───────────────────────────────────────


def test_agent_mode_run_is_harness_based() -> None:
    sig = inspect.signature(AgentMode.run)
    params = list(sig.parameters)
    assert "harness" in params
    assert "user_input" in params
    assert "session" not in params  # the old contract is gone
    assert "router" not in params


def test_agent_run_result_has_events_field() -> None:
    fields = AgentRunResult.model_fields
    assert "events" in fields
    assert AgentRunResult(text="x").events == ()
    # prior fields intact
    for name in ("text", "messages", "mode_state", "usage", "usage_breakdown"):
        assert name in fields


def test_agent_run_result_events_round_trip() -> None:
    ev = StageStartedEvent(stage_name="s")
    result = AgentRunResult(text="x", events=(ev,))
    assert result.events == (ev,)


# ── AgentMode.run yields an AgentEvent stream ──────────────────────────────


@pytest.mark.asyncio
async def test_stub_mode_yields_event_stream() -> None:
    from molexp.agent.harness.session_storage import InMemorySessionStorage

    collected: list[AgentEvent] = []

    async def sink(ev: AgentEvent) -> None:
        collected.append(ev)

    session = Session(storage=InMemorySessionStorage())
    harness = AgentHarness(session=session, event_sink=sink)
    mode = StubMode()
    streamed = [ev async for ev in mode.run(harness=harness, user_input="hi")]
    # the generator's own yields end with mode_completed
    assert isinstance(streamed[-1], ModeCompletedEvent)
    # the harness sink saw the lifecycle events
    assert any(isinstance(e, ModeStartedEvent) for e in collected)
    assert any(isinstance(e, StageStartedEvent) for e in collected)
    assert any(isinstance(e, StageCompletedEvent) for e in collected)


# ── AgentRunner.run drives the stub mode (ac-012) ──────────────────────────


@pytest.mark.asyncio
async def test_runner_run_returns_terminal_result_with_events() -> None:
    runner = AgentRunner(mode=StubMode(), router=_NullRouter())
    session = runner.session("s1")
    result = await runner.run(session, "ping")
    assert isinstance(result, AgentRunResult)
    assert result.text == "handled:ping"
    # the accumulated stream is on result.events
    assert len(result.events) >= 1
    assert any(isinstance(e, ModeCompletedEvent) for e in result.events)
    assert any(isinstance(e, ModeStartedEvent) for e in result.events)


@pytest.mark.asyncio
async def test_runner_run_events_yields_live_stream() -> None:
    runner = AgentRunner(mode=StubMode(), router=_NullRouter())
    session = runner.session("s2")
    streamed: list[AgentEvent] = []
    async for ev in runner.run_events(session, "pong"):
        streamed.append(ev)
    assert any(isinstance(e, ModeStartedEvent) for e in streamed)
    assert isinstance(streamed[-1], ModeCompletedEvent)


@pytest.mark.asyncio
async def test_runner_session_returns_a_session_instance() -> None:
    runner = AgentRunner(mode=StubMode(), router=_NullRouter())
    session = runner.session("named")
    assert isinstance(session, Session)
    assert session.session_id == "named"


def test_agent_session_export_resolves_to_session() -> None:
    from molexp.agent import AgentSession

    assert AgentSession is Session


class _NullRouter:
    """A do-nothing router — StubMode never calls the LLM."""

    async def complete_text(self, **_: object) -> object:
        raise AssertionError("StubMode never calls the router")

    async def complete_structured(self, **_: object) -> object:
        raise AssertionError("StubMode never calls the router")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self):  # type: ignore[no-untyped-def]
        from molexp.agent.types import UsageBreakdown

        return UsageBreakdown()
