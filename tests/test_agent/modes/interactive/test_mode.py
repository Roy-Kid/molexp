"""``InteractiveMode`` — emergent loop, sink-driven (spec 03b).

Drives the mode through :class:`AgentRunner` with a scripted fake
:class:`~molexp.agent.router.Router`. After spec 03b the ``/plan``
slash-command delegation is gone (PlanMode left the agent layer);
InteractiveMode is purely the emergent tool loop translated into
sink events.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from molexp.agent.events import (
    AgentEvent,
    ModeCompletedEvent,
    ModeStartedEvent,
    TokenDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from molexp.agent.modes.interactive import InteractiveMode, InteractiveModeConfig
from molexp.agent.router import (
    AgenticChunk,
    FinalChunk,
    ModelTier,
    RouterTextResult,
    TextDeltaChunk,
    ToolCallChunk,
    ToolResultChunk,
)
from molexp.agent.runner import AgentRunner
from molexp.agent.session import Session
from molexp.agent.session_entry import MessageEntry
from molexp.agent.session_storage import InMemorySessionStorage
from molexp.agent.types import UsageBreakdown


class _ScriptedRouter:
    """Fake :class:`~molexp.agent.router.Router` for the InteractiveMode tests.

    ``stream_agentic`` replays a fixed chunk script that includes one
    tool round-trip. ``complete_text`` / ``complete_structured`` are
    inert (the emergent loop only touches ``stream_agentic``).
    """

    def __init__(self) -> None:
        self.stream_agentic_calls = 0

    async def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
    ) -> AsyncIterator[AgenticChunk]:
        self.stream_agentic_calls += 1
        yield TextDeltaChunk(text="Looking ")
        yield TextDeltaChunk(text="into it. ")
        yield ToolCallChunk(tool_name="read_file", args_summary="path=README.md")
        yield ToolResultChunk(tool_name="read_file", result_summary="20 lines", ok=True)
        yield TextDeltaChunk(text="Done.")
        yield FinalChunk(text="Looking into it. Done.")

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
        raise AssertionError("InteractiveMode never reaches complete_structured")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def _mode(tmp_path: Path) -> InteractiveMode:
    return InteractiveMode(config=InteractiveModeConfig(workspace_root=tmp_path))


def _kinds(events: list[AgentEvent]) -> list[type]:
    return [type(event) for event in events]


@pytest.mark.asyncio
async def test_emergent_loop_translates_chunks_to_events(tmp_path: Path) -> None:
    router = _ScriptedRouter()
    runner = AgentRunner(mode=_mode(tmp_path), router=router)  # type: ignore[arg-type]
    session = Session(storage=InMemorySessionStorage(), session_id="emergent")

    events = [ev async for ev in runner.run_events(session, "inspect the project")]

    assert router.stream_agentic_calls == 1
    kinds = _kinds(events)
    assert ModeStartedEvent in kinds
    assert TokenDeltaEvent in kinds
    assert ToolCallStartedEvent in kinds
    assert ToolCallCompletedEvent in kinds
    assert isinstance(events[-1], ModeCompletedEvent)
    assert events[-1].text == "Looking into it. Done."


@pytest.mark.asyncio
async def test_emergent_loop_event_ordering(tmp_path: Path) -> None:
    router = _ScriptedRouter()
    runner = AgentRunner(mode=_mode(tmp_path), router=router)  # type: ignore[arg-type]
    session = Session(storage=InMemorySessionStorage(), session_id="ordering")

    events = [ev async for ev in runner.run_events(session, "inspect")]

    def first(kind: type) -> int:
        return next(i for i, e in enumerate(events) if isinstance(e, kind))

    assert (
        first(ModeStartedEvent)
        < first(TokenDeltaEvent)
        < first(ToolCallStartedEvent)
        < first(ToolCallCompletedEvent)
        < first(ModeCompletedEvent)
    )


@pytest.mark.asyncio
async def test_tool_call_events_carry_names_and_summaries(tmp_path: Path) -> None:
    router = _ScriptedRouter()
    runner = AgentRunner(mode=_mode(tmp_path), router=router)  # type: ignore[arg-type]
    session = Session(storage=InMemorySessionStorage(), session_id="toolnames")

    events = [ev async for ev in runner.run_events(session, "inspect")]

    started = next(e for e in events if isinstance(e, ToolCallStartedEvent))
    completed = next(e for e in events if isinstance(e, ToolCallCompletedEvent))
    assert started.tool_name == "read_file"
    assert "README.md" in started.args_summary
    assert completed.tool_name == "read_file"
    assert completed.ok is True


@pytest.mark.asyncio
async def test_emergent_loop_persists_user_and_assistant_turns(tmp_path: Path) -> None:
    router = _ScriptedRouter()
    runner = AgentRunner(mode=_mode(tmp_path), router=router)  # type: ignore[arg-type]
    session = Session(storage=InMemorySessionStorage(), session_id="turns")

    async for _ in runner.run_events(session, "what is here?"):
        pass

    messages = [e.message for e in session.path_to_root() if isinstance(e, MessageEntry)]
    roles_contents = [(m.role, m.content) for m in messages]
    assert ("user", "what is here?") in roles_contents
    assert ("assistant", "Looking into it. Done.") in roles_contents


@pytest.mark.asyncio
async def test_run_returns_terminal_result(tmp_path: Path) -> None:
    router = _ScriptedRouter()
    runner = AgentRunner(mode=_mode(tmp_path), router=router)  # type: ignore[arg-type]
    session = Session(storage=InMemorySessionStorage(), session_id="result")

    result = await runner.run(session, "inspect")

    assert result.text == "Looking into it. Done."
    assert any(isinstance(e, TokenDeltaEvent) for e in result.events)


def test_interactive_mode_name() -> None:
    mode = InteractiveMode()
    assert mode.name == "interactive"
