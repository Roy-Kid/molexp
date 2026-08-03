"""``InteractiveLoop`` — the emergent, sink-driven tool loop (spec 03b).

Drives the loop through :class:`AgentRunner` with a scripted fake
:class:`~molexp.agent.router.Router`. Covers the loop's own responsibilities:
translating :data:`AgenticChunk`\\ s into :data:`AgentEvent`\\ s, mounting the
default tool set (and keeping lifecycle tools off), surfacing reasoning
chunks, and persisting each turn onto the session entry tree.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from molexp.agent.events import (
    AgentEvent,
    LoopCompletedEvent,
    LoopStartedEvent,
    ThinkingDeltaEvent,
    TokenDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from molexp.agent.loops.interactive import InteractiveLoop, InteractiveLoopConfig
from molexp.agent.router import (
    AgenticChunk,
    FinalChunk,
    ModelTier,
    RouterTextResult,
    TextDeltaChunk,
    ThinkingDeltaChunk,
    ToolCallChunk,
    ToolResultChunk,
)
from molexp.agent.runner import AgentRunner
from molexp.agent.session import Session
from molexp.agent.session_entry import MessageEntry
from molexp.agent.session_storage import InMemorySessionStorage
from molexp.agent.types import UsageBreakdown

pytestmark = pytest.mark.asyncio


class _ScriptedRouter:
    """Fake Router replaying a fixed chunk script with one tool round-trip."""

    def __init__(self) -> None:
        self.stream_agentic_calls = 0
        self.last_tools: tuple[Any, ...] = ()

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
        del toolsets  # optional MCP toolsets; scripted path ignores them
        self.stream_agentic_calls += 1
        self.last_tools = tools
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
        raise AssertionError("InteractiveLoop never reaches complete_structured")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


class _ThinkingRouter(_ScriptedRouter):
    """A scripted router whose turn opens with a reasoning chunk.

    Mirrors a reasoning model (DeepSeek reasoner / Claude extended thinking):
    the model reasons before it answers, so a ``ThinkingDeltaChunk`` precedes
    the answer's ``TextDeltaChunk``\\ s.
    """

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
        del tools, toolsets
        self.stream_agentic_calls += 1
        yield ThinkingDeltaChunk(text="weighing the options")
        yield TextDeltaChunk(text="The answer.")
        yield FinalChunk(text="The answer.")


def _loop(tmp_path: Path) -> InteractiveLoop:
    return InteractiveLoop(config=InteractiveLoopConfig(workspace_root=tmp_path))


def _kinds(events: list[AgentEvent]) -> list[type]:
    return [type(event) for event in events]


def _tool_names(tools: tuple[Any, ...]) -> set[str]:
    return {getattr(t, "__name__", "") for t in tools}


class TestInteractiveLoop:
    async def test_translates_chunks_to_sink_events(self, tmp_path: Path) -> None:
        router = _ScriptedRouter()
        runner = AgentRunner(loop=_loop(tmp_path), router=router)  # type: ignore[arg-type]
        session = Session(storage=InMemorySessionStorage(), session_id="emergent")

        events = [ev async for ev in runner.run_events(session, "inspect the project")]

        assert router.stream_agentic_calls == 1
        kinds = _kinds(events)
        assert LoopStartedEvent in kinds
        assert TokenDeltaEvent in kinds
        assert ToolCallStartedEvent in kinds
        assert ToolCallCompletedEvent in kinds
        assert isinstance(events[-1], LoopCompletedEvent)
        assert events[-1].text == "Looking into it. Done."

    async def test_default_mode_mounts_chat_tools_and_omits_lifecycle(self, tmp_path: Path) -> None:
        """Chat Mode default: scratch tools mount; no ensure/land; lifecycle stays off."""
        router = _ScriptedRouter()
        loop = InteractiveLoop(
            config=InteractiveLoopConfig(workspace_root=tmp_path, operation_mode="chat")
        )
        runner = AgentRunner(loop=loop, router=router)  # type: ignore[arg-type]
        session = Session(storage=InMemorySessionStorage(), session_id="code-tools")

        _ = [ev async for ev in runner.run_events(session, "write a script")]

        names = _tool_names(router.last_tools)
        assert {"code_write", "code_run", "workspace_inspect", "discover"}.issubset(names)
        assert "workspace_ensure" not in names
        assert "run_land" not in names
        assert "cancel_run" not in names
        assert "harvest_run" not in names

    async def test_surfaces_thinking_chunk_before_answer(self, tmp_path: Path) -> None:
        router = _ThinkingRouter()
        runner = AgentRunner(loop=_loop(tmp_path), router=router)  # type: ignore[arg-type]
        session = Session(storage=InMemorySessionStorage(), session_id="thinking")

        events = [ev async for ev in runner.run_events(session, "decide")]

        thinking = [e for e in events if isinstance(e, ThinkingDeltaEvent)]
        assert thinking, "ThinkingDeltaChunk must map to a ThinkingDeltaEvent"
        assert thinking[0].text == "weighing the options"

        def first(kind: type) -> int:
            return next(i for i, e in enumerate(events) if isinstance(e, kind))

        # reasoning streams before the answer text, before the terminal result
        assert first(ThinkingDeltaEvent) < first(TokenDeltaEvent) < first(LoopCompletedEvent)
        # the final answer never includes the reasoning text
        assert isinstance(events[-1], LoopCompletedEvent)
        assert events[-1].text == "The answer."

    async def test_persists_user_and_assistant_turns_to_entry_tree(self, tmp_path: Path) -> None:
        router = _ScriptedRouter()
        runner = AgentRunner(loop=_loop(tmp_path), router=router)  # type: ignore[arg-type]
        session = Session(storage=InMemorySessionStorage(), session_id="turns")

        async for _ in runner.run_events(session, "what is here?"):
            pass

        messages = [e.message for e in session.path_to_root() if isinstance(e, MessageEntry)]
        roles_contents = [(m.role, m.content) for m in messages]
        assert ("user", "what is here?") in roles_contents
        assert ("assistant", "Looking into it. Done.") in roles_contents
