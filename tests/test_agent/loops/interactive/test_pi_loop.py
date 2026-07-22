"""Pi-style outer steering loop of :class:`InteractiveLoop` (plan-emergent-02).

The dual of the harness ``RepairLoop`` at the agent layer: a single
``ShouldStopGuard`` drives every branch after each inner ReAct pass ends in a
``FinalChunk``. Three orthogonal behaviors:

* **steer** — a ``deny`` appends the deny message as the next user turn and
  runs another pass seeded with it, terminating in one ``LoopCompletedEvent``.
* **suspend** — a ``suspend`` outcome terminates with a ``LoopSuspendedEvent``
  (reason + leaf_id), runs no further pass, writes no pending record, and
  round-trips through the ``AgentEvent`` union.
* **bound** — an always-``deny`` guard halts at ``config.max_passes`` and still
  completes (no infinite loop).

All offline: a scripted fake ``Router`` yields canned chunks and counts its
``stream_agentic`` calls; a scripted ``ShouldStopGuard`` replays a fixed
outcome sequence. No SDK, no clock, no RNG.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from molexp.agent.compaction import CompactionSettings
from molexp.agent.events import (
    AgentEvent,
    AsyncIteratorEventSink,
    LoopCompletedEvent,
    LoopSuspendedEvent,
)
from molexp.agent.execution_env import LocalExecutionEnv
from molexp.agent.loops import LoopHooks
from molexp.agent.loops.hooks import HookOutcome, LoopState
from molexp.agent.loops.interactive import InteractiveLoop, InteractiveLoopConfig
from molexp.agent.router import (
    AgenticChunk,
    FinalChunk,
    McpToolSpec,
    ModelTier,
    RouterTextResult,
    TextDeltaChunk,
    ToolCallChunk,
    ToolResultChunk,
)
from molexp.agent.runtime import AgentRuntime
from molexp.agent.session import Session
from molexp.agent.session_entry import MessageEntry
from molexp.agent.session_storage import (
    InMemorySessionStorage,
    JsonlSessionStorage,
    SessionStorage,
)
from molexp.agent.types import UsageBreakdown

pytestmark = pytest.mark.asyncio


class _ScriptedRouter:
    """Fake ``Router`` recording each pass's prompt + history.

    Every ``stream_agentic`` call yields one tool round-trip then a
    ``FinalChunk`` whose text is ``final-<n>`` (``n`` = 1-based call count),
    so the ``ShouldStopGuard`` observes a distinct draft final per pass.
    """

    def __init__(self, *, tool_name: str = "read_file") -> None:
        self.stream_agentic_calls = 0
        self.prompts: list[str] = []
        self.histories: list[tuple[Any, ...]] = []
        self._tool_name = tool_name

    async def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        toolsets: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
        before_tool: Any = None,
        after_tool: Any = None,
        should_stop: Any = None,
    ) -> AsyncIterator[AgenticChunk]:
        del system, tools, toolsets, tier, before_tool, after_tool, should_stop
        self.stream_agentic_calls += 1
        n = self.stream_agentic_calls
        self.prompts.append(prompt)
        self.histories.append(tuple(message_history))
        yield ToolCallChunk(tool_name=self._tool_name, args_summary="")
        yield ToolResultChunk(tool_name=self._tool_name, result_summary="ok", ok=True)
        yield TextDeltaChunk(text=f"draft-{n}")
        yield FinalChunk(text=f"final-{n}")

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        return RouterTextResult(text=f"echo:{prompt}")

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[Any],
        node_id: str = "",
        mcp_tools: tuple[McpToolSpec, ...] = (),
    ) -> Any:
        raise AssertionError("InteractiveLoop never reaches complete_structured")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


class _ScriptedGuard:
    """A ``ShouldStopGuard`` replaying a fixed sequence of ``HookOutcome``\\ s.

    Call ``i`` returns ``outcomes[i]`` when in range, else ``default``. Records
    every ``LoopState`` observed so a test can assert the pass count and the
    draft-final the loop handed it.
    """

    def __init__(
        self,
        outcomes: list[HookOutcome],
        *,
        default: HookOutcome | None = None,
    ) -> None:
        self._outcomes = list(outcomes)
        self._default = default if default is not None else HookOutcome.proceed()
        self.calls = 0
        self.states: list[LoopState] = []

    async def __call__(self, *, state: LoopState) -> HookOutcome:
        self.states.append(state)
        index = self.calls
        self.calls += 1
        if index < len(self._outcomes):
            return self._outcomes[index]
        return self._default


def _runtime(
    router: _ScriptedRouter, storage: SessionStorage, tmp_path: Path, *, session_id: str
) -> tuple[AgentRuntime, Session]:
    session = Session(storage=storage, session_id=session_id)
    runtime = AgentRuntime(
        session=session,
        router=router,  # type: ignore[arg-type]
        execution_env=LocalExecutionEnv(scratch_dir=tmp_path / ".scratch"),
    )
    return runtime, session


def _config(tmp_path: Path, *, max_passes: int = 8) -> InteractiveLoopConfig:
    return InteractiveLoopConfig(
        workspace_root=tmp_path,
        compaction=CompactionSettings(enabled=False),
        max_passes=max_passes,
    )


async def _drive(loop: InteractiveLoop, runtime: AgentRuntime, user_input: str) -> list[AgentEvent]:
    """Run ``loop`` to completion and return the whole emitted event list."""
    sink = AsyncIteratorEventSink()
    await loop.run(runtime=runtime, sink=sink, user_input=user_input)
    await sink.close()
    return [event async for event in sink]


def _terminal(events: list[AgentEvent]) -> list[AgentEvent]:
    return [e for e in events if isinstance(e, (LoopCompletedEvent, LoopSuspendedEvent))]


def _user_contents(session: Session) -> list[str]:
    return [
        entry.message.content
        for entry in session.path_to_root()
        if isinstance(entry, MessageEntry) and entry.message.role == "user"
    ]


class TestInteractiveLoopSteering:
    """The Pi-style outer loop: steer / suspend / bound."""

    async def test_deny_steers_a_second_pass_seeded_with_the_deny_message(
        self, tmp_path: Path
    ) -> None:
        """One ``deny`` runs a second pass seeded (as prompt + user turn) with it."""
        router = _ScriptedRouter()
        guard = _ScriptedGuard([HookOutcome.deny("look again")])
        loop = InteractiveLoop(config=_config(tmp_path), hooks=LoopHooks(should_stop=guard))
        runtime, session = _runtime(
            router, InMemorySessionStorage(), tmp_path, session_id="steer"
        )

        events = await _drive(loop, runtime, "inspect the project")

        assert router.stream_agentic_calls == 2
        assert router.prompts[1] == "look again"
        assert "look again" in _user_contents(session)
        # The guard sees a monotonic pass count and each pass's draft final.
        assert [state.step for state in guard.states] == [1, 2]
        assert guard.states[0].draft_final == "final-1"
        # Exactly one terminal event, and it is a normal completion.
        assert len(_terminal(events)) == 1
        assert isinstance(events[-1], LoopCompletedEvent)

    async def test_suspend_terminates_with_loop_suspended_event(self, tmp_path: Path) -> None:
        """A ``suspend`` ends the loop durably with a ``LoopSuspendedEvent``."""
        router = _ScriptedRouter()
        guard = _ScriptedGuard([HookOutcome.suspend("out of budget")])
        loop = InteractiveLoop(config=_config(tmp_path), hooks=LoopHooks(should_stop=guard))
        session_dir = tmp_path / "sess"
        runtime, session = _runtime(
            router, JsonlSessionStorage(session_dir), tmp_path, session_id="suspend"
        )

        events = await _drive(loop, runtime, "long task")

        # Suspend runs no further pass and never falls back to a completion.
        assert router.stream_agentic_calls == 1
        assert not any(isinstance(e, LoopCompletedEvent) for e in events)
        terminal = _terminal(events)
        assert len(terminal) == 1
        suspended = terminal[0]
        assert isinstance(suspended, LoopSuspendedEvent)
        assert suspended.reason == "out of budget"
        # The event carries the session's live leaf id (a persisted tip).
        assert suspended.leaf_id == session.leaf_id
        assert suspended.leaf_id

        # The entry tree + leaf pointer are durably persisted (no pending store).
        assert (session_dir / "entries.jsonl").is_file()
        assert (session_dir / "leaf").is_file()

        # The suspend event round-trips through the AgentEvent discriminated union.
        adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)
        reparsed = adapter.validate_json(suspended.model_dump_json())
        assert isinstance(reparsed, LoopSuspendedEvent)
        assert reparsed.reason == "out of budget"
        assert reparsed.leaf_id == suspended.leaf_id

    async def test_always_deny_stops_at_max_passes(self, tmp_path: Path) -> None:
        """An always-``deny`` guard halts at ``max_passes`` (no infinite loop)."""
        router = _ScriptedRouter()
        guard = _ScriptedGuard([], default=HookOutcome.deny("keep going"))
        loop = InteractiveLoop(
            config=_config(tmp_path, max_passes=3),
            hooks=LoopHooks(should_stop=guard),
        )
        runtime, _session = _runtime(
            router, InMemorySessionStorage(), tmp_path, session_id="bound"
        )

        events = await _drive(loop, runtime, "never satisfied")

        assert router.stream_agentic_calls == 3
        assert not any(isinstance(e, LoopSuspendedEvent) for e in events)
        assert isinstance(events[-1], LoopCompletedEvent)
