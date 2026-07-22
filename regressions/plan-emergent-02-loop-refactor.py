"""Regression: the Pi-style ``InteractiveLoop`` outer steering loop, offline.

Binding runtime example for spec ``plan-emergent-02-loop-refactor`` (ac-009).
A library-external user drives the emergent :class:`InteractiveLoop` through
the **public** loop API only — ``molexp.agent.loops`` for the loop + hooks,
``molexp.agent.events`` for the sink + terminal events, and the SDK-free
``molexp.agent.router`` chunk vocabulary for a scripted offline fake. No
network, no model, no reach into ``_pydanticai``.

Two ac-009 reference scenarios:

1. **steer→proceed** — a ``ShouldStopGuard`` that returns
   ``HookOutcome.deny("keep going")`` once then ``HookOutcome.proceed()``
   steers a second inner pass: exactly **2** ``stream_agentic`` passes and a
   terminal :class:`LoopCompletedEvent`.
2. **suspend** — a guard that returns ``HookOutcome.suspend("out of budget")``
   terminates with a :class:`LoopSuspendedEvent` carrying ``reason`` / ``leaf_id``,
   runs **no** further pass, and writes no pending-record store (the agent layer
   has none — the durable state is the session entry tree + its ``leaf`` pointer).

Deterministic + offline: a scripted fake :class:`~molexp.agent.router.Router`
yields canned chunks and counts its own ``stream_agentic`` calls; a scripted
``ShouldStopGuard`` replays a fixed outcome sequence. Run standalone with
``python regressions/plan-emergent-02-loop-refactor.py``; the final line on
success is ``plan-emergent-02-loop-refactor: ok``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter

from molexp.agent.compaction import CompactionSettings
from molexp.agent.events import (
    AgentEvent,
    AsyncIteratorEventSink,
    LoopCompletedEvent,
    LoopSuspendedEvent,
)
from molexp.agent.execution_env import LocalExecutionEnv
from molexp.agent.loops import (
    AfterToolHook,
    BeforeToolHook,
    HookOutcome,
    InteractiveLoop,
    InteractiveLoopConfig,
    LoopHooks,
    LoopState,
    ShouldStopGuard,
)
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
from molexp.agent.session_storage import JsonlSessionStorage, SessionStorage
from molexp.agent.types import UsageBreakdown

_SchemaT = TypeVar("_SchemaT", bound=BaseModel)


class _ScriptedRouter:
    """Fake :class:`~molexp.agent.router.Router` counting each ReAct pass.

    Every ``stream_agentic`` call yields one tool round-trip then a
    ``FinalChunk`` whose text is ``final-<n>`` (``n`` = 1-based call count), so
    the ``ShouldStopGuard`` observes a distinct draft final per pass. The prompt
    handed to each call is recorded so a scenario can assert what the second
    (steered) pass received.
    """

    def __init__(self, *, tool_name: str = "read_file") -> None:
        self.stream_agentic_calls = 0
        self.prompts: list[str] = []
        self._tool_name = tool_name

    async def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[object, ...] = (),
        toolsets: tuple[object, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[object, ...] = (),
        before_tool: BeforeToolHook | None = None,
        after_tool: AfterToolHook | None = None,
        should_stop: ShouldStopGuard | None = None,
    ) -> AsyncIterator[AgenticChunk]:
        del system, tools, toolsets, tier, message_history
        del before_tool, after_tool, should_stop
        self.stream_agentic_calls += 1
        n = self.stream_agentic_calls
        self.prompts.append(prompt)
        yield ToolCallChunk(tool_name=self._tool_name, args_summary="")
        yield ToolResultChunk(tool_name=self._tool_name, result_summary="ok", ok=True)
        yield TextDeltaChunk(text=f"draft-{n}")
        yield FinalChunk(text=f"final-{n}")

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[object, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        del system, message_history, tier
        return RouterTextResult(text=f"echo:{prompt}")

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[_SchemaT],
        node_id: str = "",
        mcp_tools: tuple[McpToolSpec, ...] = (),
    ) -> _SchemaT:
        del tier, system, user, schema, node_id, mcp_tools
        raise AssertionError("InteractiveLoop never reaches complete_structured")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


class _ScriptedGuard:
    """A ``ShouldStopGuard`` replaying a fixed sequence of ``HookOutcome``\\ s.

    Call ``i`` returns ``outcomes[i]`` when in range, else ``proceed()``. Records
    every ``LoopState`` observed so a scenario can assert the pass count and the
    draft-final the loop handed it after each inner pass.
    """

    def __init__(self, outcomes: list[HookOutcome]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0
        self.states: list[LoopState] = []

    async def __call__(self, *, state: LoopState) -> HookOutcome:
        self.states.append(state)
        index = self.calls
        self.calls += 1
        if index < len(self._outcomes):
            return self._outcomes[index]
        return HookOutcome.proceed()


def _config(root: Path, *, max_passes: int = 8) -> InteractiveLoopConfig:
    return InteractiveLoopConfig(
        workspace_root=root,
        compaction=CompactionSettings(enabled=False),
        max_passes=max_passes,
    )


def _runtime(
    router: _ScriptedRouter, storage: SessionStorage, root: Path, *, session_id: str
) -> tuple[AgentRuntime, Session]:
    session = Session(storage=storage, session_id=session_id)
    runtime = AgentRuntime(
        session=session,
        router=router,
        execution_env=LocalExecutionEnv(scratch_dir=root / ".scratch"),
    )
    return runtime, session


async def _drive(loop: InteractiveLoop, runtime: AgentRuntime, user_input: str) -> list[AgentEvent]:
    """Run ``loop`` to completion and return the whole emitted event list."""
    sink = AsyncIteratorEventSink()
    await loop.run(runtime=runtime, sink=sink, user_input=user_input)
    await sink.close()
    return [event async for event in sink]


def _terminal(events: list[AgentEvent]) -> list[AgentEvent]:
    return [e for e in events if isinstance(e, (LoopCompletedEvent, LoopSuspendedEvent))]


async def _scenario_steer_proceed() -> None:
    """ac-009 (1): one ``deny`` then ``proceed`` runs exactly two passes."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        router = _ScriptedRouter()
        guard = _ScriptedGuard([HookOutcome.deny("keep going")])
        loop = InteractiveLoop(config=_config(root), hooks=LoopHooks(should_stop=guard))
        runtime, _session = _runtime(
            router, JsonlSessionStorage(root / "sess"), root, session_id="steer"
        )

        events = await _drive(loop, runtime, "inspect the project")

        # Reference observations (printed before asserting).
        print(f"steer→proceed: stream_agentic passes = {router.stream_agentic_calls}")
        print(f"steer→proceed: guard steps observed  = {[s.step for s in guard.states]}")
        print(f"steer→proceed: second-pass prompt    = {router.prompts[1]!r}")
        print(f"steer→proceed: terminal event        = {type(events[-1]).__name__}")

        assert router.stream_agentic_calls == 2, (
            f"expected exactly 2 passes, saw {router.stream_agentic_calls}"
        )
        assert [s.step for s in guard.states] == [1, 2], "guard must see monotonic steps"
        assert router.prompts[1] == "keep going", "second pass must be seeded with the steer"
        terminal = _terminal(events)
        assert len(terminal) == 1, f"expected one terminal event, saw {len(terminal)}"
        assert isinstance(events[-1], LoopCompletedEvent), (
            f"steer→proceed must end in LoopCompletedEvent, got {type(events[-1]).__name__}"
        )
        assert not any(isinstance(e, LoopSuspendedEvent) for e in events), (
            "a steered proceed must never suspend"
        )


async def _scenario_suspend() -> None:
    """ac-009 (2): a ``suspend`` ends the loop with a ``LoopSuspendedEvent``."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        session_dir = root / "sess"
        router = _ScriptedRouter()
        guard = _ScriptedGuard([HookOutcome.suspend("out of budget")])
        loop = InteractiveLoop(config=_config(root), hooks=LoopHooks(should_stop=guard))
        runtime, session = _runtime(
            router, JsonlSessionStorage(session_dir), root, session_id="suspend"
        )

        events = await _drive(loop, runtime, "long task")

        terminal = _terminal(events)
        assert len(terminal) == 1, f"expected one terminal event, saw {len(terminal)}"
        suspended = terminal[0]

        # Reference observations (printed before asserting).
        print(f"suspend: stream_agentic passes = {router.stream_agentic_calls}")
        print(f"suspend: guard calls           = {guard.calls}")
        print(f"suspend: terminal event        = {type(suspended).__name__}")
        if isinstance(suspended, LoopSuspendedEvent):
            print(f"suspend: reason = {suspended.reason!r}, leaf_id = {suspended.leaf_id!r}")

        # No extra pass after suspend, and no fall-through to a completion.
        assert router.stream_agentic_calls == 1, (
            f"suspend must run no further pass, saw {router.stream_agentic_calls}"
        )
        assert not any(isinstance(e, LoopCompletedEvent) for e in events), (
            "suspend must not also emit a LoopCompletedEvent"
        )
        assert isinstance(suspended, LoopSuspendedEvent), (
            f"suspend must end in LoopSuspendedEvent, got {type(suspended).__name__}"
        )
        assert suspended.reason == "out of budget", "suspend event must carry the reason"
        assert suspended.leaf_id == session.leaf_id and suspended.leaf_id, (
            "suspend event must anchor on the session's live persisted leaf id"
        )

        # No pending-record store exists in the agent layer: the durable resume
        # anchor is the session entry tree + its leaf pointer, both on disk.
        assert (session_dir / "entries.jsonl").is_file(), "entry tree must be persisted"
        assert (session_dir / "leaf").is_file(), "leaf pointer must be persisted"

        # The suspend event round-trips through the AgentEvent discriminated union.
        adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)
        reparsed = adapter.validate_json(suspended.model_dump_json())
        assert isinstance(reparsed, LoopSuspendedEvent), "union must route back to suspend"
        assert reparsed.reason == suspended.reason
        assert reparsed.leaf_id == suspended.leaf_id


async def main() -> int:
    """Run both ac-009 scenarios; print the success marker as the final line."""
    await _scenario_steer_proceed()
    await _scenario_suspend()
    print("plan-emergent-02-loop-refactor: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
