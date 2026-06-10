"""Compaction wiring tests — the loops trigger compaction at the seam.

Both shipped loops call :func:`molexp.agent.loops._compact.maybe_compact`
after appending the user message and before the model call. These tests
prove: below the trigger budget the session is untouched; above it the
session is compacted, the loop still completes, and the invariants
:mod:`molexp.agent.compaction` promises (recent window kept verbatim,
``build_context`` swaps the old span for one system summary) hold.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from molexp.agent.compaction import CompactionSettings
from molexp.agent.events import (
    AsyncIteratorEventSink,
    CompactionPerformedEvent,
    LoopCompletedEvent,
)
from molexp.agent.loops import (
    ChatLoop,
    ChatLoopConfig,
    InteractiveLoop,
    InteractiveLoopConfig,
)
from molexp.agent.router import (
    AgenticChunk,
    FinalChunk,
    ModelTier,
    RouterTextResult,
)
from molexp.agent.runtime import AgentRuntime
from molexp.agent.session import Session
from molexp.agent.session_entry import CompactionEntry, MessageEntry
from molexp.agent.session_storage import InMemorySessionStorage
from molexp.agent.types import Message, UsageBreakdown

# Settings small enough that a handful of long messages cross the
# trigger (keep_recent_tokens + reserve_tokens = 150 estimated tokens).
_SMALL_BUDGET = CompactionSettings(keep_recent_tokens=100, reserve_tokens=50)
_BIG_MESSAGE = "x" * 400  # ~100 estimated tokens each


class _FakeRouter:
    """Canned router covering both text and agentic paths."""

    def __init__(self, responses: Sequence[str] = ("ok",), final_text: str = "done") -> None:
        self._responses = list(responses)
        self._final_text = final_text
        self.text_calls: list[dict[str, object]] = []

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        self.text_calls.append({"prompt": prompt, "system": system, "tier": tier})
        text = self._responses.pop(0) if self._responses else "ok"
        return RouterTextResult(text=text)

    async def complete_structured(self, **_: object) -> object:
        raise AssertionError("loops never reach complete_structured")

    def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
    ) -> AsyncIterator[AgenticChunk]:
        final_text = self._final_text

        async def _gen() -> AsyncIterator[AgenticChunk]:
            yield FinalChunk(text=final_text)

        return _gen()

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def _runtime(router: object, tmp_path: Any) -> tuple[AgentRuntime, Session]:
    from molexp.agent.execution_env import LocalExecutionEnv

    session = Session(storage=InMemorySessionStorage(), session_id="compact")
    runtime = AgentRuntime(
        session=session,
        router=router,  # type: ignore[arg-type]
        execution_env=LocalExecutionEnv(scratch_dir=tmp_path),
    )
    return runtime, session


def _seed_history(session: Session, turns: int) -> None:
    """Append ``turns`` long prior user/assistant message pairs."""
    for index in range(turns):
        session.append_message(Message(role="user", content=f"u{index} {_BIG_MESSAGE}"))
        session.append_message(Message(role="assistant", content=f"a{index} {_BIG_MESSAGE}"))


def _compaction_entries(session: Session) -> list[CompactionEntry]:
    return [e for e in session.path_to_root() if isinstance(e, CompactionEntry)]


# ── below threshold: untouched ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_loop_below_threshold_leaves_session_untouched(tmp_path: Any) -> None:
    router = _FakeRouter(responses=["the answer"])
    runtime, session = _runtime(router, tmp_path)
    sink = AsyncIteratorEventSink()
    loop = ChatLoop(config=ChatLoopConfig(compaction=_SMALL_BUDGET))
    await loop.run(runtime=runtime, sink=sink, user_input="short question")
    await sink.close()
    events = [ev async for ev in sink]
    assert not _compaction_entries(session)
    assert not [ev for ev in events if isinstance(ev, CompactionPerformedEvent)]
    # only the answering call reached the router — no summary call.
    assert len(router.text_calls) == 1
    assert events[-1].text == "the answer"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_chat_loop_disabled_compaction_never_fires(tmp_path: Any) -> None:
    router = _FakeRouter(responses=["the answer"])
    runtime, session = _runtime(router, tmp_path)
    _seed_history(session, turns=4)
    sink = AsyncIteratorEventSink()
    disabled = CompactionSettings(enabled=False, keep_recent_tokens=100, reserve_tokens=50)
    loop = ChatLoop(config=ChatLoopConfig(compaction=disabled))
    await loop.run(runtime=runtime, sink=sink, user_input="another question")
    await sink.close()
    assert not _compaction_entries(session)
    assert len(router.text_calls) == 1


# ── above threshold: compacted, loop still functions ───────────────────────


@pytest.mark.asyncio
async def test_chat_loop_above_threshold_compacts_and_completes(tmp_path: Any) -> None:
    router = _FakeRouter(responses=["a tidy summary", "the answer"])
    runtime, session = _runtime(router, tmp_path)
    _seed_history(session, turns=4)
    sink = AsyncIteratorEventSink()
    loop = ChatLoop(config=ChatLoopConfig(compaction=_SMALL_BUDGET))
    await loop.run(runtime=runtime, sink=sink, user_input="new question")
    await sink.close()
    events = [ev async for ev in sink]

    cuts = _compaction_entries(session)
    assert len(cuts) == 1
    assert cuts[0].summary == "a tidy summary"
    assert cuts[0].tokens_before > 0

    fired = [ev for ev in events if isinstance(ev, CompactionPerformedEvent)]
    assert len(fired) == 1
    assert fired[0].entries_summarized > 0

    # the loop still functions: terminal event carries the answer.
    completed = events[-1]
    assert isinstance(completed, LoopCompletedEvent)
    assert completed.text == "the answer"

    # the summary call used the CHEAP tier and saw the old content.
    summary_call = router.text_calls[0]
    assert summary_call["tier"] is ModelTier.CHEAP
    assert "u0" in str(summary_call["prompt"])


@pytest.mark.asyncio
async def test_chat_loop_compaction_preserves_build_context_invariants(tmp_path: Any) -> None:
    """build_context swaps the old span for one system summary message."""
    router = _FakeRouter(responses=["a tidy summary", "the answer"])
    runtime, session = _runtime(router, tmp_path)
    _seed_history(session, turns=4)
    loop = ChatLoop(config=ChatLoopConfig(compaction=_SMALL_BUDGET))
    sink = AsyncIteratorEventSink()
    await loop.run(runtime=runtime, sink=sink, user_input="new question")
    await sink.close()

    context = session.build_context()
    assert context[0].role == "system"
    assert "a tidy summary" in context[0].content
    contents = [m.content for m in context]
    # the new turn survives verbatim …
    assert "new question" in contents
    assert "the answer" in contents
    # … and the oldest span is gone from the rebuilt context.
    assert not any("u0" in c for c in contents[1:])


@pytest.mark.asyncio
async def test_chat_loop_recompaction_folds_in_prior_summary(tmp_path: Any) -> None:
    """A second cut summarizes only the post-cut span, carrying the old summary."""
    router = _FakeRouter(responses=["summary one", "answer one", "summary two", "answer two"])
    runtime, session = _runtime(router, tmp_path)
    _seed_history(session, turns=4)
    loop = ChatLoop(config=ChatLoopConfig(compaction=_SMALL_BUDGET))
    for prompt in ("q one " + _BIG_MESSAGE, "q two " + _BIG_MESSAGE):
        sink = AsyncIteratorEventSink()
        await loop.run(runtime=runtime, sink=sink, user_input=prompt)
        await sink.close()

    cuts = _compaction_entries(session)
    assert len(cuts) == 2
    # the second summary call carried the first summary forward.
    second_summary_prompt = str(router.text_calls[2]["prompt"])
    assert "[earlier summary] summary one" in second_summary_prompt
    # only the most recent cut shapes the rebuilt context.
    context = session.build_context()
    assert context[0].role == "system"
    assert "summary two" in context[0].content


@pytest.mark.asyncio
async def test_interactive_loop_above_threshold_compacts_and_completes(tmp_path: Any) -> None:
    router = _FakeRouter(responses=["a tidy summary"], final_text="tool-loop answer")
    runtime, session = _runtime(router, tmp_path)
    _seed_history(session, turns=4)
    sink = AsyncIteratorEventSink()
    loop = InteractiveLoop(
        config=InteractiveLoopConfig(compaction=_SMALL_BUDGET, workspace_root=tmp_path)
    )
    await loop.run(runtime=runtime, sink=sink, user_input="new question")
    await sink.close()
    events = [ev async for ev in sink]

    assert len(_compaction_entries(session)) == 1
    assert [ev for ev in events if isinstance(ev, CompactionPerformedEvent)]
    completed = events[-1]
    assert isinstance(completed, LoopCompletedEvent)
    assert completed.text == "tool-loop answer"
    # the new turn is recorded after the cut.
    messages = [e.message for e in session.path_to_root() if isinstance(e, MessageEntry)]
    assert ("assistant", "tool-loop answer") in [(m.role, m.content) for m in messages]


@pytest.mark.asyncio
async def test_interactive_loop_below_threshold_leaves_session_untouched(tmp_path: Any) -> None:
    router = _FakeRouter(final_text="tool-loop answer")
    runtime, session = _runtime(router, tmp_path)
    sink = AsyncIteratorEventSink()
    loop = InteractiveLoop(
        config=InteractiveLoopConfig(compaction=_SMALL_BUDGET, workspace_root=tmp_path)
    )
    await loop.run(runtime=runtime, sink=sink, user_input="short question")
    await sink.close()
    assert not _compaction_entries(session)
    assert not router.text_calls  # no summary call ever reached the router


# ── default config keeps compaction on with conservative budgets ───────────


def test_loop_configs_default_compaction_is_enabled_and_conservative() -> None:
    for config in (ChatLoopConfig(), InteractiveLoopConfig()):
        assert config.compaction.enabled is True
        assert config.compaction.keep_recent_tokens >= 8_000
