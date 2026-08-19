"""Compaction seam wiring — ``loops/_compact.py::maybe_compact``.

:class:`AgentRunner` calls ``maybe_compact`` after appending the user
message, before the model call (text or ReAct).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from molexp.agent.compaction import CompactionSettings
from molexp.agent.events import CompactionPerformedEvent, LoopCompletedEvent
from molexp.agent.router import (
    AgenticChunk,
    FinalChunk,
    ModelTier,
    RouterTextResult,
)
from molexp.agent.runner import AgentRunner
from molexp.agent.session import Session
from molexp.agent.session_entry import CompactionEntry, MessageEntry
from molexp.agent.session_storage import InMemorySessionStorage
from molexp.agent.types import Message, UsageBreakdown

_SMALL_BUDGET = CompactionSettings(keep_recent_tokens=100, reserve_tokens=50)
_BIG_MESSAGE = "x" * 400


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
        raise AssertionError("unused")

    def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        toolsets: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
    ) -> AsyncIterator[AgenticChunk]:
        del prompt, system, tools, toolsets, tier, message_history

        async def _gen() -> AsyncIterator[AgenticChunk]:
            yield FinalChunk(text=self._final_text)

        return _gen()

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def _session() -> Session:
    return Session(storage=InMemorySessionStorage(), session_id="compact")


def _seed_history(session: Session, turns: int) -> None:
    for index in range(turns):
        session.append_message(Message(role="user", content=f"u{index} {_BIG_MESSAGE}"))
        session.append_message(Message(role="assistant", content=f"a{index} {_BIG_MESSAGE}"))


def _compaction_entries(session: Session) -> list[CompactionEntry]:
    return [e for e in session.path_to_root() if isinstance(e, CompactionEntry)]


class TestMaybeCompact:
    @pytest.mark.asyncio
    async def test_below_trigger_leaves_session_untouched(self) -> None:
        router = _FakeRouter(responses=["the answer"])
        runner = AgentRunner(router=router, mode="text", compaction=_SMALL_BUDGET)  # type: ignore[arg-type]
        session = _session()
        result = await runner.run(session, "short question")
        assert not _compaction_entries(session)
        assert not [ev for ev in result.events if isinstance(ev, CompactionPerformedEvent)]
        assert len(router.text_calls) == 1
        assert result.text == "the answer"

    @pytest.mark.asyncio
    async def test_above_trigger_compacts_on_cheap_tier_and_completes(self) -> None:
        router = _FakeRouter(responses=["a tidy summary", "the answer"])
        runner = AgentRunner(router=router, mode="text", compaction=_SMALL_BUDGET)  # type: ignore[arg-type]
        session = _session()
        _seed_history(session, turns=4)
        result = await runner.run(session, "new question")

        cuts = _compaction_entries(session)
        assert len(cuts) == 1
        assert cuts[0].summary == "a tidy summary"
        assert cuts[0].tokens_before > 0
        fired = [ev for ev in result.events if isinstance(ev, CompactionPerformedEvent)]
        assert len(fired) == 1
        assert isinstance(result.events[-1], LoopCompletedEvent)
        assert result.text == "the answer"
        summary_call = router.text_calls[0]
        assert summary_call["tier"] is ModelTier.CHEAP
        assert "u0" in str(summary_call["prompt"])

    @pytest.mark.asyncio
    async def test_recompaction_folds_prior_summary_into_next_prompt(self) -> None:
        router = _FakeRouter(responses=["summary one", "answer one", "summary two", "answer two"])
        runner = AgentRunner(router=router, mode="text", compaction=_SMALL_BUDGET)  # type: ignore[arg-type]
        session = _session()
        _seed_history(session, turns=4)
        await runner.run(session, "q one " + _BIG_MESSAGE)
        await runner.run(session, "q two " + _BIG_MESSAGE)
        cuts = _compaction_entries(session)
        assert len(cuts) == 2
        second_summary_prompt = str(router.text_calls[2]["prompt"])
        assert "[earlier summary] summary one" in second_summary_prompt

    @pytest.mark.asyncio
    async def test_agentic_turn_wires_seam_above_trigger(self, tmp_path: Any) -> None:
        router = _FakeRouter(responses=["a tidy summary"], final_text="tool-loop answer")
        runner = AgentRunner(
            router=router,  # type: ignore[arg-type]
            mode="agentic",
            compaction=_SMALL_BUDGET,
            workspace=tmp_path,
        )
        session = _session()
        _seed_history(session, turns=4)
        result = await runner.run(session, "new question")
        assert len(_compaction_entries(session)) == 1
        assert [ev for ev in result.events if isinstance(ev, CompactionPerformedEvent)]
        assert result.text == "tool-loop answer"
        messages = [e.message for e in session.path_to_root() if isinstance(e, MessageEntry)]
        assert ("assistant", "tool-loop answer") in [(m.role, m.content) for m in messages]
