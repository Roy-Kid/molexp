"""``ChatLoop`` unit tests — the one-turn LLM loop (mirrors ``loops/chat.py``).

Loop-level ownership: ChatLoop's own contract — the sink-event sequence it
emits, the turns it records on the session entry tree, and the config
``system_prompt`` it hands to the router. ``AgentRunner`` orchestration of
ChatLoop lives in ``test_runner.py``; the compaction seam lives in
``test_compaction_wiring.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from molexp.agent.events import (
    AsyncIteratorEventSink,
    LoopCompletedEvent,
    LoopStartedEvent,
)
from molexp.agent.execution_env import LocalExecutionEnv
from molexp.agent.loops import ChatLoop, ChatLoopConfig
from molexp.agent.router import ModelTier, RouterTextResult
from molexp.agent.runtime import AgentRuntime
from molexp.agent.session import Session
from molexp.agent.session_entry import MessageEntry
from molexp.agent.session_storage import InMemorySessionStorage
from molexp.agent.types import UsageBreakdown


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
        raise AssertionError("ChatLoop never reaches complete_structured")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def _runtime(router: object, scratch: Path) -> tuple[AgentRuntime, Session]:
    """Build a minimal :class:`AgentRuntime` for a ChatLoop run."""
    session = Session(storage=InMemorySessionStorage(), session_id="chat")
    runtime = AgentRuntime(
        session=session,
        router=router,  # type: ignore[arg-type]
        execution_env=LocalExecutionEnv(scratch_dir=scratch),
    )
    return runtime, session


class TestChatLoop:
    @pytest.mark.asyncio
    async def test_run_emits_started_then_completed_with_answer(self, tmp_path: Path) -> None:
        router = _CapturingRouter(responses=["the answer"])
        runtime, _ = _runtime(router, tmp_path)
        sink = AsyncIteratorEventSink()
        await ChatLoop().run(runtime=runtime, sink=sink, user_input="ping")
        await sink.close()
        events = [ev async for ev in sink]
        assert isinstance(events[0], LoopStartedEvent)
        assert isinstance(events[-1], LoopCompletedEvent)
        assert events[-1].text == "the answer"

    @pytest.mark.asyncio
    async def test_run_records_user_and_assistant_turns_in_session(self, tmp_path: Path) -> None:
        router = _CapturingRouter(responses=["a reply"])
        runtime, session = _runtime(router, tmp_path)
        sink = AsyncIteratorEventSink()
        await ChatLoop().run(runtime=runtime, sink=sink, user_input="a question")
        await sink.close()
        messages = [e.message for e in session.path_to_root() if isinstance(e, MessageEntry)]
        roles_contents = [(m.role, m.content) for m in messages]
        assert ("user", "a question") in roles_contents
        assert ("assistant", "a reply") in roles_contents

    @pytest.mark.asyncio
    async def test_run_passes_config_system_prompt_to_router(self, tmp_path: Path) -> None:
        router = _CapturingRouter()
        runtime, _ = _runtime(router, tmp_path)
        sink = AsyncIteratorEventSink()
        loop = ChatLoop(config=ChatLoopConfig(system_prompt="be terse"))
        await loop.run(runtime=runtime, sink=sink, user_input="hi")
        await sink.close()
        assert router.calls[0]["system"] == "be terse"
