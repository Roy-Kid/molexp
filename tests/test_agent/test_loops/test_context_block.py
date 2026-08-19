"""AgentRunner ``context_block`` composition."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from molexp.agent.router import FinalChunk
from molexp.agent.runner import AgentRunner
from molexp.agent.session import Session
from molexp.agent.session_storage import InMemorySessionStorage
from molexp.agent.types import UsageBreakdown

pytestmark = pytest.mark.asyncio

_BASE_PROMPT = "You are the molexp assistant."
_CONTEXT_BLOCK = "## Mounted run\n\n- sigma: 0.25\n- status: succeeded"


class _CapturingRouter:
    """Fake Router capturing the ``system=`` handed to ``stream_agentic``."""

    def __init__(self) -> None:
        self.system: str | None = None

    async def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[object, ...] = (),
        **_: object,
    ) -> AsyncIterator[object]:
        self.system = system
        yield FinalChunk(text="ok")

    async def complete_text(self, **_: object) -> object:
        raise AssertionError("unused")

    async def complete_structured(self, **_: object) -> object:
        raise AssertionError("unused")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


async def _captured_system(*, system_prompt: str, context_block: str, tmp_path: Path) -> str:
    router = _CapturingRouter()
    runner = AgentRunner(
        router=router,  # type: ignore[arg-type]
        workspace=tmp_path,
        mode="agentic",
        system_prompt=system_prompt,
        context_block=context_block,
    )
    session = Session(storage=InMemorySessionStorage(), session_id="ctx-block")
    async for _ in runner.run_events(session, "hello"):
        pass
    assert router.system is not None, "stream_agentic was never reached"
    return router.system


class TestContextBlockComposition:
    async def test_context_block_lands_after_user_system_prompt(self, tmp_path: Path) -> None:
        system = await _captured_system(
            system_prompt=_BASE_PROMPT, context_block=_CONTEXT_BLOCK, tmp_path=tmp_path
        )
        assert _BASE_PROMPT in system
        assert _CONTEXT_BLOCK in system
        assert system.index(_BASE_PROMPT) < system.index(_CONTEXT_BLOCK)

    async def test_empty_context_block_injects_nothing(self, tmp_path: Path) -> None:
        system = await _captured_system(
            system_prompt=_BASE_PROMPT, context_block="", tmp_path=tmp_path
        )
        assert _BASE_PROMPT in system
        assert _CONTEXT_BLOCK not in system
