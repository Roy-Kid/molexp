"""``InteractiveLoopConfig.context_block`` → ``stream_agentic(system=…)``.

Composition order: ops preamble → user ``system_prompt`` →
``context_block``. The loop never sources the block itself — that is
the services builder's job (``molexp.services.agent_context``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from molexp.agent.loops.interactive import InteractiveLoop, InteractiveLoopConfig
from molexp.agent.router import FinalChunk
from molexp.agent.runner import AgentRunner
from molexp.agent.session import Session
from molexp.agent.session_storage import InMemorySessionStorage
from molexp.agent.types import UsageBreakdown

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


async def _captured_system(config: InteractiveLoopConfig) -> str:
    router = _CapturingRouter()
    runner = AgentRunner(loop=InteractiveLoop(config=config), router=router)  # type: ignore[arg-type]
    session = Session(storage=InMemorySessionStorage(), session_id="ctx-block")
    async for _ in runner.run_events(session, "hello"):
        pass
    assert router.system is not None, "stream_agentic was never reached"
    return router.system


async def test_context_block_composed_after_base_prompt(tmp_path: Path) -> None:
    config = InteractiveLoopConfig(
        system_prompt=_BASE_PROMPT,
        workspace_root=tmp_path,
        context_block=_CONTEXT_BLOCK,
    )
    system = await _captured_system(config)
    assert _BASE_PROMPT in system
    assert _CONTEXT_BLOCK in system
    assert system.index(_BASE_PROMPT) < system.index(_CONTEXT_BLOCK)
    # ops preamble precedes user prompt
    assert "code_run" in system
    assert system.index("code_run") < system.index(_BASE_PROMPT)


async def test_empty_context_block_still_has_ops_preamble(tmp_path: Path) -> None:
    config = InteractiveLoopConfig(
        system_prompt=_BASE_PROMPT,
        workspace_root=tmp_path,
    )
    system = await _captured_system(config)
    assert _BASE_PROMPT in system
    assert "code_run" in system
    assert _CONTEXT_BLOCK not in system
