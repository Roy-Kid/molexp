"""``InteractiveLoopConfig.context_block`` → ``stream_agentic(system=…)`` (vision-loop-11, RED).

The agent layer stays mechanism-only: the loop composes
``system = system_prompt + "\\n\\n" + context_block`` when the block is
non-empty and leaves the system prompt untouched otherwise. It never sources
the block itself — that is the services builder's job
(``molexp.services.agent_context.build_mount_context``).

Spec: ``.claude/specs/vision-loop-11-mount-context.md`` (Design §2).
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
    assert system == f"{_BASE_PROMPT}\n\n{_CONTEXT_BLOCK}"


async def test_empty_context_block_leaves_system_unchanged(tmp_path: Path) -> None:
    config = InteractiveLoopConfig(
        system_prompt=_BASE_PROMPT,
        workspace_root=tmp_path,
    )
    system = await _captured_system(config)
    assert system == _BASE_PROMPT
