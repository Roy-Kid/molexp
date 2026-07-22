"""InteractiveLoop system-prompt composition (agent-code-loop-05-behavior).

The system string handed to ``stream_agentic`` is composed from the stable
ops preamble (auto-discovery law: no hard-coded third-party MCP tool names)
plus the user's optional ``system_prompt``, in that order.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from molexp.agent.loops.interactive import InteractiveLoop, InteractiveLoopConfig
from molexp.agent.loops.interactive.loop import DEFAULT_CODE_LOOP_PREAMBLE
from molexp.agent.router import (
    AgenticChunk,
    FinalChunk,
    ModelTier,
    TextDeltaChunk,
)
from molexp.agent.runner import AgentRunner
from molexp.agent.session import Session
from molexp.agent.session_storage import InMemorySessionStorage
from molexp.agent.types import UsageBreakdown

pytestmark = pytest.mark.asyncio


class _CaptureSystemRouter:
    """Scripted router that records the ``system`` string it was handed."""

    def __init__(self) -> None:
        self.system = ""
        self.tools: tuple[Any, ...] = ()

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
        del prompt, toolsets, tier, message_history
        self.system = system
        self.tools = tools
        yield TextDeltaChunk(text="ok")
        yield FinalChunk(text="ok")

    async def complete_text(self, **_: object) -> object:
        raise AssertionError("unused")

    async def complete_structured(self, **_: object) -> object:
        raise AssertionError("unused")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


class TestInteractiveLoopSystemPrompt:
    async def test_preamble_carries_stable_ops_names_without_hardcoded_mcp(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The static preamble names stable ops tools but never hard-codes MCP names.

        Runtime MCP catalogs may inject live tool names into the system appendix
        (auto-discovery); the static preamble must not. Isolation: stub
        open_mcp_toolsets so this unit does not depend on the operator's
        ~/.molexp MCP config.
        """
        monkeypatch.setattr(
            "molexp.agent.loops.interactive.loop.open_mcp_toolsets",
            lambda _root: (),
        )
        router = _CaptureSystemRouter()
        loop = InteractiveLoop(config=InteractiveLoopConfig(workspace_root=tmp_path))
        runner = AgentRunner(loop=loop, router=router)  # type: ignore[arg-type]
        session = Session(storage=InMemorySessionStorage(), session_id="preamble")
        _ = [ev async for ev in runner.run_events(session, "hi")]

        system = router.system
        assert "code_run" in system
        assert "code_write" in system
        assert "discover" in system
        # Static ops preamble — no hard-coded third-party MCP tool names.
        assert "code_run" in DEFAULT_CODE_LOOP_PREAMBLE
        assert "molexp_add_project" not in DEFAULT_CODE_LOOP_PREAMBLE
        assert "molexp_add_project" not in system  # isolated: no live MCP catalog

    async def test_user_system_prompt_composes_after_preamble(self, tmp_path: Path) -> None:
        """A user ``system_prompt`` coexists with the default preamble, appended after it."""
        router = _CaptureSystemRouter()
        marker = "USER_CUSTOM_PROMPT_XYZ"
        loop = InteractiveLoop(
            config=InteractiveLoopConfig(workspace_root=tmp_path, system_prompt=marker)
        )
        runner = AgentRunner(loop=loop, router=router)  # type: ignore[arg-type]
        session = Session(storage=InMemorySessionStorage(), session_id="compose")
        _ = [ev async for ev in runner.run_events(session, "hi")]

        assert "code_run" in router.system
        assert marker in router.system
        assert router.system.index("code_run") < router.system.index(marker)
