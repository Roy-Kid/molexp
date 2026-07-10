"""DEFAULT_CODE_LOOP_PREAMBLE composition (agent-code-loop-05-behavior)."""

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


class _CaptureSystemRouter:
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


@pytest.mark.asyncio
async def test_default_preamble_mentions_stable_ops_not_hardcoded_mcp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-001: system has stable ops tools; static preamble never hard-codes MCP names.

    Runtime MCP catalogs *may* inject live tool names into the system
    appendix (auto-discovery). Isolation: stub open_mcp_toolsets so this
    unit test does not depend on the operator's ~/.molexp MCP config.
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
    # Static ops preamble — no hard-coded third-party MCP tool names
    assert "code_run" in DEFAULT_CODE_LOOP_PREAMBLE
    assert "molexp_add_project" not in DEFAULT_CODE_LOOP_PREAMBLE
    assert "molexp_add_project" not in system  # isolated: no live MCP catalog


@pytest.mark.asyncio
async def test_operation_mode_readonly_still_mounts_code_tools(tmp_path: Path) -> None:
    """AC-002: operation_mode is not a capability mask."""
    router = _CaptureSystemRouter()
    loop = InteractiveLoop(
        config=InteractiveLoopConfig(workspace_root=tmp_path, operation_mode="readonly")
    )
    runner = AgentRunner(loop=loop, router=router)  # type: ignore[arg-type]
    session = Session(storage=InMemorySessionStorage(), session_id="mode")
    _ = [ev async for ev in runner.run_events(session, "hi")]

    names = {t.__name__ for t in router.tools}
    assert {"code_write", "code_run", "workspace_ensure", "discover"}.issubset(names)


@pytest.mark.asyncio
async def test_user_system_prompt_composes_with_preamble(tmp_path: Path) -> None:
    """AC-004: user system_prompt coexists with default preamble."""
    router = _CaptureSystemRouter()
    marker = "USER_CUSTOM_PROMPT_XYZ"
    loop = InteractiveLoop(
        config=InteractiveLoopConfig(
            workspace_root=tmp_path,
            system_prompt=marker,
        )
    )
    runner = AgentRunner(loop=loop, router=router)  # type: ignore[arg-type]
    session = Session(storage=InMemorySessionStorage(), session_id="compose")
    _ = [ev async for ev in runner.run_events(session, "hi")]

    assert "code_run" in router.system
    assert marker in router.system
    # preamble comes first
    assert router.system.index("code_run") < router.system.index(marker)


def test_config_docstring_describes_behavior_not_capability_mask() -> None:
    """AC-003: docs state operation_mode is not a capability mask."""
    doc = InteractiveLoopConfig.__doc__ or ""
    assert "capability mask" in doc.lower()
    assert "behavior" in doc.lower()
