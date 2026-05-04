"""Phase 1a: ContextManager + PromptComposer determinism."""

from __future__ import annotations

import pytest

from molexp.agent import Message
from molexp.agent.context import (
    ContextBuildRequest,
    DefaultContextManager,
    PromptComposer,
    TailCompressor,
)


def test_prompt_composer_concatenates_layers_in_order() -> None:
    composer = PromptComposer()
    out = composer.compose(
        base="You are a research assistant.",
        workspace="Workspace lives at /tmp/lab.",
        skill="When in plan mode, never run tools.",
    )
    assert out.startswith("## Base\n")
    assert out.index("## Workspace") > out.index("## Base")
    assert out.index("## Skill") > out.index("## Workspace")


def test_prompt_composer_skips_empty_layers() -> None:
    composer = PromptComposer()
    out = composer.compose(base="base", workspace="", skill="")
    assert out == "## Base\nbase"


def test_prompt_composer_override_short_circuits() -> None:
    composer = PromptComposer()
    out = composer.compose(
        base="base",
        workspace="ws",
        skill="sk",
        override="custom prompt",
    )
    assert out == "custom prompt"


@pytest.mark.asyncio
async def test_default_context_manager_builds_packet() -> None:
    manager = DefaultContextManager()
    request = ContextBuildRequest(
        session_id="s",
        turn_id="t1",
        base_system="base",
        workspace_addendum="",
        skill_addendum="",
        instructions_override=None,
        history=(
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there"),
        ),
    )
    packet = await manager.build(request)
    assert "## Base" in packet.system
    assert len(packet.messages) == 2
    assert packet.budget.history_chars > 0


@pytest.mark.asyncio
async def test_tail_compressor_keeps_recent_messages() -> None:
    compressor = TailCompressor()
    history = tuple(
        Message(role="user", content=f"msg {i}" * 10) for i in range(50)
    )
    out = await compressor.compress(history, max_chars=200)
    # The compressor walks the tail; the result must be a suffix of the
    # original history with at most ~budget characters of content.
    assert len(out) < len(history)
    assert out[-1] == history[-1]
    assert sum(len(m.content) + len(m.role) for m in out) <= 400  # tolerance for role bytes


@pytest.mark.asyncio
async def test_tail_compressor_keeps_system_messages() -> None:
    compressor = TailCompressor()
    history = (
        Message(role="system", content="rules"),
        *(Message(role="user", content=f"msg {i}" * 50) for i in range(20)),
    )
    out = await compressor.compress(history, max_chars=100)
    assert any(m.role == "system" for m in out)
