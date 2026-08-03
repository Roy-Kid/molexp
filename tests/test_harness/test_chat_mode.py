"""ChatMode — peer of Plan; InteractiveLoop, scratch-only, no default land."""

from __future__ import annotations

from pathlib import Path

from molexp.harness import ChatMode, chat_loop_config
from molexp.harness.modes.chat import CHAT_SCRATCH_PREFIX


def test_chat_loop_config_is_chat_surface() -> None:
    cfg = chat_loop_config(workspace_root=Path("/tmp/ws"))
    assert cfg.operation_mode == "chat"
    assert cfg.workspace_root == Path("/tmp/ws")


def test_chat_mode_build_loop_uses_interactive_chat() -> None:
    mode = ChatMode()
    assert mode.name == "chat"
    loop = mode.build_loop(workspace_root=Path("/tmp/ws"))
    assert loop.config.operation_mode == "chat"
    assert loop.name == "agent"


def test_chat_scratch_prefix() -> None:
    assert CHAT_SCRATCH_PREFIX == "agent/.scratch"
