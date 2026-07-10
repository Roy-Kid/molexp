"""Lifecycle tools are opt-in; default operation_mode does not strip write/exec."""

from __future__ import annotations

from pathlib import Path

from molexp.agent.loops.interactive.knowledge_tools import knowledge_tools
from molexp.agent.loops.interactive.lifecycle_tools import (
    LIFECYCLE_TOOL_NAMES,
    lifecycle_tools,
)
from molexp.agent.loops.interactive.loop import InteractiveLoopConfig
from molexp.agent.loops.interactive.tools import readonly_tools


def test_default_config_operation_mode_label() -> None:
    """Legacy default label is ``readonly`` but it is not a capability mask."""
    assert InteractiveLoopConfig().operation_mode == "readonly"


def test_default_tool_names_exclude_lifecycle(tmp_path: Path) -> None:
    names = {t.__name__ for t in readonly_tools(workspace_root=tmp_path)} | {
        t.__name__ for t in knowledge_tools(workspace_root=tmp_path)
    }
    assert names.isdisjoint(LIFECYCLE_TOOL_NAMES)


def test_lifecycle_mode_adds_workspace_tools(tmp_path: Path) -> None:
    tools = lifecycle_tools(workspace_root=tmp_path)
    names = {t.__name__ for t in tools}
    assert names == LIFECYCLE_TOOL_NAMES
    # Still no harness import from this module (docstring may mention it).
    import ast

    import molexp.agent.loops.interactive.lifecycle_tools as mod

    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("molexp.harness")
        ):
            raise AssertionError(f"illegal import {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("molexp.harness"):
                    raise AssertionError(f"illegal import {alias.name}")
