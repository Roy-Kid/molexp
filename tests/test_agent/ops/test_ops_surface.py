"""Agent ops surface — stable tool adapter + auto-discovery law.

Covers ``molexp.agent.ops.tools`` (the frozen stable tool-name set and the
working tool callables) and the auto-discovery law (no hard-coded MCP /
upstream tool tables) enforced across ``molexp.agent.ops``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from molexp.agent.execution_env import LocalExecutionEnv
from molexp.agent.ops import (
    OPS_TOOL_NAMES,
    build_ops_tools,
    build_session_context,
)
from molexp.agent.ops.preamble import DEFAULT_OPS_PREAMBLE


class TestBuildOpsTools:
    def test_exposes_only_the_frozen_stable_tool_names(self, tmp_path: Path) -> None:
        ctx = build_session_context(
            workspace_root=tmp_path,
            execution_env=LocalExecutionEnv(scratch_dir=tmp_path / "scratch"),
        )
        names = {t.__name__ for t in build_ops_tools(ctx)}
        assert names == OPS_TOOL_NAMES

    def test_workspace_ensure_and_code_run_wire_through_to_backends(self, tmp_path: Path) -> None:
        ctx = build_session_context(
            workspace_root=tmp_path,
            execution_env=LocalExecutionEnv(scratch_dir=tmp_path / "scratch"),
        )
        tools = {t.__name__: t for t in build_ops_tools(ctx)}
        assert tools["workspace_ensure"]("project", "demo-xx").startswith("ok")
        assert tools["workspace_ensure"]("experiment", "yy-scan", project="demo-xx").startswith(
            "ok"
        )
        assert (tmp_path / "projects" / "demo-xx").is_dir()
        wrote = tools["code_write"]("scripts/t.py", "print(1+1)\n")
        assert wrote.startswith("wrote")
        out = tools["code_run"](path="scripts/t.py")
        assert "exit_code=0" in out
        assert "2" in out


class TestAutoDiscoveryLaw:
    def test_preamble_names_stable_tools_not_hardcoded_mcp_tools(self) -> None:
        """Auto-discovery law: default preamble must not list concrete MCP tools."""
        banned = (
            "molexp_add_project",
            "molexp_materialize",
            "molmcp__",
            "write_file",
            "execute_python",
        )
        text = DEFAULT_OPS_PREAMBLE
        for token in banned:
            assert token not in text, f"preamble hard-codes {token!r}"
        assert "code_write" in text and "code_run" in text
        assert "discover" in text

    def test_ops_package_ships_no_upstream_symbol_table(self) -> None:
        """ops/ must not ship a static list of molpy/molpack APIs."""
        ops_dir = Path(__file__).resolve().parents[3] / "src" / "molexp" / "agent" / "ops"
        for path in ops_dir.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    v = node.value
                    # Allow prose mentions of package names, not qualname tables.
                    assert "molpy.compute" not in v
                    assert "molpack." not in v or "molpack" in path.name
