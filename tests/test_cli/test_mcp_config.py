"""Tests for ``molexp mcp ...`` scope resolution — the pure-unit slice.

MCP-config file management is CLI-owned (there is no lower layer). These
tests cover ``_resolve_mcp_path`` — the scope → path mapping (user →
``~/.claude.json``, project → ``./.mcp.json``, unknown → rejected) —
called directly, without booting the CLI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer


class TestScope:
    def test_user_scope_resolves_to_claude_json(self):
        from molexp.cli.workspace.mcp_config import _resolve_mcp_path

        path = _resolve_mcp_path("user", None)
        assert path.name == ".claude.json"
        assert path.parent == Path.home()

    def test_project_scope_resolves_to_cwd_mcp_json(self, tmp_path, monkeypatch):
        from molexp.cli.workspace.mcp_config import _resolve_mcp_path

        monkeypatch.chdir(tmp_path)
        path = _resolve_mcp_path("project", None)
        assert path.name == ".mcp.json"
        assert path.parent == tmp_path.resolve()

    def test_unknown_scope_rejected(self):
        from molexp.cli.workspace.mcp_config import _resolve_mcp_path

        with pytest.raises(typer.BadParameter):
            _resolve_mcp_path("galaxy", None)
