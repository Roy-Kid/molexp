"""Tests for ``molexp mcp ...`` — the MCP-server registry subcommand.

Every test passes ``--config`` so we never touch the user's real
``~/.claude.json`` or any project-level ``.mcp.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from molexp.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    """Strip ANSI colour/format codes — Rich's option highlighter splits
    ``--env`` into ``-``/``-env`` segments with codes in between, which
    defeats naive substring matches on terminal-enabled CI runs."""
    return _ANSI_RE.sub("", text)


@pytest.fixture
def cfg(tmp_path: Path) -> Path:
    """Path to an isolated registry file (does not exist yet)."""
    return tmp_path / "mcp.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


class TestAdd:
    def test_stdio_default(self, cfg):
        result = runner.invoke(
            app,
            ["mcp", "add", "molmcp", "molmcp", "--config", str(cfg)],
        )
        assert result.exit_code == 0, result.output
        assert _read(cfg)["mcpServers"]["molmcp"] == {
            "type": "stdio",
            "command": "molmcp",
            "args": [],
            "env": {},
        }

    def test_duplicate_without_force_fails(self, cfg):
        runner.invoke(app, ["mcp", "add", "n", "cmd", "--config", str(cfg)])
        result = runner.invoke(app, ["mcp", "add", "n", "other", "--config", str(cfg)])
        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_duplicate_with_force_overwrites(self, cfg):
        runner.invoke(app, ["mcp", "add", "n", "cmd", "--config", str(cfg)])
        result = runner.invoke(
            app,
            ["mcp", "add", "n", "other", "-f", "--config", str(cfg)],
        )
        assert result.exit_code == 0, result.output
        assert _read(cfg)["mcpServers"]["n"]["command"] == "other"

    def test_http_rejects_env(self, cfg):
        result = runner.invoke(
            app,
            ["mcp", "add", "h", "https://x", "-t", "http", "-e", "A=B", "--config", str(cfg)],
        )
        assert result.exit_code != 0
        assert "stdio" in result.output

    def test_stdio_rejects_header(self, cfg):
        result = runner.invoke(
            app,
            ["mcp", "add", "s", "cmd", "-H", "X: 1", "--config", str(cfg)],
        )
        assert result.exit_code != 0
        assert "http" in result.output or "sse" in result.output

    def test_unknown_transport(self, cfg):
        result = runner.invoke(
            app,
            ["mcp", "add", "x", "cmd", "-t", "smoke", "--config", str(cfg)],
        )
        assert result.exit_code != 0
        assert "transport" in result.output.lower()


# ---------------------------------------------------------------------------
# add-json
# ---------------------------------------------------------------------------


class TestAddJson:
    def test_stores_arbitrary_shape(self, cfg):
        payload = json.dumps(
            {
                "type": "stdio",
                "command": "molmcp",
                "args": [],
                "env": {},
                "usage_instructions": "use it",
            }
        )
        result = runner.invoke(
            app,
            ["mcp", "add-json", "molmcp", payload, "--config", str(cfg)],
        )
        assert result.exit_code == 0, result.output
        entry = _read(cfg)["mcpServers"]["molmcp"]
        assert entry["usage_instructions"] == "use it"

    def test_invalid_json(self, cfg):
        result = runner.invoke(
            app,
            ["mcp", "add-json", "x", "{not json", "--config", str(cfg)],
        )
        assert result.exit_code == 1
        assert "Invalid JSON" in result.output


# ---------------------------------------------------------------------------
# get / list / remove
# ---------------------------------------------------------------------------


class TestGet:
    def test_existing(self, cfg):
        runner.invoke(app, ["mcp", "add", "n", "cmd", "--config", str(cfg)])
        result = runner.invoke(app, ["mcp", "get", "n", "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "stdio" in result.output

    def test_missing(self, cfg):
        result = runner.invoke(app, ["mcp", "get", "ghost", "--config", str(cfg)])
        assert result.exit_code == 1


class TestList:
    def test_two_servers(self, cfg):
        runner.invoke(app, ["mcp", "add", "a", "cmd-a", "--config", str(cfg)])
        runner.invoke(
            app,
            ["mcp", "add", "b", "https://x", "-t", "http", "--config", str(cfg)],
        )
        result = runner.invoke(app, ["mcp", "list", "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "a" in result.output
        assert "b" in result.output
        assert "stdio" in result.output
        assert "http" in result.output


class TestRemove:
    def test_existing(self, cfg):
        runner.invoke(app, ["mcp", "add", "n", "cmd", "--config", str(cfg)])
        result = runner.invoke(app, ["mcp", "remove", "n", "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        assert _read(cfg)["mcpServers"] == {}

    def test_missing(self, cfg):
        result = runner.invoke(app, ["mcp", "remove", "ghost", "--config", str(cfg)])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Schema preservation
# ---------------------------------------------------------------------------


def test_unknown_toplevel_keys_round_trip(cfg):
    """Top-level keys outside `mcpServers` (e.g. claude's settings) must survive add/remove."""
    cfg.write_text(
        json.dumps(
            {
                "_comment": "hand-written file",
                "version": 1,
                "mcpServers": {"old": {"type": "stdio", "command": "x"}},
                "projects": {"/some/path": {"some": "claude-only key"}},
            }
        )
    )
    result = runner.invoke(app, ["mcp", "add", "new", "cmd", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    data = _read(cfg)
    assert data["_comment"] == "hand-written file"
    assert data["version"] == 1
    assert data["projects"] == {"/some/path": {"some": "claude-only key"}}
    assert set(data["mcpServers"]) == {"old", "new"}


# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# import / export
# ---------------------------------------------------------------------------


class TestImport:
    def test_skips_existing_without_force(self, cfg, tmp_path):
        src = tmp_path / "src.json"
        src.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "shared": {"type": "stdio", "command": "from-src"},
                        "newone": {"type": "stdio", "command": "src-only"},
                    }
                }
            )
        )
        runner.invoke(
            app,
            ["mcp", "add", "shared", "from-dst", "--config", str(cfg)],
        )
        result = runner.invoke(app, ["mcp", "import", str(src), "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        servers = _read(cfg)["mcpServers"]
        assert servers["shared"]["command"] == "from-dst"  # kept
        assert servers["newone"]["command"] == "src-only"  # added
        assert "skipped" in result.output
        assert "shared" in result.output
        assert "added:       newone" in result.output

    def test_force_overwrites(self, cfg, tmp_path):
        src = tmp_path / "src.json"
        src.write_text(
            json.dumps({"mcpServers": {"shared": {"type": "stdio", "command": "from-src"}}})
        )
        runner.invoke(
            app,
            ["mcp", "add", "shared", "from-dst", "--config", str(cfg)],
        )
        result = runner.invoke(
            app,
            ["mcp", "import", str(src), "-f", "--config", str(cfg)],
        )
        assert result.exit_code == 0, result.output
        assert _read(cfg)["mcpServers"]["shared"]["command"] == "from-src"

    def test_missing_source(self, cfg, tmp_path):
        src = tmp_path / "nope.json"
        result = runner.invoke(app, ["mcp", "import", str(src), "--config", str(cfg)])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_preserves_destination_unrelated_keys(self, cfg, tmp_path):
        cfg.write_text(
            json.dumps(
                {
                    "claude_setting": "keep me",
                    "mcpServers": {},
                }
            )
        )
        src = tmp_path / "src.json"
        src.write_text(json.dumps({"mcpServers": {"a": {"type": "stdio", "command": "x"}}}))
        runner.invoke(app, ["mcp", "import", str(src), "--config", str(cfg)])
        data = _read(cfg)
        assert data["claude_setting"] == "keep me"
        assert "a" in data["mcpServers"]


class TestExport:
    def test_copies_to_destination(self, cfg, tmp_path):
        runner.invoke(app, ["mcp", "add", "a", "cmd", "--config", str(cfg)])
        dst = tmp_path / "dst.json"
        result = runner.invoke(app, ["mcp", "export", str(dst), "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        assert _read(dst)["mcpServers"]["a"]["command"] == "cmd"

    def test_skips_existing_in_destination(self, cfg, tmp_path):
        runner.invoke(app, ["mcp", "add", "n", "from-src", "--config", str(cfg)])
        dst = tmp_path / "dst.json"
        dst.write_text(json.dumps({"mcpServers": {"n": {"type": "stdio", "command": "from-dst"}}}))
        result = runner.invoke(app, ["mcp", "export", str(dst), "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        assert _read(dst)["mcpServers"]["n"]["command"] == "from-dst"
        assert "skipped" in result.output

    def test_creates_parent_dirs(self, cfg, tmp_path):
        runner.invoke(app, ["mcp", "add", "a", "cmd", "--config", str(cfg)])
        dst = tmp_path / "deep" / "nested" / "dst.json"
        result = runner.invoke(app, ["mcp", "export", str(dst), "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        assert dst.exists()


# typer is referenced by TestScope.test_unknown_scope_rejected
import typer  # noqa: E402
