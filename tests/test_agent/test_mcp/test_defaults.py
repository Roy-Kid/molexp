"""Unit tests for ``molexp.agent.mcp.defaults`` (spec ``molmcp-agent-default``)."""

from __future__ import annotations

import json
import shutil
import stat
import subprocess
import sys

import pytest

from molexp.agent.mcp import defaults as defaults_mod
from molexp.agent.mcp import store as mcp_mod
from molexp.agent.mcp.defaults import (
    MCP_DEFAULTS,
    MCP_SEEDED_FILENAME,
    MOLMCP_USAGE_INSTRUCTIONS,
    seed_user_defaults,
)
from molexp.agent.mcp.store import MCP_CONFIG_FILENAME, McpScope, McpStore

# ── ac-004: MCP_DEFAULTS shape ────────────────────────────────────────────


@pytest.mark.unit
def test_mcp_defaults_shape():
    """ac-004 — exactly one entry, ``molmcp``, with the documented shape."""
    assert len(MCP_DEFAULTS) == 1, MCP_DEFAULTS
    name, spec = MCP_DEFAULTS[0]
    assert name == "molmcp"
    assert spec["type"] == "stdio"
    assert spec["command"] == "molmcp"
    assert spec["args"] == []
    usage = spec["usage_instructions"]
    assert isinstance(usage, str) and usage
    assert "molmcp__" in usage
    assert "molcrafts" in usage
    assert any(token in usage for token in ("molpy", "molpack", "molrs", "lammps", "molexp"))
    # After the workspace-as-tool-parameter refactor, the prompt must
    # tell the LLM how to fill the `workspace` argument.
    assert "workspace" in usage.lower()


# ── Contract: the seeded command + args must be invocable ────────────────


@pytest.mark.unit
@pytest.mark.skipif(
    shutil.which("molmcp") is None,
    reason="molmcp not installed on PATH; cannot validate the contract",
)
def test_seeded_molmcp_command_is_invocable():
    """The exact ``command + args`` we seed must be accepted by ``molmcp``.

    Regression guard: the seeded ``command + args`` need to drive the
    ``molmcp`` CLI cleanly through ``--help``. Originally caught a
    drift where ``defaults.py`` shipped ``("gateway",)`` while the
    ``molmcp`` CLI no longer had a ``gateway`` subcommand;
    ``test_mcp_defaults_shape`` only asserted the constant equals
    itself — this test asserts the constant is *usable*.
    """
    name, spec = MCP_DEFAULTS[0]
    assert name == "molmcp"
    cmd = [spec["command"], *spec["args"], "--help"]
    result = subprocess.run(cmd, capture_output=True, timeout=15, text=True)
    assert result.returncode == 0, (
        f"`{' '.join(cmd)}` failed: rc={result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


# ── ac-005: fresh-tempdir seeding ─────────────────────────────────────────


@pytest.mark.unit
def test_seed_writes_fresh_user_config(tmp_path):
    """ac-005 — seeding writes both ``mcp.json`` and the sentinel."""
    user_dir = tmp_path / "home" / ".molexp"
    config = user_dir / MCP_CONFIG_FILENAME
    sentinel = user_dir / MCP_SEEDED_FILENAME
    assert not config.exists()
    assert not sentinel.exists()

    changed = seed_user_defaults(config, sentinel)
    assert changed is True
    assert config.exists()
    assert sentinel.exists()

    data = json.loads(config.read_text())
    assert "molmcp" in data["mcpServers"]
    assert data["mcpServers"]["molmcp"]["command"] == "molmcp"
    assert data["mcpServers"]["molmcp"]["args"] == []
    assert data["mcpServers"]["molmcp"]["usage_instructions"] == MOLMCP_USAGE_INSTRUCTIONS

    sentinel_data = json.loads(sentinel.read_text())
    assert "molmcp" in sentinel_data["seeded"]


# ── ac-006: idempotency ───────────────────────────────────────────────────


@pytest.mark.unit
def test_seed_idempotent(tmp_path):
    """ac-006 — second call writes nothing; mtime unchanged."""
    user_dir = tmp_path / "home" / ".molexp"
    config = user_dir / MCP_CONFIG_FILENAME
    sentinel = user_dir / MCP_SEEDED_FILENAME

    assert seed_user_defaults(config, sentinel) is True
    mtime_before = config.stat().st_mtime_ns
    sentinel_mtime_before = sentinel.stat().st_mtime_ns

    assert seed_user_defaults(config, sentinel) is False
    assert config.stat().st_mtime_ns == mtime_before
    assert sentinel.stat().st_mtime_ns == sentinel_mtime_before


# ── ac-007: env override ──────────────────────────────────────────────────


@pytest.mark.unit
def test_seed_honors_command_env_override(tmp_path, monkeypatch):
    """ac-007 — ``MOLEXP_MOLMCP_COMMAND="echo hi"`` rewrites command/args."""
    monkeypatch.setenv("MOLEXP_MOLMCP_COMMAND", "echo hi")
    user_dir = tmp_path / "home" / ".molexp"
    config = user_dir / MCP_CONFIG_FILENAME
    sentinel = user_dir / MCP_SEEDED_FILENAME

    assert seed_user_defaults(config, sentinel) is True

    data = json.loads(config.read_text())
    assert data["mcpServers"]["molmcp"]["command"] == "echo"
    assert data["mcpServers"]["molmcp"]["args"] == ["hi"]


@pytest.mark.unit
def test_seed_default_command_when_env_unset(tmp_path, monkeypatch):
    """ac-007 (companion) — env unset → documented default."""
    monkeypatch.delenv("MOLEXP_MOLMCP_COMMAND", raising=False)
    user_dir = tmp_path / "home" / ".molexp"
    config = user_dir / MCP_CONFIG_FILENAME
    sentinel = user_dir / MCP_SEEDED_FILENAME

    assert seed_user_defaults(config, sentinel) is True

    data = json.loads(config.read_text())
    assert data["mcpServers"]["molmcp"]["command"] == "molmcp"
    assert data["mcpServers"]["molmcp"]["args"] == []


# ── ac-008: disable-by-deletion ───────────────────────────────────────────


@pytest.mark.unit
def test_seed_respects_user_deletion(tmp_path):
    """ac-008 — pre-seeded sentinel + missing molmcp entry → no re-seed."""
    user_dir = tmp_path / "home" / ".molexp"
    user_dir.mkdir(parents=True)
    config = user_dir / MCP_CONFIG_FILENAME
    sentinel = user_dir / MCP_SEEDED_FILENAME

    sentinel.write_text(json.dumps({"seeded": ["molmcp"]}))
    config.write_text(json.dumps({"mcpServers": {"github": {"type": "stdio", "command": "x"}}}))

    config_mtime_before = config.stat().st_mtime_ns
    changed = seed_user_defaults(config, sentinel)
    assert changed is False

    data = json.loads(config.read_text())
    assert "molmcp" not in data["mcpServers"]
    assert data["mcpServers"]["github"]["command"] == "x"
    assert config.stat().st_mtime_ns == config_mtime_before


# ── ac-009: McpStore.__init__ triggers seeding ─────────────────────────────


@pytest.mark.unit
def test_store_init_triggers_seed(tmp_path, monkeypatch):
    """ac-009 — constructing ``McpStore`` seeds molmcp under the User dir."""
    fake_home = tmp_path / "home" / ".molexp"
    monkeypatch.setattr(mcp_mod, "USER_DIR", fake_home)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = McpStore(workspace)
    rows = store.list()
    by_name = {row.name: row for row in rows}
    assert "molmcp" in by_name
    entry = by_name["molmcp"]
    assert entry.scope is McpScope.USER
    assert entry.usage_instructions == MOLMCP_USAGE_INSTRUCTIONS


# ── ac-010: read-only HOME is graceful ────────────────────────────────────


@pytest.mark.unit
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission semantics")
def test_seed_read_only_home_warns(tmp_path, monkeypatch):
    """ac-010 — read-only User dir: warn, do not raise; ``False`` returned."""
    parent = tmp_path / "home"
    parent.mkdir()
    user_dir = parent / ".molexp"
    user_dir.mkdir()
    config = user_dir / MCP_CONFIG_FILENAME
    sentinel = user_dir / MCP_SEEDED_FILENAME

    warnings: list[str] = []
    monkeypatch.setattr(
        defaults_mod._LOG, "warning", lambda msg, *_a, **_kw: warnings.append(str(msg))
    )

    user_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # 0500 — read+execute, no write
    try:
        changed = seed_user_defaults(config, sentinel)
        assert changed is False
        assert any("seed" in w.lower() for w in warnings), warnings
    finally:
        user_dir.chmod(stat.S_IRWXU)

    assert not config.exists()


@pytest.mark.unit
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission semantics")
def test_store_list_survives_read_only_home(tmp_path, monkeypatch):
    """ac-010 (companion) — ``McpStore.list()`` returns ``[]`` on read-only HOME."""
    parent = tmp_path / "home"
    parent.mkdir()
    fake_home = parent / ".molexp"
    fake_home.mkdir()
    monkeypatch.setattr(mcp_mod, "USER_DIR", fake_home)

    fake_home.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        store = McpStore(workspace)
        assert store.list() == []
    finally:
        fake_home.chmod(stat.S_IRWXU)


# ── Round-trip safety: seeded entry survives McpStore.list() ──────────────


@pytest.mark.unit
def test_seeded_entry_round_trips_via_list(tmp_path, monkeypatch):
    """The molmcp entry pulled back through ``list()`` matches what was seeded."""
    fake_home = tmp_path / "home" / ".molexp"
    monkeypatch.setattr(mcp_mod, "USER_DIR", fake_home)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = McpStore(workspace)
    rows = [r for r in store.list() if r.name == "molmcp"]
    assert len(rows) == 1
    entry = rows[0]
    assert entry.transport == "stdio"
    assert entry.command == "molmcp"
    assert entry.args == ()
    assert entry.usage_instructions == MOLMCP_USAGE_INSTRUCTIONS
    assert entry.valid is True


# ── Module-level invariants ───────────────────────────────────────────────


@pytest.mark.unit
def test_module_exposes_documented_names():
    """The defaults module's public API is what the spec promises."""
    for name in (
        "MCP_DEFAULTS",
        "MOLMCP_USAGE_INSTRUCTIONS",
        "MCP_SEEDED_FILENAME",
        "seed_user_defaults",
    ):
        assert hasattr(defaults_mod, name), name
