"""Unit tests for ``molexp.agent.mcp.store`` — multi-scope MCP store + secrets store."""

from __future__ import annotations

import json
import sys

import pytest

from molexp.agent.mcp import defaults as defaults_mod
from molexp.agent.mcp import store as mcp_mod
from molexp.agent.mcp.store import (
    MCP_CONFIG_FILENAME,
    McpScope,
    McpSecretsStore,
    McpStore,
    UnresolvedSecretError,
)


@pytest.fixture
def isolated_user_dir(tmp_path, monkeypatch):
    """Redirect ``USER_DIR`` to a temp dir so tests never touch ``~/.molexp``.

    Also stubs out platform-default seeding (``molmcp``) so these tests observe
    an empty User config; the seeding contract owns its coverage in
    :mod:`tests.test_agent.test_mcp.test_defaults`.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    fake_user_dir = fake_home / ".molexp"
    monkeypatch.setattr(mcp_mod, "USER_DIR", fake_user_dir)
    monkeypatch.setattr(defaults_mod, "seed_user_defaults", lambda *a, **kw: False)  # noqa: ARG005
    return fake_user_dir


@pytest.fixture
def store(tmp_path, isolated_user_dir):
    """Fresh ``McpStore`` rooted at a temp workspace dir."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return McpStore(workspace)


class TestMcpSecretsStore:
    """File-backed KV for MCP secrets (``${SECRET:K}`` targets)."""

    @pytest.mark.unit
    def test_set_round_trips_through_get(self, tmp_path):
        s = McpSecretsStore(tmp_path / ".mcp_secrets.json")
        s.set("GITHUB_TOKEN", "ghp_abc123")
        assert s.get("GITHUB_TOKEN") == "ghp_abc123"
        assert s.list_keys() == ["GITHUB_TOKEN"]

    @pytest.mark.unit
    def test_empty_value_deletes_key(self, tmp_path):
        """Delete-by-clear UX: setting an empty string removes the key."""
        s = McpSecretsStore(tmp_path / ".mcp_secrets.json")
        s.set("FOO", "bar")
        s.set("FOO", "")
        assert s.get("FOO") is None
        assert s.list_keys() == []

    @pytest.mark.unit
    def test_delete_reports_whether_key_existed(self, tmp_path):
        s = McpSecretsStore(tmp_path / ".mcp_secrets.json")
        assert s.delete("MISSING") is False
        s.set("FOO", "bar")
        assert s.delete("FOO") is True
        assert s.get("FOO") is None

    @pytest.mark.unit
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission semantics")
    def test_file_chmodded_to_owner_only(self, tmp_path):
        """Secrets must not leak to other local users: file mode is 0o600."""
        path = tmp_path / ".mcp_secrets.json"
        McpSecretsStore(path).set("FOO", "bar")
        assert path.stat().st_mode & 0o777 == 0o600

    @pytest.mark.unit
    def test_corrupt_file_returns_empty(self, tmp_path):
        path = tmp_path / ".mcp_secrets.json"
        path.write_text("{not json")
        s = McpSecretsStore(path)
        assert s.list_keys() == []
        assert s.get("FOO") is None


class TestMcpStore:
    """Two-tier (User + Workspace) MCP config store with secret separation."""

    # ── list / shadowing ────────────────────────────────────────────────────

    @pytest.mark.unit
    def test_workspace_shadows_user_when_same_name(self, store):
        """A Workspace entry fully overrides a same-named User entry."""
        store.upsert(McpScope.USER, "github", {"type": "http", "url": "https://api.example/mcp"})
        store.upsert(
            McpScope.WORKSPACE,
            "github",
            {"type": "http", "url": "https://workspace.example/mcp"},
        )
        by_scope = {r.scope: r for r in store.list()}
        assert by_scope[McpScope.USER].shadowed is True
        assert by_scope[McpScope.WORKSPACE].shadowed is False
        assert by_scope[McpScope.WORKSPACE].url == "https://workspace.example/mcp"

    # ── upsert validation ───────────────────────────────────────────────────

    @pytest.mark.unit
    def test_upsert_rejects_invalid_name(self, store):
        with pytest.raises(ValueError, match="Invalid server name"):
            store.upsert(McpScope.WORKSPACE, "Bad Name!", {"type": "stdio", "command": "x"})

    @pytest.mark.unit
    def test_upsert_rejects_missing_type_discriminator(self, store):
        with pytest.raises(ValueError, match="Invalid spec"):
            store.upsert(McpScope.WORKSPACE, "x", {"command": "x"})

    @pytest.mark.unit
    def test_upsert_replaces_full_entry(self, store):
        """Replacement is whole-entry — no per-field merge of the prior spec."""
        store.upsert(
            McpScope.WORKSPACE,
            "x",
            {"type": "stdio", "command": "old", "args": ["a"], "env": {"K": "v"}},
        )
        store.upsert(McpScope.WORKSPACE, "x", {"type": "stdio", "command": "new"})
        entry = store.get(McpScope.WORKSPACE, "x")
        assert entry is not None
        assert entry.command == "new"
        assert entry.args == ()
        assert entry.env_keys == ()

    @pytest.mark.unit
    def test_upsert_writes_to_correct_scope(self, store):
        store.upsert(McpScope.USER, "u", {"type": "stdio", "command": "x"})
        store.upsert(McpScope.WORKSPACE, "w", {"type": "stdio", "command": "x"})
        user_data = json.loads(store.config_path(McpScope.USER).read_text())
        ws_data = json.loads(store.config_path(McpScope.WORKSPACE).read_text())
        assert "u" in user_data["mcpServers"]
        assert "u" not in ws_data["mcpServers"]
        assert "w" in ws_data["mcpServers"]
        assert "w" not in user_data["mcpServers"]

    # ── delete ──────────────────────────────────────────────────────────────

    @pytest.mark.unit
    def test_delete_only_affects_target_scope(self, store):
        store.upsert(McpScope.USER, "x", {"type": "stdio", "command": "y"})
        store.upsert(McpScope.WORKSPACE, "x", {"type": "stdio", "command": "z"})
        assert store.delete(McpScope.WORKSPACE, "x") is True
        rows = store.list()
        assert len(rows) == 1
        assert rows[0].scope is McpScope.USER
        assert rows[0].shadowed is False

    # ── secret references + resolution ──────────────────────────────────────

    @pytest.mark.unit
    def test_secret_refs_detected_from_env_values(self, store):
        store.upsert(
            McpScope.WORKSPACE,
            "gh",
            {"type": "stdio", "command": "x", "env": {"TOKEN": "${SECRET:GITHUB_TOKEN}"}},
        )
        entry = store.get(McpScope.WORKSPACE, "gh")
        assert entry is not None
        assert entry.secret_refs == ("GITHUB_TOKEN",)
        assert entry.unresolved_secrets == ("GITHUB_TOKEN",)

    @pytest.mark.unit
    def test_user_secret_satisfies_workspace_entry(self, store):
        """One shared keyring per user: a User secret covers a Workspace entry."""
        store.upsert(
            McpScope.WORKSPACE,
            "gh",
            {
                "type": "http",
                "url": "https://gh/mcp",
                "headers": {"Authorization": "Bearer ${SECRET:GH}"},
            },
        )
        store.secrets(McpScope.USER).set("GH", "ghp_user")
        entry = store.get(McpScope.WORKSPACE, "gh")
        assert entry is not None
        assert entry.unresolved_secrets == ()

    @pytest.mark.unit
    def test_workspace_secret_takes_precedence_over_user_secret(self, store):
        store.upsert(
            McpScope.WORKSPACE,
            "gh",
            {"type": "stdio", "command": "x", "env": {"TOKEN": "${SECRET:GH}"}},
        )
        store.secrets(McpScope.USER).set("GH", "user-value")
        store.secrets(McpScope.WORKSPACE).set("GH", "workspace-value")
        entry = store.get(McpScope.WORKSPACE, "gh")
        assert entry is not None
        assert store.resolve(entry).env["TOKEN"] == "workspace-value"

    @pytest.mark.unit
    def test_resolve_substitutes_secrets_in_env(self, store):
        store.upsert(
            McpScope.WORKSPACE,
            "gh",
            {
                "type": "stdio",
                "command": "gh-mcp",
                "args": ["--server"],
                "env": {"GITHUB_TOKEN": "${SECRET:GH_TOKEN}", "STATIC": "literal-value"},
            },
        )
        store.secrets(McpScope.WORKSPACE).set("GH_TOKEN", "ghp_real")
        entry = store.get(McpScope.WORKSPACE, "gh")
        assert entry is not None
        resolved = store.resolve(entry)
        assert resolved.transport == "stdio"
        assert resolved.command == "gh-mcp"
        assert resolved.args == ("--server",)
        assert resolved.env == {"GITHUB_TOKEN": "ghp_real", "STATIC": "literal-value"}
        assert resolved.headers == {}

    @pytest.mark.unit
    def test_resolve_substitutes_secrets_in_headers(self, store):
        store.upsert(
            McpScope.WORKSPACE,
            "gh",
            {
                "type": "http",
                "url": "https://gh.example/mcp",
                "headers": {"Authorization": "Bearer ${SECRET:T}"},
            },
        )
        store.secrets(McpScope.WORKSPACE).set("T", "tok-123")
        entry = store.get(McpScope.WORKSPACE, "gh")
        assert entry is not None
        resolved = store.resolve(entry)
        assert resolved.transport == "http"
        assert resolved.url == "https://gh.example/mcp"
        assert resolved.headers == {"Authorization": "Bearer tok-123"}

    @pytest.mark.unit
    def test_no_fallback_to_os_environ(self, store, monkeypatch):
        """Critical invariant: env-var fallback was removed. A secret present
        only in ``os.environ`` stays unresolved, and ``resolve`` raises with the
        missing key carried in the error payload."""
        monkeypatch.setenv("GITHUB_TOKEN", "from-env")
        store.upsert(
            McpScope.WORKSPACE,
            "gh",
            {"type": "stdio", "command": "x", "env": {"T": "${SECRET:GITHUB_TOKEN}"}},
        )
        entry = store.get(McpScope.WORKSPACE, "gh")
        assert entry is not None
        assert entry.unresolved_secrets == ("GITHUB_TOKEN",)
        with pytest.raises(UnresolvedSecretError) as exc:
            store.resolve(entry)
        assert exc.value.keys == ["GITHUB_TOKEN"]

    @pytest.mark.unit
    def test_secret_references_groups_server_names_by_key(self, store):
        store.upsert(
            McpScope.WORKSPACE,
            "gh1",
            {"type": "stdio", "command": "x", "env": {"T": "${SECRET:GH}"}},
        )
        store.upsert(
            McpScope.WORKSPACE,
            "gh2",
            {"type": "http", "url": "https://x/mcp", "headers": {"H": "Bearer ${SECRET:GH}"}},
        )
        assert store.secret_references(McpScope.WORKSPACE) == {"GH": ["gh1", "gh2"]}

    # ── invalid entries ─────────────────────────────────────────────────────

    @pytest.mark.unit
    def test_entry_without_type_marked_invalid(self, tmp_path, isolated_user_dir):
        """A legacy entry missing the ``type`` discriminator surfaces as invalid."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / MCP_CONFIG_FILENAME).write_text(
            json.dumps({"mcpServers": {"legacy": {"command": "x"}}})
        )
        rows = McpStore(workspace).list()
        assert len(rows) == 1
        assert rows[0].valid is False
        assert (
            "type" in rows[0].invalid_reason.lower()
            or "discriminator" in rows[0].invalid_reason.lower()
        )

    @pytest.mark.unit
    def test_corrupt_config_returns_empty(self, tmp_path, isolated_user_dir):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / MCP_CONFIG_FILENAME).write_text("{not json")
        assert McpStore(workspace).list() == []

    # ── usage_instructions (molmcp-agent-default ac-003) ────────────────────

    @pytest.mark.unit
    def test_entry_surfaces_usage_instructions(self, store):
        """ac-003 — the on-disk ``usage_instructions`` round-trips onto the entry."""
        store.upsert(
            McpScope.USER,
            "x",
            {"type": "stdio", "command": "x", "usage_instructions": "DOC"},
        )
        entry = store.get(McpScope.USER, "x")
        assert entry is not None
        assert entry.usage_instructions == "DOC"
