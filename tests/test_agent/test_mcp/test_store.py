"""Unit tests for the multi-scope MCP store + secrets store."""

from __future__ import annotations

import json
import sys

import pytest

from molexp.agent.mcp import defaults as defaults_mod
from molexp.agent.mcp import store as mcp_mod
from molexp.agent.mcp.store import (
    _SPEC_ADAPTER,
    MCP_CONFIG_FILENAME,
    HttpSpec,
    McpScope,
    McpSecretsStore,
    McpStore,
    StdioSpec,
    UnresolvedSecretError,
)


@pytest.fixture
def isolated_user_dir(tmp_path, monkeypatch):
    """Redirect ``USER_DIR`` to a temp dir so tests never touch ``~/.molexp``.

    Also stubs out platform-default seeding (``molmcp``) so tests in this
    module — which predate the seeding feature — observe an empty User
    config. The seeding contract has its own coverage in
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


# ── Secrets store ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_secrets_get_returns_none_when_missing(tmp_path):
    s = McpSecretsStore(tmp_path / ".mcp_secrets.json")
    assert s.get("FOO") is None
    assert s.list_keys() == []


@pytest.mark.unit
def test_secrets_set_round_trips_through_get(tmp_path):
    s = McpSecretsStore(tmp_path / ".mcp_secrets.json")
    s.set("GITHUB_TOKEN", "ghp_abc123")
    assert s.get("GITHUB_TOKEN") == "ghp_abc123"
    assert s.list_keys() == ["GITHUB_TOKEN"]


@pytest.mark.unit
def test_secrets_empty_value_deletes_key(tmp_path):
    s = McpSecretsStore(tmp_path / ".mcp_secrets.json")
    s.set("FOO", "bar")
    s.set("FOO", "")
    assert s.get("FOO") is None
    assert s.list_keys() == []


@pytest.mark.unit
def test_secrets_delete_returns_false_when_missing(tmp_path):
    s = McpSecretsStore(tmp_path / ".mcp_secrets.json")
    assert s.delete("MISSING") is False
    s.set("FOO", "bar")
    assert s.delete("FOO") is True
    assert s.get("FOO") is None


@pytest.mark.unit
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission semantics")
def test_secrets_file_chmodded_to_owner_only(tmp_path):
    path = tmp_path / ".mcp_secrets.json"
    s = McpSecretsStore(path)
    s.set("FOO", "bar")
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600


@pytest.mark.unit
def test_secrets_corrupt_file_returns_empty(tmp_path):
    path = tmp_path / ".mcp_secrets.json"
    path.write_text("{not json")
    s = McpSecretsStore(path)
    assert s.list_keys() == []
    assert s.get("FOO") is None


# ── McpStore: list / shadowing ─────────────────────────────────────────────


@pytest.mark.unit
def test_list_empty_when_no_files(store):
    assert store.list() == []


@pytest.mark.unit
def test_list_workspace_only(store):
    store.upsert(
        McpScope.WORKSPACE,
        "molexp-data",
        {"type": "stdio", "command": "molexp", "args": ["mcp-serve"]},
    )
    rows = store.list()
    assert len(rows) == 1
    assert rows[0].name == "molexp-data"
    assert rows[0].scope is McpScope.WORKSPACE
    assert rows[0].transport == "stdio"
    assert rows[0].command == "molexp"
    assert rows[0].shadowed is False
    assert rows[0].valid is True


@pytest.mark.unit
def test_list_user_only(store):
    store.upsert(
        McpScope.USER,
        "github",
        {"type": "http", "url": "https://api.example/mcp"},
    )
    rows = store.list()
    assert len(rows) == 1
    assert rows[0].scope is McpScope.USER
    assert rows[0].transport == "http"
    assert rows[0].url == "https://api.example/mcp"


@pytest.mark.unit
def test_workspace_shadows_user_when_same_name(store):
    store.upsert(
        McpScope.USER,
        "github",
        {"type": "http", "url": "https://api.example/mcp"},
    )
    store.upsert(
        McpScope.WORKSPACE,
        "github",
        {"type": "http", "url": "https://workspace.example/mcp"},
    )
    rows = store.list()
    by_scope = {r.scope: r for r in rows}
    assert by_scope[McpScope.USER].shadowed is True
    assert by_scope[McpScope.WORKSPACE].shadowed is False
    assert by_scope[McpScope.WORKSPACE].url == "https://workspace.example/mcp"


@pytest.mark.unit
def test_different_names_at_different_scopes_do_not_shadow(store):
    store.upsert(
        McpScope.USER,
        "github",
        {"type": "http", "url": "https://gh.example/mcp"},
    )
    store.upsert(
        McpScope.WORKSPACE,
        "molexp-data",
        {"type": "stdio", "command": "molexp"},
    )
    rows = store.list()
    assert {r.name for r in rows} == {"github", "molexp-data"}
    assert all(r.shadowed is False for r in rows)


# ── McpStore: upsert validation ────────────────────────────────────────────


@pytest.mark.unit
def test_upsert_rejects_invalid_name(store):
    with pytest.raises(ValueError, match="Invalid server name"):
        store.upsert(
            McpScope.WORKSPACE,
            "Bad Name!",
            {"type": "stdio", "command": "x"},
        )


@pytest.mark.unit
def test_upsert_rejects_missing_type(store):
    with pytest.raises(ValueError, match="Invalid spec"):
        store.upsert(
            McpScope.WORKSPACE,
            "x",
            {"command": "x"},  # no type → no fallback inference
        )


@pytest.mark.unit
def test_upsert_rejects_unknown_type(store):
    with pytest.raises(ValueError, match="Invalid spec"):
        store.upsert(
            McpScope.WORKSPACE,
            "x",
            {"type": "carrier-pigeon", "command": "x"},
        )


@pytest.mark.unit
def test_upsert_rejects_empty_command(store):
    with pytest.raises(ValueError, match="Invalid spec"):
        store.upsert(
            McpScope.WORKSPACE,
            "x",
            {"type": "stdio", "command": ""},
        )


@pytest.mark.unit
def test_upsert_rejects_non_http_url(store):
    with pytest.raises(ValueError, match="URL scheme"):
        store.upsert(
            McpScope.WORKSPACE,
            "x",
            {"type": "http", "url": "ftp://example.com/mcp"},
        )


@pytest.mark.unit
def test_upsert_replaces_full_entry(store):
    store.upsert(
        McpScope.WORKSPACE,
        "x",
        {"type": "stdio", "command": "old", "args": ["a"], "env": {"K": "v"}},
    )
    store.upsert(
        McpScope.WORKSPACE,
        "x",
        {"type": "stdio", "command": "new"},
    )
    entry = store.get(McpScope.WORKSPACE, "x")
    assert entry is not None
    assert entry.command == "new"
    assert entry.args == ()
    assert entry.env_keys == ()


@pytest.mark.unit
def test_upsert_writes_to_correct_scope(store):
    store.upsert(
        McpScope.USER,
        "u",
        {"type": "stdio", "command": "x"},
    )
    store.upsert(
        McpScope.WORKSPACE,
        "w",
        {"type": "stdio", "command": "x"},
    )
    user_data = json.loads(store.config_path(McpScope.USER).read_text())
    ws_data = json.loads(store.config_path(McpScope.WORKSPACE).read_text())
    assert "u" in user_data["mcpServers"]
    assert "u" not in ws_data["mcpServers"]
    assert "w" in ws_data["mcpServers"]
    assert "w" not in user_data["mcpServers"]


# ── McpStore: delete ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_delete_returns_false_when_missing(store):
    assert store.delete(McpScope.WORKSPACE, "ghost") is False


@pytest.mark.unit
def test_delete_only_affects_target_scope(store):
    store.upsert(McpScope.USER, "x", {"type": "stdio", "command": "y"})
    store.upsert(McpScope.WORKSPACE, "x", {"type": "stdio", "command": "z"})
    assert store.delete(McpScope.WORKSPACE, "x") is True
    rows = store.list()
    assert len(rows) == 1
    assert rows[0].scope is McpScope.USER
    assert rows[0].shadowed is False


# ── McpStore: secret references + resolution ──────────────────────────────


@pytest.mark.unit
def test_secret_refs_detected_from_env_values(store):
    store.upsert(
        McpScope.WORKSPACE,
        "gh",
        {
            "type": "stdio",
            "command": "x",
            "env": {"TOKEN": "${SECRET:GITHUB_TOKEN}"},
        },
    )
    entry = store.get(McpScope.WORKSPACE, "gh")
    assert entry is not None
    assert entry.secret_refs == ("GITHUB_TOKEN",)
    assert entry.unresolved_secrets == ("GITHUB_TOKEN",)


@pytest.mark.unit
def test_secret_refs_detected_from_header_values(store):
    store.upsert(
        McpScope.WORKSPACE,
        "gh",
        {
            "type": "http",
            "url": "https://gh/mcp",
            "headers": {"Authorization": "Bearer ${SECRET:GH_TOKEN}"},
        },
    )
    entry = store.get(McpScope.WORKSPACE, "gh")
    assert entry is not None
    assert entry.secret_refs == ("GH_TOKEN",)


@pytest.mark.unit
def test_workspace_secret_resolves_unresolved_secret(store):
    store.upsert(
        McpScope.WORKSPACE,
        "gh",
        {
            "type": "stdio",
            "command": "x",
            "env": {"TOKEN": "${SECRET:GITHUB_TOKEN}"},
        },
    )
    store.secrets(McpScope.WORKSPACE).set("GITHUB_TOKEN", "ghp_abc")
    entry = store.get(McpScope.WORKSPACE, "gh")
    assert entry is not None
    assert entry.unresolved_secrets == ()


@pytest.mark.unit
def test_user_secret_satisfies_workspace_entry(store):
    """User secrets cover workspace entries — single shared keyring per user."""
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
def test_workspace_secret_takes_precedence_over_user_secret(store):
    store.upsert(
        McpScope.WORKSPACE,
        "gh",
        {
            "type": "stdio",
            "command": "x",
            "env": {"TOKEN": "${SECRET:GH}"},
        },
    )
    store.secrets(McpScope.USER).set("GH", "user-value")
    store.secrets(McpScope.WORKSPACE).set("GH", "workspace-value")
    entry = store.get(McpScope.WORKSPACE, "gh")
    assert entry is not None
    resolved = store.resolve(entry)
    assert resolved.env["TOKEN"] == "workspace-value"


@pytest.mark.unit
def test_resolve_substitutes_secrets_in_env(store):
    store.upsert(
        McpScope.WORKSPACE,
        "gh",
        {
            "type": "stdio",
            "command": "gh-mcp",
            "args": ["--server"],
            "env": {
                "GITHUB_TOKEN": "${SECRET:GH_TOKEN}",
                "STATIC": "literal-value",
            },
        },
    )
    store.secrets(McpScope.WORKSPACE).set("GH_TOKEN", "ghp_real")
    entry = store.get(McpScope.WORKSPACE, "gh")
    assert entry is not None
    resolved = store.resolve(entry)
    assert resolved.transport == "stdio"
    assert resolved.command == "gh-mcp"
    assert resolved.args == ("--server",)
    assert resolved.env == {
        "GITHUB_TOKEN": "ghp_real",
        "STATIC": "literal-value",
    }
    assert resolved.headers == {}


@pytest.mark.unit
def test_resolve_substitutes_secrets_in_headers(store):
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
def test_legacy_streamable_http_normalized_to_http(store, tmp_path):
    """Older configs with type='streamable-http' load as type='http'."""
    import json

    config = tmp_path / "workspace" / MCP_CONFIG_FILENAME
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "old": {
                        "type": "streamable-http",
                        "url": "https://old.example/mcp",
                    }
                }
            }
        )
    )
    entry = store.get(McpScope.WORKSPACE, "old")
    assert entry is not None
    assert entry.transport == "http"
    assert entry.valid is True


@pytest.mark.unit
def test_resolve_raises_when_secret_missing(store):
    store.upsert(
        McpScope.WORKSPACE,
        "gh",
        {
            "type": "stdio",
            "command": "x",
            "env": {"T": "${SECRET:MISSING}"},
        },
    )
    entry = store.get(McpScope.WORKSPACE, "gh")
    assert entry is not None
    with pytest.raises(UnresolvedSecretError) as exc:
        store.resolve(entry)
    assert exc.value.keys == ["MISSING"]


@pytest.mark.unit
def test_no_fallback_to_os_environ(store, monkeypatch):
    """Critical: env-var fallback was explicitly removed."""
    monkeypatch.setenv("GITHUB_TOKEN", "from-env")
    store.upsert(
        McpScope.WORKSPACE,
        "gh",
        {
            "type": "stdio",
            "command": "x",
            "env": {"T": "${SECRET:GITHUB_TOKEN}"},
        },
    )
    entry = store.get(McpScope.WORKSPACE, "gh")
    assert entry is not None
    assert entry.unresolved_secrets == ("GITHUB_TOKEN",)
    with pytest.raises(UnresolvedSecretError):
        store.resolve(entry)


# ── secret_references ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_secret_references_groups_by_key(store):
    store.upsert(
        McpScope.WORKSPACE,
        "gh1",
        {
            "type": "stdio",
            "command": "x",
            "env": {"T": "${SECRET:GH}"},
        },
    )
    store.upsert(
        McpScope.WORKSPACE,
        "gh2",
        {
            "type": "http",
            "url": "https://x/mcp",
            "headers": {"H": "Bearer ${SECRET:GH}"},
        },
    )
    refs = store.secret_references(McpScope.WORKSPACE)
    assert refs == {"GH": ["gh1", "gh2"]}


# ── Invalid entries surface clearly ────────────────────────────────────────


@pytest.mark.unit
def test_entry_without_type_marked_invalid(tmp_path, isolated_user_dir):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Hand-craft a legacy-style entry to exercise the read path
    (workspace / MCP_CONFIG_FILENAME).write_text(
        json.dumps({"mcpServers": {"legacy": {"command": "x"}}})
    )
    store = McpStore(workspace)
    rows = store.list()
    assert len(rows) == 1
    assert rows[0].valid is False
    assert (
        "type" in rows[0].invalid_reason.lower()
        or "discriminator" in rows[0].invalid_reason.lower()
    )


@pytest.mark.unit
def test_corrupt_config_returns_empty(tmp_path, isolated_user_dir):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / MCP_CONFIG_FILENAME).write_text("{not json")
    store = McpStore(workspace)
    assert store.list() == []


# ── usage_instructions field (molmcp-agent-default) ───────────────────────


@pytest.mark.unit
def test_stdio_spec_usage_instructions_round_trip():
    """ac-001 — StdioSpec round-trips an optional ``usage_instructions``."""
    spec = _SPEC_ADAPTER.validate_python(
        {"type": "stdio", "command": "x", "usage_instructions": "FOO"}
    )
    assert isinstance(spec, StdioSpec)
    assert spec.usage_instructions == "FOO"
    dumped = spec.model_dump()
    assert dumped["usage_instructions"] == "FOO"

    # Default: absent → empty string.
    bare = _SPEC_ADAPTER.validate_python({"type": "stdio", "command": "x"})
    assert isinstance(bare, StdioSpec)
    assert bare.usage_instructions == ""


@pytest.mark.unit
def test_http_spec_usage_instructions_round_trip():
    """ac-002 — HttpSpec round-trips an optional ``usage_instructions``."""
    spec = _SPEC_ADAPTER.validate_python(
        {"type": "http", "url": "https://x", "usage_instructions": "BAR"}
    )
    assert isinstance(spec, HttpSpec)
    assert spec.usage_instructions == "BAR"
    dumped = spec.model_dump()
    assert dumped["usage_instructions"] == "BAR"

    bare = _SPEC_ADAPTER.validate_python({"type": "http", "url": "https://x"})
    assert isinstance(bare, HttpSpec)
    assert bare.usage_instructions == ""


@pytest.mark.unit
def test_entry_surfaces_usage_instructions(store):
    """ac-003 — McpServerEntry surfaces the on-disk ``usage_instructions``."""
    store.upsert(
        McpScope.USER,
        "x",
        {"type": "stdio", "command": "x", "usage_instructions": "DOC"},
    )
    entry = store.get(McpScope.USER, "x")
    assert entry is not None
    assert entry.usage_instructions == "DOC"


@pytest.mark.unit
def test_entry_usage_instructions_default_empty(store):
    """Companion to ac-003 — entries written without the field expose ``""``."""
    store.upsert(
        McpScope.USER,
        "y",
        {"type": "stdio", "command": "y"},
    )
    entry = store.get(McpScope.USER, "y")
    assert entry is not None
    assert entry.usage_instructions == ""
