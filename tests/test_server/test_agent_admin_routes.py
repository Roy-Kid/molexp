"""Tests for /api/agent admin routes (MCP, tools, skills)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_user_dir(tmp_path, monkeypatch):
    """Redirect both McpStore.USER_DIR and Path.home() so tests never see ``~/.molexp``.

    The agent's :class:`SkillStore` reads ``~/.molexp/skills.json`` for the
    user-home tier. Without the ``Path.home`` patch a developer's local
    skills would leak into the test suite (and writes would persist).
    """
    from molexp.plugins.agent_pydanticai import mcp_store as mcp_mod

    fake_home_root = tmp_path / "_home"
    fake_user_dir = fake_home_root / ".molexp"
    monkeypatch.setattr(mcp_mod, "USER_DIR", fake_user_dir)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home_root))


# ── MCP server: list ────────────────────────────────────────────────────────


@pytest.mark.integration
def test_get_mcp_servers_empty(client):
    response = client.get("/api/agent/mcp/servers")
    assert response.status_code == 200
    body = response.json()
    assert body["workspacePath"].endswith(".mcp.json")
    assert body["userPath"].endswith(".mcp.json")
    assert body["servers"] == []


@pytest.mark.integration
def test_get_mcp_servers_lists_configured(workspace, client):
    payload = {
        "mcpServers": {
            "molcrafts": {
                "type": "stdio",
                "command": "molcrafts-mcp",
                "env": {"MOLEXP_WORKSPACE": "${workspaceRoot}"},
            }
        }
    }
    (workspace.root / ".mcp.json").write_text(json.dumps(payload))
    response = client.get("/api/agent/mcp/servers")
    assert response.status_code == 200
    body = response.json()
    server = body["servers"][0]
    assert server["name"] == "molcrafts"
    assert server["scope"] == "workspace"
    assert server["transport"] == "stdio"
    assert server["valid"] is True
    assert server["envKeys"] == ["MOLEXP_WORKSPACE"]
    assert server["shadowed"] is False


# ── MCP server: create / replace / delete ──────────────────────────────────


@pytest.mark.integration
def test_create_mcp_server_stdio(client):
    response = client.post(
        "/api/agent/mcp/servers",
        json={
            "name": "molexp-data",
            "scope": "workspace",
            "spec": {
                "type": "stdio",
                "command": "molexp",
                "args": ["mcp-serve"],
            },
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "molexp-data"
    assert body["scope"] == "workspace"
    assert body["transport"] == "stdio"
    assert body["command"] == "molexp"
    assert body["args"] == ["mcp-serve"]


@pytest.mark.integration
def test_create_mcp_server_http_with_secret_ref(client):
    response = client.post(
        "/api/agent/mcp/servers",
        json={
            "name": "github",
            "scope": "user",
            "spec": {
                "type": "http",
                "url": "https://api.example/mcp",
                "headers": {"Authorization": "Bearer ${SECRET:GH_TOKEN}"},
            },
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["scope"] == "user"
    assert body["transport"] == "http"
    assert body["url"] == "https://api.example/mcp"
    assert body["secretRefs"] == ["GH_TOKEN"]
    assert body["unresolvedSecrets"] == ["GH_TOKEN"]


@pytest.mark.integration
def test_create_mcp_server_rejects_duplicate_at_scope(client):
    client.post(
        "/api/agent/mcp/servers",
        json={
            "name": "x",
            "scope": "workspace",
            "spec": {"type": "stdio", "command": "y"},
        },
    )
    dup = client.post(
        "/api/agent/mcp/servers",
        json={
            "name": "x",
            "scope": "workspace",
            "spec": {"type": "stdio", "command": "z"},
        },
    )
    assert dup.status_code == 409


@pytest.mark.integration
def test_create_mcp_server_rejects_invalid_url(client):
    response = client.post(
        "/api/agent/mcp/servers",
        json={
            "name": "bad",
            "scope": "workspace",
            "spec": {"type": "http", "url": "ftp://example/mcp"},
        },
    )
    assert response.status_code == 400


@pytest.mark.integration
def test_put_mcp_server_replaces(client):
    client.post(
        "/api/agent/mcp/servers",
        json={
            "name": "x",
            "scope": "workspace",
            "spec": {
                "type": "stdio",
                "command": "old",
                "args": ["a"],
                "env": {"K": "v"},
            },
        },
    )
    response = client.put(
        "/api/agent/mcp/servers/x",
        json={
            "name": "x",
            "scope": "workspace",
            "spec": {"type": "stdio", "command": "new"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["command"] == "new"
    assert body["args"] == []
    assert body["envKeys"] == []


@pytest.mark.integration
def test_put_mcp_server_rejects_path_body_mismatch(client):
    response = client.put(
        "/api/agent/mcp/servers/foo",
        json={
            "name": "bar",
            "scope": "workspace",
            "spec": {"type": "stdio", "command": "x"},
        },
    )
    assert response.status_code == 400


@pytest.mark.integration
def test_delete_mcp_server(client):
    client.post(
        "/api/agent/mcp/servers",
        json={
            "name": "x",
            "scope": "workspace",
            "spec": {"type": "stdio", "command": "y"},
        },
    )
    deleted = client.delete("/api/agent/mcp/servers/x?scope=workspace")
    assert deleted.status_code == 200
    again = client.delete("/api/agent/mcp/servers/x?scope=workspace")
    assert again.status_code == 404


@pytest.mark.integration
def test_delete_only_targets_named_scope(client):
    client.post(
        "/api/agent/mcp/servers",
        json={
            "name": "x",
            "scope": "user",
            "spec": {"type": "stdio", "command": "y"},
        },
    )
    client.post(
        "/api/agent/mcp/servers",
        json={
            "name": "x",
            "scope": "workspace",
            "spec": {"type": "stdio", "command": "z"},
        },
    )
    client.delete("/api/agent/mcp/servers/x?scope=workspace")
    body = client.get("/api/agent/mcp/servers").json()
    remaining = [s for s in body["servers"] if s["name"] == "x"]
    assert len(remaining) == 1
    assert remaining[0]["scope"] == "user"


# ── MCP server: test endpoint ──────────────────────────────────────────────


@pytest.mark.integration
def test_test_mcp_server_404_when_missing(client):
    response = client.post("/api/agent/mcp/servers/ghost/test?scope=workspace")
    assert response.status_code == 404


@pytest.mark.integration
def test_test_mcp_server_uses_probe_module(client, monkeypatch):
    from molexp.plugins.agent_pydanticai import mcp_probe as probe_mod

    async def fake_probe(store, entry):
        return probe_mod.ProbeOutcome(ok=True, latency_ms=42, tool_count=3)

    monkeypatch.setattr(
        "molexp.server.routes.agent_admin.probe_server", fake_probe
    )
    client.post(
        "/api/agent/mcp/servers",
        json={
            "name": "x",
            "scope": "workspace",
            "spec": {"type": "stdio", "command": "y"},
        },
    )
    response = client.post("/api/agent/mcp/servers/x/test?scope=workspace")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["toolCount"] == 3
    assert body["latencyMs"] == 42
    assert body["transport"] == "stdio"


@pytest.mark.integration
def test_test_mcp_server_reports_missing_secrets(client, monkeypatch):
    from molexp.plugins.agent_pydanticai import mcp_probe as probe_mod

    async def fake_probe(store, entry):
        # Real probe would short-circuit on entry.unresolved_secrets;
        # mirror that here so the assertion is realistic.
        if entry.unresolved_secrets:
            return probe_mod.ProbeOutcome(
                ok=False,
                latency_ms=0,
                error=f"Missing secrets: {', '.join(entry.unresolved_secrets)}",
            )
        return probe_mod.ProbeOutcome(ok=True, latency_ms=10, tool_count=1)

    monkeypatch.setattr(
        "molexp.server.routes.agent_admin.probe_server", fake_probe
    )
    client.post(
        "/api/agent/mcp/servers",
        json={
            "name": "gh",
            "scope": "workspace",
            "spec": {
                "type": "http",
                "url": "https://gh/mcp",
                "headers": {"H": "Bearer ${SECRET:NOPE}"},
            },
        },
    )
    body = client.post("/api/agent/mcp/servers/gh/test?scope=workspace").json()
    assert body["ok"] is False
    assert "NOPE" in body["error"]


# ── MCP secrets ────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_secrets_list_empty(client):
    body = client.get("/api/agent/mcp/secrets?scope=workspace").json()
    assert body["scope"] == "workspace"
    assert body["secrets"] == []


@pytest.mark.integration
def test_secret_set_get_delete_round_trip(client):
    set_resp = client.put(
        "/api/agent/mcp/secrets/MY_TOKEN",
        json={"value": "tok-value", "scope": "workspace"},
    )
    assert set_resp.status_code == 200

    body = client.get("/api/agent/mcp/secrets?scope=workspace").json()
    rows = {r["key"]: r for r in body["secrets"]}
    assert rows["MY_TOKEN"]["isSet"] is True
    assert "tok-value" not in json.dumps(body)  # never echoed back

    cleared = client.put(
        "/api/agent/mcp/secrets/MY_TOKEN",
        json={"value": "", "scope": "workspace"},
    ).json()
    assert "cleared" in cleared["message"].lower()
    body2 = client.get("/api/agent/mcp/secrets?scope=workspace").json()
    assert body2["secrets"] == []


@pytest.mark.integration
def test_secrets_list_groups_referenced_servers(client):
    client.post(
        "/api/agent/mcp/servers",
        json={
            "name": "gh1",
            "scope": "workspace",
            "spec": {
                "type": "stdio",
                "command": "x",
                "env": {"T": "${SECRET:GH}"},
            },
        },
    )
    client.post(
        "/api/agent/mcp/servers",
        json={
            "name": "gh2",
            "scope": "workspace",
            "spec": {
                "type": "http",
                "url": "https://gh/mcp",
                "headers": {"H": "Bearer ${SECRET:GH}"},
            },
        },
    )
    body = client.get("/api/agent/mcp/secrets?scope=workspace").json()
    rows = {r["key"]: r for r in body["secrets"]}
    assert rows["GH"]["isSet"] is False
    assert rows["GH"]["referencedBy"] == ["gh1", "gh2"]


@pytest.mark.integration
def test_get_agent_tools_lists_natives(client):
    response = client.get("/api/agent/tools")
    assert response.status_code == 200
    names = {t["name"] for t in response.json()["tools"]}
    assert {"submit_run", "retry_run", "wait_for_run", "ask_user"} <= names


@pytest.mark.integration
def test_get_agent_tools_reports_default_approval_flags(client):
    """Default policy is friction-free; production opts in via custom policy."""
    response = client.get("/api/agent/tools")
    by_name = {t["name"]: t for t in response.json()["tools"]}
    assert by_name["submit_run"]["requiresApproval"] is False
    assert by_name["execute_run"]["requiresApproval"] is False
    assert by_name["get_run_status"]["requiresApproval"] is False


@pytest.mark.integration
def test_get_agent_tools_includes_mcp_groups(monkeypatch, workspace, client):
    """Each configured MCP server contributes a group + sourced tools."""
    payload = {
        "mcpServers": {
            "weather": {
                "type": "http",
                "url": "https://weather.example/mcp",
            }
        }
    }
    (workspace.root / ".mcp.json").write_text(json.dumps(payload))

    from molexp.plugins.agent_pydanticai.mcp_probe import (
        McpServerToolList,
        McpToolSummary,
    )

    async def fake_list(_store, **_):
        return [
            McpServerToolList(
                server="weather",
                scope="workspace",
                ok=True,
                tools=[
                    McpToolSummary(
                        name="get_forecast",
                        description="Forecast for a city.",
                        server="weather",
                        scope="workspace",
                    )
                ],
                error=None,
            )
        ]

    monkeypatch.setattr(
        "molexp.server.routes.agent_admin.list_mcp_tools",
        fake_list,
    )

    response = client.get("/api/agent/tools")
    assert response.status_code == 200
    body = response.json()
    groups = {g["server"]: g for g in body["mcpGroups"]}
    assert groups["weather"]["ok"] is True
    assert groups["weather"]["toolCount"] == 1
    sources = {t["source"] for t in body["tools"]}
    assert "mcp:weather" in sources


@pytest.mark.integration
def test_get_agent_tools_surfaces_mcp_failure(monkeypatch, workspace, client):
    """Broken MCP servers come back as ok=False groups so UI can show error."""
    payload = {
        "mcpServers": {
            "broken": {
                "type": "http",
                "url": "https://broken.example/mcp",
            }
        }
    }
    (workspace.root / ".mcp.json").write_text(json.dumps(payload))

    from molexp.plugins.agent_pydanticai.mcp_probe import McpServerToolList

    async def fake_list(_store, **_):
        return [
            McpServerToolList(
                server="broken",
                scope="workspace",
                ok=False,
                tools=[],
                error="HTTPStatusError: 401 Unauthorized",
            )
        ]

    monkeypatch.setattr(
        "molexp.server.routes.agent_admin.list_mcp_tools",
        fake_list,
    )

    response = client.get("/api/agent/tools")
    body = response.json()
    grp = next(g for g in body["mcpGroups"] if g["server"] == "broken")
    assert grp["ok"] is False
    assert "401" in grp["error"]
    assert grp["toolCount"] == 0


def _user_skills(payload: dict) -> list[dict]:
    """Filter the API listing down to user-created (non-builtin) skills."""
    return [s for s in payload["skills"] if not s.get("builtin", False)]


@pytest.mark.integration
def test_skill_lifecycle_via_api(client):
    list_resp = client.get("/api/agent/skills")
    assert list_resp.status_code == 200
    # Builtin /plan skill always present; user list starts empty.
    assert _user_skills(list_resp.json()) == []
    builtin_ids = {s["id"] for s in list_resp.json()["skills"] if s["builtin"]}
    assert "builtin-plan" in builtin_ids

    create_resp = client.post(
        "/api/agent/skills",
        json={
            "name": "Plot energy",
            "goal_template": "plot {{metric}} in {{project}}",
            "tags": ["plot"],
        },
    )
    assert create_resp.status_code == 201
    skill = create_resp.json()
    skill_id = skill["id"]
    assert skill["goalTemplate"].startswith("plot {{metric}}")
    assert skill["tags"] == ["plot"]
    assert skill["scope"] == "workspace"
    assert skill["builtin"] is False

    update_resp = client.patch(
        f"/api/agent/skills/{skill_id}",
        json={"name": "Renamed"},
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["name"] == "Renamed"

    list_resp = client.get("/api/agent/skills")
    assert len(_user_skills(list_resp.json())) == 1

    delete_resp = client.delete(f"/api/agent/skills/{skill_id}")
    assert delete_resp.status_code == 200
    assert client.get(f"/api/agent/skills/{skill_id}").status_code == 404


@pytest.mark.integration
def test_builtin_plan_skill_is_immutable_via_api(client):
    """/plan is a builtin — the API rejects update + delete with a 400/404."""
    update_resp = client.patch(
        "/api/agent/skills/builtin-plan",
        json={"name": "Hijacked"},
    )
    # Builtin skills aren't in the workspace tier, so the workspace-scoped
    # PATCH yields a 404 (no record at that scope).
    assert update_resp.status_code == 404

    delete_resp = client.delete("/api/agent/skills/builtin-plan")
    assert delete_resp.status_code == 404


@pytest.mark.integration
def test_create_skill_with_tool_scope(client):
    """Allow/deny globs and requires_exit_tool round-trip through the API."""
    resp = client.post(
        "/api/agent/skills",
        json={
            "name": "Plot only",
            "goal_template": "render {{metric}}",
            "allowed_tools": ["list_*", "mcp:python.*"],
            "denied_tools": ["execute_run"],
            "requires_exit_tool": "exit_plan_mode",
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["allowedTools"] == ["list_*", "mcp:python.*"]
    assert body["deniedTools"] == ["execute_run"]
    assert body["requiresExitTool"] == "exit_plan_mode"


@pytest.mark.integration
def test_get_skill_404s_when_missing(client):
    assert client.get("/api/agent/skills/nope").status_code == 404


@pytest.mark.integration
def test_update_skill_rejects_unknown_id(client):
    resp = client.patch("/api/agent/skills/nope", json={"name": "x"})
    assert resp.status_code == 404


# ── Provider config endpoints ─────────────────────────────────────────────


@pytest.mark.integration
def test_get_provider_default(client):
    response = client.get("/api/agent/provider")
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "anthropic"
    assert body["apiKeySet"] is False
    assert body["apiKeyPreview"] == ""
    assert "anthropic" in body["supportedProviders"]
    assert "openai" in body["supportedProviders"]


@pytest.mark.integration
def test_put_provider_persists_and_redacts(client):
    response = client.put(
        "/api/agent/provider",
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "sk-test-1234567890",
            "base_url": "",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4o-mini"
    assert body["apiKeySet"] is True
    assert body["apiKeyPreview"].endswith("7890")
    assert "1234" not in body["apiKeyPreview"]
    assert "api_key" not in body
    # Subsequent GET should match.
    follow = client.get("/api/agent/provider").json()
    assert follow["model"] == "gpt-4o-mini"
    assert follow["apiKeySet"] is True


@pytest.mark.integration
def test_put_provider_rejects_unknown_provider(client):
    response = client.put("/api/agent/provider", json={"provider": "wat"})
    assert response.status_code == 400


@pytest.mark.integration
def test_put_provider_clears_key_with_empty_string(client):
    client.put("/api/agent/provider", json={"api_key": "sk-test-1234567890"})
    cleared = client.put("/api/agent/provider", json={"api_key": ""}).json()
    assert cleared["apiKeySet"] is False
    assert cleared["apiKeyPreview"] == ""


@pytest.mark.integration
def test_put_provider_switch_resets_model(client):
    client.put(
        "/api/agent/provider",
        json={"provider": "anthropic", "model": "claude-sonnet-4-6"},
    )
    response = client.put("/api/agent/provider", json={"provider": "openai"}).json()
    assert response["provider"] == "openai"
    # Default OpenAI model — shouldn't carry over the Anthropic model name.
    assert response["model"] == "gpt-4o"


# ── Provider test endpoint ────────────────────────────────────────────────


@pytest.mark.integration
def test_test_provider_returns_no_key_error_when_unconfigured(client):
    response = client.post("/api/agent/provider/test", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert "No API key" in body["error"]
    assert body["latencyMs"] == 0


@pytest.mark.integration
def test_test_provider_uses_draft_key_without_persisting(client, monkeypatch):
    """Probing with an inline api_key must not save it."""
    from molexp.plugins.agent_pydanticai import provider as provider_mod

    captured: dict[str, str] = {}

    async def fake_probe(config):
        captured["api_key"] = config.api_key
        captured["model"] = config.model
        captured["provider"] = config.provider
        return provider_mod.ProbeResult(ok=True, latency_ms=42, reply="pong")

    monkeypatch.setattr(
        "molexp.server.routes.agent_admin.probe_provider",
        fake_probe,
    )
    response = client.post(
        "/api/agent/provider/test",
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "sk-draft-not-saved",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["model"] == "gpt-4o-mini"
    assert body["latencyMs"] == 42
    assert captured["api_key"] == "sk-draft-not-saved"

    # The store must remain unset.
    after = client.get("/api/agent/provider").json()
    assert after["apiKeySet"] is False
    assert after["provider"] == "anthropic"  # default unchanged


@pytest.mark.integration
def test_test_provider_falls_back_to_stored_key(client, monkeypatch):
    """If api_key is omitted, probe should use the persisted key."""
    from molexp.plugins.agent_pydanticai import provider as provider_mod

    client.put(
        "/api/agent/provider",
        json={"provider": "anthropic", "api_key": "sk-stored-key"},
    )

    captured: dict[str, str] = {}

    async def fake_probe(config):
        captured["api_key"] = config.api_key
        return provider_mod.ProbeResult(ok=True, latency_ms=10, reply="pong")

    monkeypatch.setattr(
        "molexp.server.routes.agent_admin.probe_provider",
        fake_probe,
    )
    response = client.post("/api/agent/provider/test", json={"model": "claude-opus-4-5"})
    assert response.status_code == 200
    assert captured["api_key"] == "sk-stored-key"
    body = response.json()
    assert body["ok"] is True
    assert body["model"] == "claude-opus-4-5"


@pytest.mark.integration
def test_test_provider_propagates_failure(client, monkeypatch):
    from molexp.plugins.agent_pydanticai import provider as provider_mod

    async def fake_probe(config):
        return provider_mod.ProbeResult(ok=False, latency_ms=200, error="401: bad key")

    monkeypatch.setattr(
        "molexp.server.routes.agent_admin.probe_provider",
        fake_probe,
    )
    response = client.post(
        "/api/agent/provider/test",
        json={"api_key": "sk-bad"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["error"] == "401: bad key"
    assert body["latencyMs"] == 200


@pytest.mark.integration
def test_test_provider_rejects_unknown_provider(client):
    response = client.post("/api/agent/provider/test", json={"provider": "wat"})
    assert response.status_code == 400

# ── MCP OAuth flow ──────────────────────────────────────────────────────────


@pytest.fixture
def oauth_server_in_workspace(workspace):
    """Persist a streamable-HTTP MCP entry with OAuth auth at workspace scope."""
    payload = {
        "mcpServers": {
            "fastmcp": {
                "type": "http",
                "url": "https://example.test/mcp",
                "headers": {},
                "auth": {"type": "oauth2", "scopes": ["openid", "email"]},
            }
        }
    }
    (workspace.root / ".mcp.json").write_text(json.dumps(payload))
    return "fastmcp"


@pytest.mark.integration
def test_oauth_status_returns_404_for_missing_server(client):
    response = client.get("/api/agent/mcp/servers/nope/oauth?scope=workspace")
    assert response.status_code == 404


@pytest.mark.integration
def test_oauth_status_returns_400_when_not_oauth(workspace, client):
    payload = {
        "mcpServers": {
            "plain": {"type": "stdio", "command": "echo"}
        }
    }
    (workspace.root / ".mcp.json").write_text(json.dumps(payload))
    response = client.get("/api/agent/mcp/servers/plain/oauth?scope=workspace")
    assert response.status_code == 400
    assert "OAuth" in response.json()["detail"]


@pytest.mark.integration
def test_oauth_status_reports_disconnected_initially(client, oauth_server_in_workspace):
    response = client.get(
        f"/api/agent/mcp/servers/{oauth_server_in_workspace}/oauth?scope=workspace"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["hasTokens"] is False
    assert body["scopes"] == ["openid", "email"]


@pytest.mark.integration
def test_oauth_disconnect_is_idempotent(client, oauth_server_in_workspace):
    response = client.delete(
        f"/api/agent/mcp/servers/{oauth_server_in_workspace}/oauth?scope=workspace"
    )
    assert response.status_code == 200
    assert "No OAuth tokens" in response.json()["message"]


@pytest.mark.integration
def test_oauth_callback_returns_410_when_no_session(client, oauth_server_in_workspace):
    response = client.post(
        f"/api/agent/mcp/servers/{oauth_server_in_workspace}/oauth/callback?scope=workspace",
        json={"code": "abc", "state": None},
    )
    assert response.status_code == 410
    assert "No in-flight" in response.json()["detail"]


@pytest.mark.integration
def test_server_response_includes_auth_summary(client, oauth_server_in_workspace):
    response = client.get("/api/agent/mcp/servers")
    assert response.status_code == 200
    server = response.json()["servers"][0]
    assert server["auth"] == {
        "type": "oauth2",
        "scopes": ["openid", "email"],
        "clientId": None,
        "connected": False,
    }


@pytest.mark.integration
def test_oauth_start_returns_400_when_no_metadata(monkeypatch, client, oauth_server_in_workspace):
    """Preflight catches non-OAuth servers before SDK tries to discover."""

    async def _fake_no_metadata(_url):
        return False

    monkeypatch.setattr(
        "molexp.server.routes.agent_admin._has_oauth_metadata",
        _fake_no_metadata,
    )
    response = client.post(
        f"/api/agent/mcp/servers/{oauth_server_in_workspace}/oauth/start?scope=workspace",
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "does not advertise OAuth metadata" in detail
    assert "Authentication to None" in detail

