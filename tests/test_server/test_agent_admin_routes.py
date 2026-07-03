"""Provider + MCP settings routes — ``/api/agent/provider`` & ``/api/agent/mcp/servers``.

Spec ``vision-loop-03-settings-operator-config``: ``routes/agent_admin.py`` is
rewritten with prefix ``/agent`` and registered before ``agent.router`` so the
provider paths win over the legacy 503 catch-all, wired over the shared
``services.operator_config`` loader/writer and the ``agent.mcp.store`` user
layer. Field names bind to the UI contract in ``ui/src/app/state/api.ts``
(``ApiAgentProvider`` / ``ApiMcpServerList`` / ``ApiMcpServer``).

Secret rule under test everywhere: the stored API-key **value** never appears
in any response body — only ``apiKeySet`` + masked ``apiKeyPreview``
(first-2 + "…" + last-4).

Isolation: ``OPERATOR_CONFIG_PATH`` and the MCP ``USER_DIR`` are monkeypatched
to tmp paths — the operator's real ``~/.molexp/`` is never read or written —
and every bridged ``molexp.config`` key is snapshot/restored.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import molexp
from molexp.agent.mcp import defaults as mcp_defaults
from molexp.agent.mcp import store as mcp_store
from molexp.server.app import create_app
from molexp.server.dependencies import get_workspace
from molexp.services import operator_config
from molexp.services.operator_config import AGENT_MODEL_KEY, LEGACY_AGENT_MODEL_KEY
from molexp.workspace import Workspace

_MODEL = "deepseek:deepseek-v4-flash"
_RAW_KEY = "sk-operator-secret-key-9876"
_MASKED_PREVIEW = "sk…9876"  # first-2 + "…" + last-4

#: Every in-code key the operator-config bridge (or a PUT re-bridge) may touch.
_BRIDGED_KEYS = (
    AGENT_MODEL_KEY,
    LEGACY_AGENT_MODEL_KEY,
    "deepseek_api_key",
    "anthropic_api_key",
    "openai_api_key",
    "google_api_key",
)


@pytest.fixture(autouse=True)
def _clean_molexp_config() -> Iterator[None]:
    """Snapshot/restore the process-global ``molexp.config`` keys we touch."""
    saved = {
        key: molexp.config.get(key) for key in _BRIDGED_KEYS if molexp.config.get(key) is not None
    }
    for key in _BRIDGED_KEYS:
        if molexp.config.get(key) is not None:
            del molexp.config[key]
    yield
    for key in _BRIDGED_KEYS:
        if molexp.config.get(key) is not None:
            del molexp.config[key]
    for key, value in saved.items():
        molexp.config[key] = value


@pytest.fixture
def operator_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the shared loader/writer at a tmp config file (never ~/.molexp)."""
    path = tmp_path / "molexp-home" / "config.json"
    monkeypatch.setattr(operator_config, "OPERATOR_CONFIG_PATH", path)
    return path


@pytest.fixture
def mcp_user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the MCP store's user layer to tmp; stub default seeding."""
    user_dir = tmp_path / "molexp-home"
    monkeypatch.setattr(mcp_store, "USER_DIR", user_dir)
    monkeypatch.setattr(mcp_defaults, "seed_user_defaults", lambda *a, **kw: False)  # noqa: ARG005
    return user_dir


@pytest.fixture
def client(workspace: Workspace, operator_config_path: Path, mcp_user_dir: Path) -> TestClient:
    app = create_app(serve_static=False)
    app.dependency_overrides[get_workspace] = lambda: workspace
    return TestClient(app)


@pytest.fixture
def seeded_config(operator_config_path: Path) -> Path:
    """A CLI-shaped operator config: model + provider key, no explicit provider."""
    operator_config_path.parent.mkdir(parents=True, exist_ok=True)
    operator_config_path.write_text(
        json.dumps({"agent": {"model": _MODEL, "deepseek_api_key": _RAW_KEY}})
    )
    return operator_config_path


# ── GET /api/agent/provider ──────────────────────────────────────────────────


class TestProviderGet:
    def test_get_provider_returns_ui_contract_fields(
        self, client: TestClient, seeded_config: Path
    ) -> None:
        """Exact camelCase field names of ``ApiAgentProvider`` (api.ts:1355)."""
        response = client.get("/api/agent/provider")
        assert response.status_code == 200
        body = response.json()
        for field in (
            "provider",
            "model",
            "baseUrl",
            "apiKeySet",
            "apiKeyPreview",
            "instructions",
            "supportedProviders",
        ):
            assert field in body, f"UI contract field {field!r} missing from {sorted(body)}"
        assert body["model"] == _MODEL
        assert body["provider"] == "deepseek"
        assert isinstance(body["baseUrl"], str)
        assert isinstance(body["instructions"], str)

    def test_get_provider_masks_api_key_preview(
        self, client: TestClient, seeded_config: Path
    ) -> None:
        body = client.get("/api/agent/provider").json()
        assert body["apiKeySet"] is True
        assert body["apiKeyPreview"] == _MASKED_PREVIEW

    def test_get_provider_never_echoes_raw_key(
        self, client: TestClient, seeded_config: Path
    ) -> None:
        response = client.get("/api/agent/provider")
        assert response.status_code == 200
        assert _RAW_KEY not in response.text

    def test_get_provider_reports_unset_key(
        self, client: TestClient, operator_config_path: Path
    ) -> None:
        operator_config_path.parent.mkdir(parents=True, exist_ok=True)
        operator_config_path.write_text(json.dumps({"agent": {"model": _MODEL}}))
        body = client.get("/api/agent/provider").json()
        assert body["apiKeySet"] is False
        assert body["apiKeyPreview"] == ""

    def test_supported_providers_include_known_registry(
        self, client: TestClient, seeded_config: Path
    ) -> None:
        body = client.get("/api/agent/provider").json()
        providers = body["supportedProviders"]
        assert isinstance(providers, list)
        assert {"anthropic", "openai", "google", "deepseek"} <= set(providers)


# ── PUT /api/agent/provider ──────────────────────────────────────────────────


class TestProviderPut:
    def test_put_persists_model_and_key_to_config_file(
        self, client: TestClient, operator_config_path: Path
    ) -> None:
        """The PUT lands in the operator-config file under the CLI spellings
        (``agent.model`` / ``agent.<provider>_api_key``) — one file, one writer."""
        response = client.put(
            "/api/agent/provider",
            json={"provider": "deepseek", "model": _MODEL, "api_key": _RAW_KEY},
        )
        assert response.status_code == 200
        stored = json.loads(operator_config_path.read_text())
        assert stored["agent"]["model"] == _MODEL
        assert stored["agent"]["deepseek_api_key"] == _RAW_KEY

    def test_put_rebridges_model_into_running_process(
        self, client: TestClient, operator_config_path: Path
    ) -> None:
        response = client.put(
            "/api/agent/provider",
            json={"provider": "deepseek", "model": _MODEL, "api_key": _RAW_KEY},
        )
        assert response.status_code == 200
        assert molexp.config.get(AGENT_MODEL_KEY) == _MODEL
        assert molexp.config.get("deepseek_api_key") == _RAW_KEY

    def test_put_unblocks_plan_task_model_guard(
        self, client: TestClient, operator_config_path: Path
    ) -> None:
        """The exact guard behind the plan-tasks 503 (``plan_tasks._configured_model``)
        finds the model after a UI-only save — no restart, no CLI round-trip."""
        from molexp.server.routes.plan_tasks import _configured_model

        assert _configured_model() is None  # clean slate — the 503 case
        response = client.put(
            "/api/agent/provider",
            json={"provider": "deepseek", "model": _MODEL, "api_key": _RAW_KEY},
        )
        assert response.status_code == 200
        assert _configured_model() == _MODEL

    def test_second_put_overrides_previously_bridged_value(
        self, client: TestClient, operator_config_path: Path
    ) -> None:
        """PUT clears the bridged keys it owns before re-bridging — a UI update
        is never shadowed by the stale value bridged by an earlier PUT."""
        first = client.put(
            "/api/agent/provider",
            json={"provider": "deepseek", "model": _MODEL, "api_key": _RAW_KEY},
        )
        assert first.status_code == 200
        second = client.put(
            "/api/agent/provider",
            json={"provider": "deepseek", "model": "deepseek:deepseek-v4-pro"},
        )
        assert second.status_code == 200
        assert molexp.config.get(AGENT_MODEL_KEY) == "deepseek:deepseek-v4-pro"

    def test_put_response_never_echoes_raw_key(
        self, client: TestClient, operator_config_path: Path
    ) -> None:
        response = client.put(
            "/api/agent/provider",
            json={"provider": "deepseek", "model": _MODEL, "api_key": _RAW_KEY},
        )
        assert response.status_code == 200
        assert _RAW_KEY not in response.text
        body = response.json()
        assert body["apiKeySet"] is True
        assert body["apiKeyPreview"] == _MASKED_PREVIEW


# ── POST /api/agent/provider/test ────────────────────────────────────────────


class TestProviderTest:
    def test_bogus_model_reports_ok_false_with_error(
        self, client: TestClient, seeded_config: Path
    ) -> None:
        """Preflight failure is a 200 ``{ok: false, error}`` — the UI's
        ``ApiAgentProviderTestResult`` shape (its Settings form reads
        ``.error``): the one-line human-readable preflight reason, never a
        raw traceback."""
        response = client.post("/api/agent/provider/test", json={"model": "stub-model"})
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        error = body["error"]
        assert isinstance(error, str)
        assert "stub-model" in error

    def test_test_endpoint_leaves_config_file_untouched(
        self, client: TestClient, seeded_config: Path
    ) -> None:
        """The test endpoint is read-only: no disk writes, config byte-stable."""
        before = seeded_config.read_bytes()
        response = client.post("/api/agent/provider/test", json={"model": "stub-model"})
        assert response.status_code == 200
        assert seeded_config.read_bytes() == before


# ── Route precedence over the legacy 503 catch-all ───────────────────────────


class TestRoutePrecedence:
    def test_provider_route_wins_over_legacy_catch_all(
        self, client: TestClient, seeded_config: Path
    ) -> None:
        """agent_admin registers before agent.router — provider paths are served,
        not swallowed by the ``/agent/{path:path}`` 503."""
        response = client.get("/api/agent/provider")
        assert response.status_code != 503
        assert response.status_code == 200

    def test_retired_session_path_still_503s(self, client: TestClient) -> None:
        """Genuinely-retired session paths keep the honest 503 catch-all."""
        response = client.get("/api/agent/sessions")
        assert response.status_code == 503


# ── GET/POST /api/agent/mcp/servers ──────────────────────────────────────────


class TestMcpServers:
    def test_mcp_servers_list_returns_ui_contract_shape(
        self, client: TestClient, mcp_user_dir: Path
    ) -> None:
        """Exact camelCase field names of ``ApiMcpServerList`` (api.ts:1189)."""
        response = client.get("/api/agent/mcp/servers")
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body["servers"], list)
        assert "workspacePath" in body
        assert "userPath" in body

    def test_mcp_server_post_upserts_user_scope_entry(
        self, client: TestClient, mcp_user_dir: Path
    ) -> None:
        """POST body binds to ``McpServerUpsertInput`` (api.ts:1232); the
        response is one ``ApiMcpServer`` row with camelCase fields."""
        response = client.post(
            "/api/agent/mcp/servers",
            json={
                "name": "echo-server",
                "scope": "user",
                "spec": {"type": "stdio", "command": "echo", "args": ["hi"], "env": {}},
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["name"] == "echo-server"
        assert body["scope"] == "user"
        assert body["transport"] == "stdio"
        for field in ("envKeys", "headerKeys", "secretRefs", "unresolvedSecrets", "valid"):
            assert field in body, f"UI contract field {field!r} missing from {sorted(body)}"

    def test_mcp_server_post_persists_into_tmp_user_store(
        self, client: TestClient, mcp_user_dir: Path
    ) -> None:
        """The upsert lands in the (tmp-scoped) user-layer ``mcp.json`` and a
        follow-up GET lists it — never the operator's real ``~/.molexp/mcp.json``."""
        response = client.post(
            "/api/agent/mcp/servers",
            json={
                "name": "echo-server",
                "scope": "user",
                "spec": {"type": "stdio", "command": "echo", "args": [], "env": {}},
            },
        )
        assert response.status_code == 201

        user_config = mcp_user_dir / "mcp.json"
        assert user_config.exists()
        stored = json.loads(user_config.read_text())
        assert stored["mcpServers"]["echo-server"]["command"] == "echo"

        listed = client.get("/api/agent/mcp/servers").json()
        names = [entry["name"] for entry in listed["servers"]]
        assert "echo-server" in names
