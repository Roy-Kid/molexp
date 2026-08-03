"""``GET /api/agent/health`` and slash-command routes must not hit the 503 catch-all."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_agent_health_returns_200_json(client: TestClient) -> None:
    response = client.get("/api/agent/health")
    assert response.status_code == 200, response.text
    body = response.json()
    assert "ready" in body
    assert "provider" in body
    assert "model" in body
    assert body["source"] in {"stored", "env", "none"}


def test_agent_commands_list_builtins(client: TestClient) -> None:
    response = client.get("/api/agent/commands")
    assert response.status_code == 200, response.text
    commands = response.json()["commands"]
    slash = {c["slashName"] for c in commands}
    assert {"plan", "clear", "model", "help"} <= slash


def test_agent_commands_parse_builtin(client: TestClient) -> None:
    response = client.post("/api/agent/commands/parse", json={"raw": "/plan"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "builtin"
    assert body["name"] == "plan"
    assert body["planMode"] is True


def test_legacy_unknown_agent_path_still_503(client: TestClient) -> None:
    response = client.get("/api/agent/sessions/does-not-exist")
    assert response.status_code == 503
