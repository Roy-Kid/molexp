"""Tests for the slash-command and instructions HTTP surface."""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_list_commands_includes_builtins(client):
    response = client.get("/api/agent/commands")
    assert response.status_code == 200
    by_slash = {c["slashName"]: c for c in response.json()["commands"]}
    for name in ("plan", "clear", "model", "help"):
        assert by_slash[name]["isBuiltin"] is True
    assert by_slash["plan"]["defaultPlanMode"] is True


@pytest.mark.integration
def test_list_commands_includes_skills_with_slash_name(client):
    create = client.post(
        "/api/agent/skills",
        json={
            "name": "Plot",
            "goal_template": "plot {{metric}} vs {{param}}",
            "slash_name": "plot",
            "instructions": "Use cm units.",
            "default_plan_mode": True,
        },
    )
    assert create.status_code == 201
    listing = client.get("/api/agent/commands").json()["commands"]
    plot = next(c for c in listing if c["slashName"] == "plot")
    assert plot["isBuiltin"] is False
    assert plot["defaultPlanMode"] is True
    assert {p["name"] for p in plot["parameters"]} == {"metric", "param"}


@pytest.mark.integration
def test_skill_without_slash_name_is_excluded_from_commands(client):
    client.post(
        "/api/agent/skills",
        json={"name": "Hidden", "goal_template": "x"},
    )
    listing = client.get("/api/agent/commands").json()["commands"]
    assert "Hidden" not in {c["name"] for c in listing}


@pytest.mark.integration
def test_parse_command_skill_args(client):
    create = client.post(
        "/api/agent/skills",
        json={
            "name": "Greet",
            "goal_template": "say {{message}}",
            "slash_name": "greet",
        },
    )
    skill_id = create.json()["id"]
    parsed = client.post(
        "/api/agent/commands/parse", json={"raw": "/greet message=hello"}
    ).json()
    assert parsed["kind"] == "skill"
    assert parsed["skillId"] == skill_id
    assert parsed["parameters"] == {"message": "hello"}


@pytest.mark.integration
def test_parse_command_unknown_returns_error(client):
    parsed = client.post(
        "/api/agent/commands/parse", json={"raw": "/unknownXyz"}
    ).json()
    assert parsed["kind"] == "error"
    assert "/unknownxyz" in parsed["error"].lower()


@pytest.mark.integration
def test_parse_command_plan_builtin(client):
    parsed = client.post("/api/agent/commands/parse", json={"raw": "/plan"}).json()
    assert parsed["kind"] == "builtin"
    assert parsed["name"] == "plan"
    assert parsed["planMode"] is True


@pytest.mark.integration
def test_provider_instructions_round_trip(client):
    response = client.put(
        "/api/agent/provider", json={"instructions": "Always cite sources."}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["instructions"] == "Always cite sources."
    again = client.get("/api/agent/provider").json()
    assert again["instructions"] == "Always cite sources."


@pytest.mark.integration
def test_provider_instructions_can_be_cleared(client):
    client.put("/api/agent/provider", json={"instructions": "abc"})
    cleared = client.put("/api/agent/provider", json={"instructions": ""}).json()
    assert cleared["instructions"] == ""


@pytest.mark.integration
def test_skill_create_rejects_reserved_slash_name(client):
    response = client.post(
        "/api/agent/skills",
        json={
            "name": "Reserved",
            "goal_template": "x",
            "slash_name": "plan",
        },
    )
    assert response.status_code == 400
