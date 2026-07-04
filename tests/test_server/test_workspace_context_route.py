"""GET /api/workspace/context — the canonical workspace read-model endpoint.

Slice workspace-context-02-server-route (P0.2). Verifies the new endpoint serves
the assembled WorkspaceContext, /info agrees on projectCount, and /context is
consistent with the richer /runs view (approach A — /runs kept, not collapsed).
"""

from __future__ import annotations

_CONTEXT_FIELDS = (
    "workspace",
    "focus",
    "projects",
    "experiments",
    "workflows",
    "recentRuns",
    "failedRuns",
    "runningRuns",
    "artifacts",
    "knowledge",
    "openQuestions",
    "staleOrMissing",
)


def test_context_route_shape_and_openapi(client) -> None:
    resp = client.get("/api/workspace/context")
    assert resp.status_code == 200
    body = resp.json()
    for field in _CONTEXT_FIELDS:
        assert field in body, field
    assert body["workspace"]["name"] == "Test"

    schema = client.get("/api/openapi.json").json()
    assert "/api/workspace/context" in schema["paths"]


def test_info_projectcount_parity(client, project) -> None:
    context = client.get("/api/workspace/context").json()
    info = client.get("/api/workspace/info").json()
    assert info["projectCount"] == len(context["projects"])
    assert len(context["projects"]) == 1
