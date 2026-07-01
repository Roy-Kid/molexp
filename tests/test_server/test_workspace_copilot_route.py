"""GET /api/workspace/copilot — the read-only Workspace Copilot summary (P0.6)."""

from __future__ import annotations

_SUMMARY_FIELDS = (
    "workspace",
    "headline",
    "counts",
    "failedRuns",
    "runningRuns",
    "healthFlags",
    "openQuestions",
    "relevantKnowledge",
    "nextActions",
)


def test_copilot_route_shape_and_openapi(client) -> None:
    resp = client.get("/api/workspace/copilot")
    assert resp.status_code == 200
    body = resp.json()
    for field in _SUMMARY_FIELDS:
        assert field in body, field
    assert body["workspace"]["name"] == "Test"

    schema = client.get("/api/openapi.json").json()
    assert "/api/workspace/copilot" in schema["paths"]


def test_copilot_surfaces_failed_run_next_action(client, run) -> None:
    from molexp.workspace.models import RunStatus
    from molexp.workspace.run_ops import RunOpsState

    run.write_ops(RunOpsState(status=RunStatus.FAILED))

    body = client.get("/api/workspace/copilot").json()
    kinds = [a["kind"] for a in body["nextActions"]]
    assert "retry_failed_run" in kinds

    retry = next(a for a in body["nextActions"] if a["kind"] == "retry_failed_run")
    assert retry["requiresProposal"] is True  # high-risk → needs a ChangeProposal
    assert retry["advisory"] is True  # never executed by the Copilot
