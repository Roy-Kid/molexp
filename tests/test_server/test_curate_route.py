"""curate-unify-03 — deterministic builder + POST /api/workspace/curate route."""

from __future__ import annotations

import pytest

from molexp.harness.schemas.change_proposal import ObjectRef
from molexp.services.curate_runtime import build_curation_proposal
from molexp.workspace import Workspace

_CURATE = "/api/workspace/curate"


def _fresh(workspace):
    """Re-read the workspace from disk (the backend mutates via its own instance)."""
    return Workspace(root=workspace.root, name="Test").get_project("p")


def _seed(workspace):
    proj = workspace.add_project("p")
    proj.add_experiment("source", workflow_source="t.py")
    proj.add_experiment("target", workflow_source="t.py")
    proj.get_experiment("source").add_run(id="subject")
    return proj


# ── builder ──────────────────────────────────────────────────────────────────


def test_build_move_run() -> None:
    """ac-001 — move_run builds an asset_move proposal."""
    p = build_curation_proposal("move_run", run="r1", target_experiment="e2")
    assert p.proposed_change.op == "asset_move"
    assert p.proposed_change.payload["curation_op"] == "move_run"
    assert ObjectRef(kind="run", id="r1") in p.affected_objects
    assert ObjectRef(kind="experiment", id="e2") in p.affected_objects


def test_build_delete_folder() -> None:
    """ac-002 — delete_folder builds an artifact_delete proposal (elevated)."""
    p = build_curation_proposal("delete_folder", folder="r1")
    assert p.proposed_change.op == "artifact_delete"
    assert p.affected_objects == [ObjectRef(kind="run", id="r1")]
    assert p.approval_level == "elevated"


def test_build_rehome_asset() -> None:
    """ac-003 — rehome_asset builds an asset_move proposal with typed scope refs."""
    p = build_curation_proposal(
        "rehome_asset",
        asset="a1",
        source={"kind": "experiment", "id": "e1"},
        target={"kind": "experiment", "id": "e2"},
        action="copy",
    )
    assert p.proposed_change.op == "asset_move"
    assert p.proposed_change.payload["curation_op"] == "rehome_asset"
    assert p.proposed_change.payload["action"] == "copy"
    assert ObjectRef(kind="data_asset", id="a1") in p.affected_objects
    assert ObjectRef(kind="experiment", id="e1") in p.affected_objects
    assert ObjectRef(kind="experiment", id="e2") in p.affected_objects


def test_build_missing_arg_raises() -> None:
    """A required arg missing for the chosen op raises ValueError."""
    with pytest.raises(ValueError):
        build_curation_proposal("move_run", run="r1")


# ── route ────────────────────────────────────────────────────────────────────


def test_route_move_run_executes(client, workspace) -> None:
    """ac-008 — POST /curate op=move_run approve=true executes on the shared backend."""
    _seed(workspace)
    resp = client.post(
        _CURATE,
        json={"op": "move_run", "run": "subject", "target_experiment": "target", "approve": True},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "executed"
    fresh = _fresh(workspace)
    assert not fresh.get_experiment("source").has_run("subject")
    assert fresh.get_experiment("target").has_run("subject")


def test_route_reject_no_mutation(client, workspace) -> None:
    """ac-007 — approve omitted/false records the proposal and mutates nothing."""
    _seed(workspace)
    resp = client.post(
        _CURATE,
        json={"op": "move_run", "run": "subject", "target_experiment": "target"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "rejected"
    assert _fresh(workspace).get_experiment("source").has_run("subject")  # unmoved


def test_curate_route_in_openapi(client) -> None:
    """ac-009 — the curate route is present in the OpenAPI schema."""
    schema = client.get("/api/openapi.json").json()
    assert _CURATE in schema["paths"]
    assert "post" in schema["paths"][_CURATE]
