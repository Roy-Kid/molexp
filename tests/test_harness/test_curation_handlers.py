"""Guarded-execution slice 02 — ObjectRef resolver + curation handlers.

RED-first: ``molexp.harness.actions.resolve`` / ``…actions.handlers.curation``
do not exist yet. Exercises the resolver over a real on-disk Workspace and the
two curation handlers (in-process over ``molexp.workspace.curation``), driven
through the slice-01 ``ProposalExecutor``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from molexp.harness.schemas.change_proposal import (
    ChangeProposal,
    ChangeSpec,
    ObjectRef,
    StateSnapshot,
)
from molexp.workspace import Workspace
from molexp.workspace.folder import Folder


def _ctx(tmp_path: Path):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "harness.sqlite"
    artifacts = FileArtifactStore(root=tmp_path / "artifacts")
    events = SQLiteEventLog(path=db)
    lineage = SQLiteArtifactLineageStore(path=db, artifact_store=artifacts)
    return HarnessRunContext(
        run_id="run-ge2",
        workspace_root=tmp_path,
        artifact_store=artifacts,
        event_log=events,
        lineage_store=lineage,
    )


def _workspace(tmp_path: Path) -> Workspace:
    """A workspace with project p / experiments e1,e2 / run r1 / a root folder."""
    ws = Workspace(root=tmp_path, name="Lab")
    project = ws.add_project("p")
    project.add_experiment("e1", workflow_source="t.py")
    project.add_experiment("e2", workflow_source="t.py")
    project.get_experiment("e1").add_run(id="r1")
    ws.add_folder(Folder(parent=ws, name="scratch", kind="curation.test"))
    return ws


def _proposal(op: str, *, affected: list[ObjectRef], payload: dict) -> ChangeProposal:
    return ChangeProposal(
        id="cp-ge2",
        intent=f"{op} action",
        current_state=StateSnapshot(objects=list(affected)),
        proposed_change=ChangeSpec(op=op, summary=op, payload=payload),
        affected_objects=list(affected),
        expected_benefit="tidier tree",
        risks=[],
        reversibility="reversible",
        approval_level="user",
        evidence=[],
        knowledge=[],
    )


def _curation_executor():
    from molexp.harness.actions import ChangeActionRegistry, ProposalExecutor
    from molexp.harness.actions.handlers import register_curation_handlers

    registry = ChangeActionRegistry()
    register_curation_handlers(registry)
    return ProposalExecutor(registry)


# ── resolver ─────────────────────────────────────────────────────────────────


def test_resolve_happy_each_kind(tmp_path: Path) -> None:
    """ac-001 — resolve_object_ref returns the live entity for each kind."""
    from molexp.harness.actions import resolve_object_ref

    ws = _workspace(tmp_path)
    # import a data asset under e1 so a data_asset ref resolves
    payload = tmp_path / "src.txt"
    payload.write_text("hello")
    asset = ws.get_project("p").get_experiment("e1").data_assets.import_asset("d", payload)

    run = resolve_object_ref(tmp_path, ObjectRef(kind="run", id="r1"))
    assert run.id == "r1"
    exp = resolve_object_ref(tmp_path, ObjectRef(kind="experiment", id="e2"))
    assert exp.id == "e2"
    folder = resolve_object_ref(tmp_path, ObjectRef(kind="folder", id="scratch"))
    assert folder.name == "scratch"
    da = resolve_object_ref(tmp_path, ObjectRef(kind="data_asset", id=asset.asset_id))
    assert da.asset_id == asset.asset_id


def test_resolve_loud_on_missing_or_unknown(tmp_path: Path) -> None:
    """ac-002 — unresolvable id / unknown kind raise ObjectRefResolutionError."""
    from molexp.harness.actions import resolve_object_ref
    from molexp.harness.errors import ObjectRefResolutionError

    _workspace(tmp_path)
    with pytest.raises(ObjectRefResolutionError):
        resolve_object_ref(tmp_path, ObjectRef(kind="run", id="ghost"))
    with pytest.raises(ObjectRefResolutionError):
        resolve_object_ref(tmp_path, ObjectRef(kind="galaxy", id="x"))


# ── handlers ─────────────────────────────────────────────────────────────────


def _result_obj(ctx, outcome):
    ref_id = outcome.result_artifact_ids[0]
    return json.loads(ctx.artifact_store.get(ref_id))


def test_move_run_relocates_and_records_executed(tmp_path: Path) -> None:
    """ac-003 — asset_move/move_run relocates the run + records executed."""
    _workspace(tmp_path)
    ctx = _ctx(tmp_path)
    affected = [ObjectRef(kind="run", id="r1"), ObjectRef(kind="experiment", id="e2")]
    proposal = _proposal(
        "asset_move",
        affected=affected,
        payload={
            "curation_op": "move_run",
            "target_experiment": {"kind": "experiment", "id": "e2"},
        },
    )
    outcome = asyncio.run(_curation_executor().dispatch(ctx, proposal))
    assert outcome.status == "executed"
    ws = Workspace(root=tmp_path, name="Lab")
    assert not ws.get_project("p").get_experiment("e1").has_run("r1")
    assert ws.get_project("p").get_experiment("e2").has_run("r1")
    assert _result_obj(ctx, outcome)["proposal_id"] == "cp-ge2"


def test_rehome_asset_reimports_under_target(tmp_path: Path) -> None:
    """ac-004 — asset_move/rehome_asset re-homes the data_asset."""
    from molexp.workspace.assets.scan import scan_assets

    ws = _workspace(tmp_path)
    payload = tmp_path / "src.txt"
    payload.write_text("payload-bytes")
    asset = ws.get_project("p").get_experiment("e1").data_assets.import_asset("d", payload)

    ctx = _ctx(tmp_path)
    affected = [
        ObjectRef(kind="data_asset", id=asset.asset_id),
        ObjectRef(kind="experiment", id="e1"),
        ObjectRef(kind="experiment", id="e2"),
    ]
    proposal = _proposal(
        "asset_move",
        affected=affected,
        payload={
            "curation_op": "rehome_asset",
            "source": {"kind": "experiment", "id": "e1"},
            "target": {"kind": "experiment", "id": "e2"},
        },
    )
    outcome = asyncio.run(_curation_executor().dispatch(ctx, proposal))
    assert outcome.status == "executed"
    e2_dir = ws.get_project("p").get_experiment("e2").experiment_dir
    assert scan_assets(e2_dir), "the re-homed asset should be present under e2"
    assert _result_obj(ctx, outcome)["proposal_id"] == "cp-ge2"


def test_artifact_delete_removes_folder(tmp_path: Path) -> None:
    """ac-005 — artifact_delete removes the folder + records executed."""
    _workspace(tmp_path)
    ctx = _ctx(tmp_path)
    proposal = _proposal(
        "artifact_delete",
        affected=[ObjectRef(kind="folder", id="scratch")],
        payload={},
    )
    outcome = asyncio.run(_curation_executor().dispatch(ctx, proposal))
    assert outcome.status == "executed"
    ws = Workspace(root=tmp_path, name="Lab")
    assert not ws.has_folder("scratch", cls=Folder)
    assert _result_obj(ctx, outcome)["proposal_id"] == "cp-ge2"


def test_artifact_delete_removes_run_kind(tmp_path: Path) -> None:
    """ac-009 (curate-unify-01) — artifact_delete deletes a run-kind affected object."""
    _workspace(tmp_path)
    ctx = _ctx(tmp_path)
    proposal = _proposal(
        "artifact_delete",
        affected=[ObjectRef(kind="run", id="r1")],
        payload={},
    )
    outcome = asyncio.run(_curation_executor().dispatch(ctx, proposal))
    assert outcome.status == "executed"
    ws = Workspace(root=tmp_path, name="Lab")
    assert not ws.get_project("p").get_experiment("e1").has_run("r1")


def test_out_of_bounds_target_fails_loud_no_mutation(tmp_path: Path) -> None:
    """ac-006 — a target outside affected_objects raises loudly, no fs change."""
    from molexp.harness.errors import OutOfAffectedScopeError

    _workspace(tmp_path)
    ctx = _ctx(tmp_path)
    # affected omits the target experiment e2, but the payload names it
    proposal = _proposal(
        "asset_move",
        affected=[ObjectRef(kind="run", id="r1")],
        payload={
            "curation_op": "move_run",
            "target_experiment": {"kind": "experiment", "id": "e2"},
        },
    )
    # a scope violation is raised inside the handler → executor records failed
    outcome = asyncio.run(_curation_executor().dispatch(ctx, proposal))
    assert outcome.status == "failed"
    ws = Workspace(root=tmp_path, name="Lab")
    assert ws.get_project("p").get_experiment("e1").has_run("r1")  # run not moved
    # and the guard itself is the loud typed error
    from molexp.harness.actions import assert_within_affected_scope

    with pytest.raises(OutOfAffectedScopeError):
        assert_within_affected_scope(proposal, [ObjectRef(kind="experiment", id="e2")])
