"""Route curation mutations through the §8 ChangeProposal gate (shared backend).

curate-unify-01. The server-tier backend that turns a curation invocation into a
first-class :class:`~molexp.harness.schemas.change_proposal.ChangeProposal` and
drives it through the P2.1 gate (``gate_change_proposal`` + ``ProposalExecutor`` +
the curation handlers), so a *single* approval decision both records the proposal
and performs (or refuses) the mutation. This is the ONE execution stack that both
front-ends converge on — the LLM-NL ``run_curation_flow`` (slice 02) and the
deterministic CLI/route (slice 03).

Single gate: this path never re-runs the side-effect approval gate — the
ChangeProposal gate's ``ApprovalGate`` is the sole approval for the mutation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from molexp.harness import (
    ChangeActionRegistry,
    ChangeProposal,
    ChangeSpec,
    ObjectRef,
    ProposalExecutor,
    StateSnapshot,
    approval_level_for,
    gate_change_proposal,
    register_curation_handlers,
)
from molexp.harness.capabilities import curation_capabilities
from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

if TYPE_CHECKING:
    from molexp.harness.stages.approval_gate import Approver
    from molexp.workspace import Run, Workspace

__all__ = ["curation_invocation_to_proposal", "run_curation_proposal"]

_MOVE_RUN = "molexp.curation.move_run"
_DELETE_FOLDER = "molexp.curation.delete_folder"


def _proposal_id(capability_id: str, references: dict[str, str]) -> str:
    """A stable content-addressed proposal id (same invocation → same id)."""
    payload = json.dumps(
        {"capability_id": capability_id, "references": dict(sorted(references.items()))},
        sort_keys=True,
    )
    return f"cp-curate-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _is_known_curation_cap(capability_id: str) -> bool:
    return any(cap.id == capability_id for cap in curation_capabilities())


def curation_invocation_to_proposal(
    capability_id: str, references: dict[str, str]
) -> ChangeProposal | None:
    """Map a planner-shaped curation invocation onto a §8 :class:`ChangeProposal`.

    Args:
        capability_id: A ``molexp.curation.*`` capability id.
        references: The planner's flat JSON reference handles (ids / slugs).

    Returns:
        A :class:`ChangeProposal` for a *mapped* destructive capability
        (``move_run`` / ``delete_folder``); ``None`` for a read-only capability
        or a destructive one this mapping does not cover (e.g. ``rehome_asset``,
        whose complex source/target refs a flat map cannot express — the
        deterministic front-end builds that proposal directly).

    Raises:
        ValueError: *capability_id* is not a known curation capability (loud, no
            silent fallback).
    """
    if not _is_known_curation_cap(capability_id):
        raise ValueError(f"{capability_id!r} is not a known curation capability")

    if capability_id == _MOVE_RUN:
        run = references["run"]
        target = references["target_experiment"]
        affected = [ObjectRef(kind="run", id=run), ObjectRef(kind="experiment", id=target)]
        return ChangeProposal(
            id=_proposal_id(capability_id, references),
            intent=f"move run {run} to experiment {target}",
            current_state=StateSnapshot(objects=affected),
            proposed_change=ChangeSpec(
                op="asset_move",
                summary=f"move_run {run} -> {target}",
                payload={
                    "curation_op": "move_run",
                    "target_experiment": {"kind": "experiment", "id": target},
                },
            ),
            affected_objects=affected,
            expected_benefit="relocate the run to the target experiment",
            reversibility="reversible",
            approval_level=approval_level_for("asset_move", "reversible"),
        )

    if capability_id == _DELETE_FOLDER:
        folder = references["folder"]
        # The curation delete_folder capability names a run (flow.py resolves a
        # "folder" reference as experiment.get_run(ref)), so the target is kind="run".
        affected = [ObjectRef(kind="run", id=folder)]
        return ChangeProposal(
            id=_proposal_id(capability_id, references),
            intent=f"delete folder {folder}",
            current_state=StateSnapshot(objects=affected),
            proposed_change=ChangeSpec(
                op="artifact_delete", summary=f"delete_folder {folder}", payload={}
            ),
            affected_objects=affected,
            expected_benefit="remove the folder and its contents",
            reversibility="irreversible",
            approval_level=approval_level_for("artifact_delete", "irreversible"),
        )

    # Known but read-only, or a destructive cap this mapping does not cover.
    return None


def _build_ctx(workspace: Workspace, run: Run) -> HarnessRunContext:
    """Build a harness ctx: audit stores under the run dir, but the REAL workspace root.

    ``workspace_root`` MUST be the real workspace root (not the run dir), because
    the curation handlers resolve every ``ObjectRef`` via
    ``resolve_object_ref(ctx.workspace_root, …)``.
    """
    run_dir = Path(str(run.run_dir))
    artifact_store = FileArtifactStore(root=run_dir / "artifacts")
    db_path = run_dir / "harness.sqlite"
    return HarnessRunContext(
        run_id=run.id,
        workspace_root=Path(workspace.resolve()),
        artifact_store=artifact_store,
        event_log=SQLiteEventLog(path=db_path),
        lineage_store=SQLiteArtifactLineageStore(path=db_path, artifact_store=artifact_store),
    )


async def run_curation_proposal(
    proposal: ChangeProposal,
    *,
    workspace: Workspace,
    run: Run,
    approve: Approver | None = None,
) -> ChangeProposal:
    """Gate + execute *proposal* through the P2.1 ChangeProposal stack (single gate).

    Args:
        proposal: The §8 change proposal to gate and (on grant) execute.
        workspace: The live workspace whose objects the proposal mutates.
        run: The content-addressed audit run whose dir hosts the artifacts + audit db.
        approve: The approver resolving the gate (default: the gate's auto-grant).

    Returns:
        A copy of *proposal* with ``execution_result`` filled — ``executed`` /
        ``failed`` on a grant, ``rejected`` on a denial.
    """
    ctx = _build_ctx(workspace, run)
    registry = ChangeActionRegistry()
    register_curation_handlers(registry)
    executor = ProposalExecutor(registry)
    return await gate_change_proposal(ctx, proposal, approve=approve, executor=executor)
