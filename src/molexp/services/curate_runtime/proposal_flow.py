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

from molexp.harness.actions import (
    ChangeActionRegistry,
    ProposalExecutor,
    register_curation_handlers,
)
from molexp.harness.actions.handlers import register_lifecycle_handlers
from molexp.harness.capabilities import curation_capabilities, lifecycle_capabilities
from molexp.harness.change_proposal_gate import approval_level_for, gate_change_proposal
from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.schemas import ChangeProposal, ChangeSpec, ObjectRef, StateSnapshot
from molexp.harness.schemas.change_proposal import Reversibility
from molexp.harness.store.approval_store import SQLiteApprovalStore
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

if TYPE_CHECKING:
    from molexp.harness.stages.approval_gate import Approver
    from molexp.workspace import Run, Workspace

__all__ = [
    "build_curation_proposal",
    "curation_invocation_to_proposal",
    "run_curation_proposal",
]

_MOVE_RUN = "molexp.curation.move_run"
_DELETE_FOLDER = "molexp.curation.delete_folder"
_REHOME_ASSET = "molexp.curation.rehome_asset"


def _proposal_id(capability_id: str, references: dict[str, str]) -> str:
    """A stable content-addressed proposal id (same invocation → same id)."""
    payload = json.dumps(
        {"capability_id": capability_id, "references": dict(sorted(references.items()))},
        sort_keys=True,
    )
    return f"cp-curate-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _is_known_curation_cap(capability_id: str) -> bool:
    known = (*curation_capabilities(), *lifecycle_capabilities())
    return any(cap.id == capability_id for cap in known)


def _move_run_proposal(run: str, target: str) -> ChangeProposal:
    affected = [ObjectRef(kind="run", id=run), ObjectRef(kind="experiment", id=target)]
    return ChangeProposal(
        id=_proposal_id("move_run", {"run": run, "target_experiment": target}),
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


def _delete_folder_proposal(folder: str) -> ChangeProposal:
    # The curation delete_folder capability names a run (flow.py resolves a
    # "folder" reference as experiment.get_run(ref)), so the target is kind="run".
    affected = [ObjectRef(kind="run", id=folder)]
    return ChangeProposal(
        id=_proposal_id("delete_folder", {"folder": folder}),
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


def _rehome_asset_proposal(
    asset: str, source: dict[str, str], target: dict[str, str], action: str
) -> ChangeProposal:
    reversibility = "reversible" if action == "copy" else "partially"
    affected = [
        ObjectRef(kind="data_asset", id=asset),
        ObjectRef(kind=source["kind"], id=source["id"]),
        ObjectRef(kind=target["kind"], id=target["id"]),
    ]
    return ChangeProposal(
        id=_proposal_id(
            "rehome_asset", {"asset": asset, "action": action, **_scope_key(source, target)}
        ),
        intent=f"re-home asset {asset} ({action})",
        current_state=StateSnapshot(objects=affected),
        proposed_change=ChangeSpec(
            op="asset_move",
            summary=f"rehome_asset {asset} ({action})",
            payload={
                "curation_op": "rehome_asset",
                "source": dict(source),
                "target": dict(target),
                "action": action,
            },
        ),
        affected_objects=affected,
        expected_benefit="re-home the asset payload into the target scope",
        reversibility=reversibility,
        approval_level=approval_level_for("asset_move", reversibility),
    )


def _scope_key(source: dict[str, str], target: dict[str, str]) -> dict[str, str]:
    return {
        "source": f"{source['kind']}:{source['id']}",
        "target": f"{target['kind']}:{target['id']}",
    }


_LIFECYCLE_VERBS = {
    "molexp.lifecycle.run_execute": "execute",
    "molexp.lifecycle.run_resume": "resume",
    "molexp.lifecycle.run_rerun": "rerun",
    "molexp.lifecycle.run_cancel": "cancel",
    "molexp.lifecycle.runs_prune": "prune",
}

#: Reversibility per lifecycle verb (spec table): prune is irreversible,
#: cancel is reversible (the run stays resumable), the execute family is
#: partially reversible (attempts persist; outputs may be overwritten).
_LIFECYCLE_REVERSIBILITY: dict[str, Reversibility] = {
    "execute": "partially",
    "resume": "partially",
    "rerun": "partially",
    "cancel": "reversible",
    "prune": "irreversible",
}


def _parse_scope_ref(ref: str) -> dict[str, str]:
    """Parse a colon-encoded scope ref (``"experiment:exp-a"``) into a dict.

    The NL planner's flat ``references`` map cannot carry nested objects, so
    ``rehome_asset``'s scope refs ride as ``"<kind>:<id>"`` strings.
    """
    kind, sep, ident = ref.partition(":")
    if not sep or not kind or not ident:
        raise ValueError(
            f"scope ref {ref!r} must be colon-encoded '<kind>:<id>' "
            "(e.g. 'experiment:exp-a', 'project:demo')"
        )
    return {"kind": kind, "id": ident}


def _lifecycle_proposal(capability_id: str, references: dict[str, str]) -> ChangeProposal:
    """Build the ``run_lifecycle`` proposal for one of the five verbs."""
    verb = _LIFECYCLE_VERBS[capability_id]
    run_id = references["run"]
    reversibility = _LIFECYCLE_REVERSIBILITY[verb]
    affected = [ObjectRef(kind="run", id=run_id)]
    payload: dict[str, object] = {"lifecycle_verb": verb}
    if verb == "rerun" and "fresh" in references:
        payload["fresh"] = references["fresh"].lower() == "true"
    if verb == "prune":
        if "execution_ids" in references:
            payload["execution_ids"] = [
                item for item in references["execution_ids"].split(",") if item
            ]
        if "statuses" in references:
            payload["statuses"] = [item for item in references["statuses"].split(",") if item]
    return ChangeProposal(
        id=_proposal_id(capability_id, references),
        intent=f"{verb} run {run_id}",
        current_state=StateSnapshot(objects=affected),
        proposed_change=ChangeSpec(
            op="run_lifecycle",
            summary=f"{verb} run {run_id}",
            payload=payload,
        ),
        affected_objects=affected,
        expected_benefit=f"run lifecycle intervention: {verb}",
        reversibility=reversibility,
        approval_level=approval_level_for("run_lifecycle", reversibility),
    )


def curation_invocation_to_proposal(
    capability_id: str, references: dict[str, str]
) -> ChangeProposal | None:
    """Map a planner-shaped curation invocation onto a §8 :class:`ChangeProposal`.

    Args:
        capability_id: A ``molexp.curation.*`` capability id.
        references: The planner's flat JSON reference handles (ids / slugs).

    Returns:
        A :class:`ChangeProposal` for a *mapped* destructive capability
        (``move_run`` / ``delete_folder`` / ``rehome_asset`` via colon-encoded
        scope refs / the five ``molexp.lifecycle.*`` verbs); ``None`` only for
        a read-only capability (the in-process invocation path handles those).

    Raises:
        ValueError: *capability_id* is not a known curation capability (loud, no
            silent fallback).
    """
    if not _is_known_curation_cap(capability_id):
        raise ValueError(f"{capability_id!r} is not a known curation capability")
    if capability_id == _MOVE_RUN:
        return _move_run_proposal(references["run"], references["target_experiment"])
    if capability_id == _DELETE_FOLDER:
        return _delete_folder_proposal(references["folder"])
    if capability_id == _REHOME_ASSET:
        # Colon-encoded scope refs close the flat-map gap (vision-loop-07):
        # source: "experiment:exp-a", target: "project:demo".
        return _rehome_asset_proposal(
            references["asset"],
            _parse_scope_ref(references["source"]),
            _parse_scope_ref(references["target"]),
            references.get("action", "copy"),
        )
    if capability_id in _LIFECYCLE_VERBS:
        return _lifecycle_proposal(capability_id, references)
    # Known but read-only — the in-process invocation path handles it.
    return None


def build_curation_proposal(
    op: str,
    *,
    run: str | None = None,
    target_experiment: str | None = None,
    folder: str | None = None,
    asset: str | None = None,
    source: dict[str, str] | None = None,
    target: dict[str, str] | None = None,
    action: str = "copy",
) -> ChangeProposal:
    """Build a §8 :class:`ChangeProposal` from explicit typed args (LLM-free path).

    The deterministic peer of :func:`curation_invocation_to_proposal`: the operator
    (CLI flags / route body) names the op + its targets directly, so ``rehome_asset``
    is supported here (unlike the flat-reference LLM path).

    Args:
        op: One of ``"move_run"`` / ``"delete_folder"`` / ``"rehome_asset"``.
        run: The run id (``move_run``).
        target_experiment: The destination experiment id (``move_run``).
        folder: The folder/run id to delete (``delete_folder``).
        asset: The data-asset id (``rehome_asset``).
        source: The source scope ref ``{"kind", "id"}`` (``rehome_asset``).
        target: The target scope ref ``{"kind", "id"}`` (``rehome_asset``).
        action: ``"copy"`` (default) or ``"move"`` (``rehome_asset``).

    Returns:
        The built :class:`ChangeProposal`.

    Raises:
        ValueError: An unknown *op*, or a required arg missing for the chosen op.
    """
    if op == "move_run":
        if not run or not target_experiment:
            raise ValueError("move_run requires 'run' and 'target_experiment'")
        return _move_run_proposal(run, target_experiment)
    if op == "delete_folder":
        if not folder:
            raise ValueError("delete_folder requires 'folder'")
        return _delete_folder_proposal(folder)
    if op == "rehome_asset":
        if not asset or not source or not target:
            raise ValueError("rehome_asset requires 'asset', 'source', and 'target'")
        return _rehome_asset_proposal(asset, source, target, action)
    raise ValueError(f"unknown curation op {op!r}")


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
        approval_store=SQLiteApprovalStore(path=db_path),
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
    register_lifecycle_handlers(registry)
    executor = ProposalExecutor(registry)
    return await gate_change_proposal(ctx, proposal, approve=approve, executor=executor)
