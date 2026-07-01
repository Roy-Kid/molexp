"""Curation :class:`ChangeActionHandler`s — the first real guarded-execution handlers.

integration.md §8.1 (the destructive ``workspace/curation`` ops), §8.3. Two
handlers bind onto the slice-01 dispatch spine and, for a *granted* proposal,
perform a curation mutation **in-process** over
:mod:`molexp.workspace.curation.reorg` — the curation ops carry live
``Run`` / ``Folder`` / ``DataAsset`` objects that cannot cross the
``InvokeCapability`` JSON-subprocess boundary, and they are pure local-filesystem
moves/deletes, so no subprocess isolation is used.

Both handlers stay bounded to the proposal's ``affected_objects`` (via
:func:`~molexp.harness.actions.proposal_executor.assert_within_affected_scope`),
persist a ``proposal_action_result`` artifact, and return
``ProposalOutcome(status="executed", …)``. A reorg that raises is not caught here
— it propagates to the :class:`ProposalExecutor`, which records
``status="failed"`` (§8.3: recorded, not swallowed).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from molexp.harness.actions.proposal_executor import assert_within_affected_scope
from molexp.harness.actions.resolve import resolve_object_ref
from molexp.harness.schemas.change_proposal import ObjectRef, ProposalOutcome
from molexp.workspace.curation import delete_folder, move_run, rehome_asset

if TYPE_CHECKING:
    from molexp.harness.actions.proposal_executor import ChangeActionRegistry
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.schemas.change_proposal import ChangeProposal
    from molexp.workspace.assets.data import DataAsset
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.folder import Folder
    from molexp.workspace.run import Run

__all__ = ["ArtifactDeleteHandler", "AssetMoveHandler", "register_curation_handlers"]


def _sole(proposal: ChangeProposal, kind: str) -> ObjectRef:
    """Return the single ``affected_objects`` ref of *kind* (else raise loudly)."""
    matches = [o for o in proposal.affected_objects if o.kind == kind]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {kind!r} in affected_objects, got {len(matches)} "
            f"(proposal {proposal.id})"
        )
    return matches[0]


def _sole_of_kinds(proposal: ChangeProposal, kinds: tuple[str, ...]) -> ObjectRef:
    """Return the single ``affected_objects`` ref whose kind is in *kinds* (else raise)."""
    matches = [o for o in proposal.affected_objects if o.kind in kinds]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one of {kinds} in affected_objects, got {len(matches)} "
            f"(proposal {proposal.id})"
        )
    return matches[0]


def _record(
    ctx: HarnessRunContext, proposal: ChangeProposal, created_by: str, detail: dict
) -> ProposalOutcome:
    """Persist a ``proposal_action_result`` artifact and build the executed outcome."""
    ref = ctx.artifact_store.put_json(
        kind="proposal_action_result",
        obj={"proposal_id": proposal.id, "op": proposal.proposed_change.op, **detail},
        created_by=created_by,
        parent_ids=list(proposal.evidence),
    )
    return ProposalOutcome(
        status="executed",
        decided_by=created_by,
        decided_at=datetime.now(tz=UTC),
        result_artifact_ids=[ref.id],
    )


class AssetMoveHandler:
    """Handle the ``asset_move`` op — ``move_run`` or ``rehome_asset``.

    The concrete curation op is selected by ``proposed_change.payload["curation_op"]``.
    """

    async def apply(self, ctx: HarnessRunContext, proposal: ChangeProposal) -> ProposalOutcome:
        payload = proposal.proposed_change.payload
        curation_op = payload.get("curation_op")

        if curation_op == "move_run":
            run_ref = _sole(proposal, "run")
            target_ref = ObjectRef(**payload["target_experiment"])
            assert_within_affected_scope(proposal, [run_ref, target_ref])
            run = cast("Run", resolve_object_ref(ctx.workspace_root, run_ref))
            target = cast("Experiment", resolve_object_ref(ctx.workspace_root, target_ref))
            move_run(run, target)
            return _record(
                ctx,
                proposal,
                "asset_move.move_run",
                {"run": run_ref.id, "target_experiment": target_ref.id},
            )

        if curation_op == "rehome_asset":
            asset_ref = _sole(proposal, "data_asset")
            source_ref = ObjectRef(**payload["source"])
            target_ref = ObjectRef(**payload["target"])
            assert_within_affected_scope(proposal, [asset_ref, source_ref, target_ref])
            asset = cast("DataAsset", resolve_object_ref(ctx.workspace_root, asset_ref))
            source = resolve_object_ref(ctx.workspace_root, source_ref)
            target = cast("Experiment", resolve_object_ref(ctx.workspace_root, target_ref))
            rehomed = rehome_asset(
                asset,
                source=source,
                target=target,
                action=payload.get("action", "copy"),
            )
            return _record(
                ctx,
                proposal,
                "asset_move.rehome_asset",
                {"data_asset": asset_ref.id, "rehomed_asset": rehomed.asset_id},
            )

        raise ValueError(
            f"asset_move: unknown curation_op {curation_op!r} (proposal {proposal.id})"
        )


class ArtifactDeleteHandler:
    """Handle the ``artifact_delete`` op — ``delete_folder`` on the bound folder.

    The affected object may be any deletable ``Folder`` — a ``run``,
    ``experiment``, or generic ``folder`` (all resolve to ``Folder`` subclasses,
    and ``delete_folder`` operates on any of them). Curation's ``delete_folder``
    capability names a run, so ``run`` is included.
    """

    async def apply(self, ctx: HarnessRunContext, proposal: ChangeProposal) -> ProposalOutcome:
        folder_ref = _sole_of_kinds(proposal, ("run", "experiment", "folder"))
        assert_within_affected_scope(proposal, [folder_ref])
        folder = cast("Folder", resolve_object_ref(ctx.workspace_root, folder_ref))
        delete_folder(folder)
        return _record(ctx, proposal, "artifact_delete", {"folder": folder_ref.id})


def register_curation_handlers(registry: ChangeActionRegistry) -> None:
    """Bind the ``asset_move`` and ``artifact_delete`` handlers onto *registry*."""
    registry.register("asset_move", AssetMoveHandler())
    registry.register("artifact_delete", ArtifactDeleteHandler())
