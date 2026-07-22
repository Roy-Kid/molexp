"""Content-addressed freezing of an experiment spec + task board.

An :class:`ExperimentPlan` bundles the (opaque) experiment ``spec`` with the
current :class:`TaskBoard`. Freezing writes the plan (or a raw spec dict)
through a :class:`~molexp.harness.store.file_artifact_store.FileArtifactStore`
via ``put_json`` — the store is content-addressed, so identical content yields
the same :class:`~molexp.harness.schemas.PlanArtifactRef` id + ``sha256``
(idempotent), while any change to the spec or board flips the id.

Determinism comes entirely from the store: a stable ``model_dump`` / dict input
produces stable bytes, which ``compute_content_hash`` addresses. This module
adds no separate in-memory hash.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from molexp.harness.plan.task_board import TaskBoard

if TYPE_CHECKING:
    from molexp.harness.schemas import PlanArtifactRef
    from molexp.harness.store.file_artifact_store import FileArtifactStore

__all__ = [
    "FROZEN_PLAN_KIND",
    "FROZEN_SPEC_KIND",
    "ExperimentPlan",
    "freeze_experiment_plan",
    "freeze_spec",
]


FROZEN_SPEC_KIND = "frozen_experiment_spec"
"""Artifact kind for a frozen raw experiment-spec dict."""

FROZEN_PLAN_KIND = "frozen_experiment_plan"
"""Artifact kind for a frozen :class:`ExperimentPlan` (spec + board)."""


class ExperimentPlan(BaseModel):
    """A frozen experiment spec paired with its task board.

    The ``spec`` is deliberately opaque (``dict[str, Any]``) — the plan layer
    does not interpret its shape. ``extra="forbid"`` rejects any field beyond
    ``spec`` / ``board`` so the frozen artifact's shape stays exact.

    Attributes:
        spec: The opaque experiment spec.
        board: The task board captured alongside the spec.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    spec: dict[str, Any]
    board: TaskBoard


def freeze_spec(
    spec: dict[str, Any],
    store: FileArtifactStore,
    *,
    created_by: str,
    parent_ids: tuple[str, ...] = (),
) -> PlanArtifactRef:
    """Freeze a raw spec dict into the content-addressed artifact store.

    Args:
        spec: The opaque spec dict to freeze.
        store: The artifact store to write through.
        created_by: Producer id recorded on the artifact ref.
        parent_ids: Lineage parents (converted to a list for ``put_json``).

    Returns:
        The :class:`PlanArtifactRef` for the frozen spec; identical content
        yields an identical id + ``sha256``.
    """
    return store.put_json(
        kind=FROZEN_SPEC_KIND,
        obj=spec,
        created_by=created_by,
        parent_ids=list(parent_ids),
    )


def freeze_experiment_plan(
    plan: ExperimentPlan,
    store: FileArtifactStore,
    *,
    created_by: str,
    parent_ids: tuple[str, ...] = (),
) -> PlanArtifactRef:
    """Freeze an :class:`ExperimentPlan` into the content-addressed store.

    The plan is serialized with ``model_dump(mode="json")`` so the bytes are
    deterministic; any change to the spec or board flips the resulting id.

    Args:
        plan: The experiment plan to freeze.
        store: The artifact store to write through.
        created_by: Producer id recorded on the artifact ref.
        parent_ids: Lineage parents (converted to a list for ``put_json``).

    Returns:
        The :class:`PlanArtifactRef` for the frozen plan; identical content
        yields an identical id + ``sha256``.
    """
    return store.put_json(
        kind=FROZEN_PLAN_KIND,
        obj=plan.model_dump(mode="json"),
        created_by=created_by,
        parent_ids=list(parent_ids),
    )
