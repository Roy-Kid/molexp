"""Side-effect approval gate for capability invocation.

Derivation glue only: it builds :class:`ApprovalRequest`\\ s from each
about-to-run item's declared ``side_effects`` and feeds them through the
**unchanged** :class:`~molexp.harness.stages.approval_gate.ApprovalGate`. No new
gate, no new ``Approver`` semantics, no schema change.

Two contracts:

* **Bypass** — an item with empty ``side_effects`` (read-only) is never gated;
  :func:`enforce_side_effect_approvals` runs no gate and persists no artifact for
  an all-read-only batch.
* **No fallback** — a destructive item whose request is denied aborts before the
  item's body runs: ``ApprovalGate`` raises
  :class:`~molexp.harness.errors.StageExecutionError`, which propagates out of
  :func:`enforce_side_effect_approvals` unchanged.

Each request reuses the existing ``overwrite`` :data:`ApprovalIntent` (the
closest destructive-write intent) so no new intent literal is introduced.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from molexp.harness.schemas import ApprovalRequest

if TYPE_CHECKING:
    from collections.abc import Sequence

    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.schemas import ArtifactRef
    from molexp.harness.stages.approval_gate import Approver

__all__ = ["enforce_side_effect_approvals", "make_side_effect_approval_requests"]


class _HasSideEffects(Protocol):
    """An item exposing an ``id`` and a declared ``side_effects`` set.

    Both :class:`~molexp.harness.schemas.capability.ToolCapability` and
    :class:`~molexp.harness.schemas.bound_workflow.BoundTask` satisfy this, so a
    caller may gate either.
    """

    id: str
    side_effects: list[str]


def make_side_effect_approval_requests(
    items: Sequence[_HasSideEffects],
) -> list[ApprovalRequest]:
    """Build one approval request per destructive item; nothing for read-only.

    Pure. For each item whose ``side_effects`` is non-empty, emit exactly one
    :class:`ApprovalRequest` carrying enough context (the item id and its
    deduplicated, sorted side-effect list) for a human or audit consumer to
    decide. Items with empty ``side_effects`` produce no request.

    Args:
        items: The about-to-run items to screen (capabilities or bound tasks).

    Returns:
        One ``ApprovalRequest`` per destructive item, in input order.
    """
    requests: list[ApprovalRequest] = []
    for item in items:
        if not item.side_effects:
            continue
        sorted_effects = sorted(set(item.side_effects))
        requests.append(
            ApprovalRequest(
                id=uuid.uuid4().hex[:12],
                intent="overwrite",
                reason=(
                    f"capability {item.id!r} declares destructive side effects "
                    f"{sorted_effects}; approval required before invocation"
                ),
                triggered_by_policy="side_effects_present",
                metadata={"capability_id": item.id, "side_effects": sorted_effects},
                created_at=datetime.now(UTC),
            )
        )
    return requests


async def enforce_side_effect_approvals(
    items: Sequence[_HasSideEffects],
    *,
    ctx: HarnessRunContext,
    approve: Approver | None = None,
    subject_artifact_ids: list[str] | None = None,
) -> ArtifactRef | None:
    """Gate the destructive items in *items* through ``ApprovalGate``.

    Builds the requests via :func:`make_side_effect_approval_requests`. With no
    destructive items the gate is skipped entirely — returns ``None``, runs no
    gate, persists no artifact (the read-only bypass). Otherwise runs an
    ``ApprovalGate`` (named ``side_effect_gate``, persisting a
    ``side_effect_approval`` summary so it never collides with a terminal gate's
    ``analysis_result``) and returns its summary ref. A denied request makes the
    gate raise :class:`StageExecutionError`, which propagates unchanged — the
    caller never reaches the item's body.

    Args:
        items: The about-to-run items to screen.
        ctx: The harness run context (event log + artifact store).
        approve: The approver callback; defaults to the gate's auto-grant.
        subject_artifact_ids: Optional ids to attribute the summary to.

    Returns:
        The ``side_effect_approval`` summary ref when a gate ran and granted, or
        ``None`` when there was nothing destructive to gate.
    """
    requests = make_side_effect_approval_requests(items)
    if not requests:
        return None
    # Lazy import: ``ApprovalGate`` lives in ``stages/``, which already imports
    # ``policy.event_log`` — a module-level import here would close a
    # policy <-> stages cycle. Importing at call time keeps the edge one-way.
    from molexp.harness.stages.approval_gate import ApprovalGate

    gate = ApprovalGate(
        requests,
        approve=approve,
        subject_artifact_ids=subject_artifact_ids,
        name="side_effect_gate",
        result_kind="side_effect_approval",
    )
    return await gate.run(ctx)
