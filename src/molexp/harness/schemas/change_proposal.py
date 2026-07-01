"""``ChangeProposal`` + ``AgentIntent`` — the high-risk-mutation coordination record.

integration.md §8. A :class:`ChangeProposal` is the first-class, reviewable record
every high-risk mutation (§8.1) must become **before any executor runs** — "the
agent proposes, the harness disposes." It is both a ``change_proposal``
:data:`~molexp.harness.schemas.artifact.ArtifactKind` (an open ``str``, so it lives
in the lineage spine) and — at the server layer — a durable record under an
``AgentTask``.

This module owns only the frozen schemas. The high-risk classifier and the dry,
audited gate live in :mod:`molexp.harness.change_proposal_gate`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from molexp.workspace.knowledge_item import SourceRef

CHANGE_PROPOSAL_KIND = "change_proposal"
"""The ``ArtifactKind`` a stored ChangeProposal is filed under (open ``str``)."""

HighRiskOp = Literal[
    "workflow_change",
    "param_space_change",
    "artifact_delete",
    "knowledge_change",
    "run_lifecycle",
    "asset_move",
    "api_schema_change",
]
"""The §8.1 high-risk-mutation set — each requires a :class:`ChangeProposal`."""

ApprovalLevel = Literal["auto", "user", "elevated"]
Reversibility = Literal["reversible", "partially", "irreversible"]
ProposalStatus = Literal["proposed", "granted", "rejected", "executed", "failed"]


class ObjectRef(BaseModel):
    """A typed pointer to a canonical object a proposal affects."""

    model_config = ConfigDict(frozen=True)

    kind: str
    id: str


class StateSnapshot(BaseModel):
    """Refs to the current versions of the objects a proposal would change."""

    model_config = ConfigDict(frozen=True)

    objects: list[ObjectRef] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict)


class ChangeSpec(BaseModel):
    """The proposed change — a skeleton typed diff (op + summary + payload).

    Per-``HighRiskOp`` typed variants (workflow IR diff / param-space delta /
    knowledge merge) land with their executors (P2.1); this carries a generic
    ``payload`` for now.
    """

    model_config = ConfigDict(frozen=True)

    op: HighRiskOp
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ProposalOutcome(BaseModel):
    """What happened to a proposal after the gate decided (filled post-gate)."""

    model_config = ConfigDict(frozen=True)

    status: ProposalStatus
    decided_by: str
    decided_at: datetime
    reason: str | None = None
    result_artifact_ids: list[str] = Field(default_factory=list)


class AgentIntent(BaseModel):
    """A structured user/agent request — the head of the coordination loop."""

    model_config = ConfigDict(frozen=True)

    id: str
    intent_text: str
    actor: str
    mode: str | None = None
    created_at: datetime


class ChangeProposal(BaseModel):
    """A reviewable proposed change to high-risk canonical state (§8.2 shape)."""

    model_config = ConfigDict(frozen=True)

    id: str
    intent: str
    current_state: StateSnapshot
    proposed_change: ChangeSpec
    affected_objects: list[ObjectRef]
    expected_benefit: str
    risks: list[str] = Field(default_factory=list)
    reversibility: Reversibility
    approval_level: ApprovalLevel
    evidence: list[str] = Field(default_factory=list)
    knowledge: list[SourceRef] = Field(default_factory=list)
    execution_result: ProposalOutcome | None = None


__all__ = [
    "CHANGE_PROPOSAL_KIND",
    "AgentIntent",
    "ApprovalLevel",
    "ChangeProposal",
    "ChangeSpec",
    "HighRiskOp",
    "ObjectRef",
    "ProposalOutcome",
    "ProposalStatus",
    "Reversibility",
    "StateSnapshot",
]
