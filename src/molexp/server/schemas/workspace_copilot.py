"""Response schema for ``GET /api/workspace/copilot`` — the Workspace Copilot summary.

A camelCase serialization wrapper over the harness-layer ``WorkspaceSummary``
(:mod:`molexp.harness.copilot`). Reuses the P0.2 ``*RefResponse`` models for the
shared identity shapes; adds a ``NextActionResponse`` for the advisory next-actions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from .workspace_context import (
    HealthFlagResponse,
    KnowledgeRefResponse,
    RunRefResponse,
    WorkspaceRefResponse,
)

if TYPE_CHECKING:
    from molexp.harness.copilot import NextAction, WorkspaceSummary


class NextActionResponse(BaseModel):
    kind: str
    target: str
    rationale: str
    requiresProposal: bool
    advisory: bool

    @classmethod
    def from_action(cls, action: NextAction) -> NextActionResponse:
        return cls(
            kind=action.kind,
            target=action.target,
            rationale=action.rationale,
            requiresProposal=action.requires_proposal,
            advisory=action.advisory,
        )


class WorkspaceSummaryResponse(BaseModel):
    """Camel-cased HTTP view of the read-only ``WorkspaceSummary``."""

    workspace: WorkspaceRefResponse
    headline: str
    counts: dict[str, int]
    failedRuns: list[RunRefResponse] = []
    runningRuns: list[RunRefResponse] = []
    healthFlags: list[HealthFlagResponse] = []
    openQuestions: list[KnowledgeRefResponse] = []
    relevantKnowledge: list[KnowledgeRefResponse] = []
    nextActions: list[NextActionResponse] = []

    @classmethod
    def from_summary(cls, summary: WorkspaceSummary) -> WorkspaceSummaryResponse:
        """Build the HTTP view from a harness :class:`WorkspaceSummary`."""
        return cls(
            workspace=WorkspaceRefResponse.from_ref(summary.workspace),
            headline=summary.headline,
            counts=dict(summary.counts),
            failedRuns=[RunRefResponse.from_ref(r) for r in summary.failed_runs],
            runningRuns=[RunRefResponse.from_ref(r) for r in summary.running_runs],
            healthFlags=[HealthFlagResponse.from_flag(h) for h in summary.health_flags],
            openQuestions=[KnowledgeRefResponse.from_ref(k) for k in summary.open_questions],
            relevantKnowledge=[
                KnowledgeRefResponse.from_ref(k) for k in summary.relevant_knowledge
            ],
            nextActions=[NextActionResponse.from_action(a) for a in summary.next_actions],
        )
