"""``molexp.harness.actions`` — the coordination-layer "Act" seam.

Where a *granted* :class:`~molexp.harness.schemas.change_proposal.ChangeProposal`
is dispatched to actually perform its high-risk mutation (integration.md §6.1
guarded execution, §8.3 lifecycle). Slice 01 ships the reusable dispatch spine:
the ``HighRiskOp`` → handler registry, the :class:`ProposalExecutor`, the
``affected_objects`` binding guard, and the :class:`ProposalActionRecorder`.
Concrete handlers (curation) and the gate wiring land in later slices.
"""

from __future__ import annotations

from molexp.harness.actions.proposal_executor import (
    ChangeActionHandler,
    ChangeActionRegistry,
    ProposalExecutor,
    assert_within_affected_scope,
)
from molexp.harness.actions.recorder import ProposalActionRecorder

__all__ = [
    "ChangeActionHandler",
    "ChangeActionRegistry",
    "ProposalActionRecorder",
    "ProposalExecutor",
    "assert_within_affected_scope",
]
