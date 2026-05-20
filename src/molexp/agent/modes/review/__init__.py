"""ReviewMode — read-only typed review against the shared IntentSpec contract.

The fifth mode (sub-spec 06). ReviewMode reviews an *existing* artefact
— a typed :class:`~molexp.agent.modes._planning.PlanGraph`, a
materialized workspace, or a completed run — against the shared typed
contracts, runs three pure conformance checks, and renders a structured
:class:`ReviewVerdict`. It is strictly read-only: it produces a verdict,
never edits code, never re-runs anything. When changes are needed the
verdict carries a :class:`~molexp.agent.modes._planning.PlanDiff` that
feeds the shared repair loop.

Public surface — import from this package root:

- :class:`ReviewMode` / :class:`ReviewModeConfig` — the mode + tunables.
- :class:`ReviewTarget` / :class:`ReviewTargetKind` — the typed
  reference to the artefact under review.
- :class:`StepFinding` / :class:`ReviewVerdict` — the typed verdict
  shape; :func:`build_review_verdict` is the pure fold.
- :func:`check_intent_conformance` / :func:`check_capability_evidence` /
  :func:`check_lifecycle_consistency` — the three pure checkers.
- :class:`ReviewVerdictFolder` — the read-only mode's only write surface.
"""

from molexp.agent.modes.review._mode import ReviewMode, ReviewModeConfig
from molexp.agent.modes.review.checks import (
    check_capability_evidence,
    check_intent_conformance,
    check_lifecycle_consistency,
)
from molexp.agent.modes.review.target import (
    ReviewTarget,
    ReviewTargetKind,
    detect_review_target,
)
from molexp.agent.modes.review.verdict import (
    ReviewVerdict,
    StepFinding,
    build_review_verdict,
)
from molexp.agent.modes.review.verdict_folder import (
    AGENT_REVIEW_KIND,
    ReviewVerdictFolder,
)

__all__ = [
    "AGENT_REVIEW_KIND",
    "ReviewMode",
    "ReviewModeConfig",
    "ReviewTarget",
    "ReviewTargetKind",
    "ReviewVerdict",
    "ReviewVerdictFolder",
    "StepFinding",
    "build_review_verdict",
    "check_capability_evidence",
    "check_intent_conformance",
    "check_lifecycle_consistency",
    "detect_review_target",
]
