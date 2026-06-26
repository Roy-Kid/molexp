"""Policy + Approval orchestration for ``molexp.harness`` (Phase 6).

Two public functions for emitting approval requests:

- :func:`evaluate_approval_policy` — pure walk of a ``BoundWorkflow``
  (+ optional ``WorkflowIR``) against an ``ApprovalPolicy``. Returns the
  list of :class:`ApprovalRequest` instances the policy demands.
- :func:`make_final_report_approval_request` — companion helper for the
  one intent (``final_report``) that has no workflow signal.

Two event-log helpers for threading approvals into the existing
``HarnessEvent`` stream:

- :func:`record_approval_request` — append an ``approval_requested`` event.
- :func:`record_approval_decision` — append ``approval_granted`` or
  ``approval_rejected`` based on the decision.

Phase 6 ships only contract + pure evaluator + event-log helpers. The
interactive UX (CLI prompts, async wait) and ``PathPolicy`` /
``ToolPolicy`` runtime enforcement defer to Phase 7+.
"""

from __future__ import annotations

from molexp.harness.policy.evaluate import ApprovalPolicyEvaluator
from molexp.harness.policy.event_log import ApprovalEventRecorder
from molexp.harness.policy.side_effect_gate import (
    enforce_side_effect_approvals,
    make_side_effect_approval_requests,
)

__all__ = [
    "ApprovalEventRecorder",
    "ApprovalPolicyEvaluator",
    "enforce_side_effect_approvals",
    "make_side_effect_approval_requests",
]
