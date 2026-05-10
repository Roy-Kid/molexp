"""Workflow-orthogonal policies for the agent layer.

Lives parallel to :mod:`molexp.agent.mode` because policies are NOT
mode-specific concepts: any workflow-bearing mode (``PlanMode`` today;
future workflow-driven modes tomorrow) consumes the same approval-gate
hook to decide "ship the materialized plan" vs "request another
iteration". Putting these types under a single mode's subpackage would
force duplication or upward imports as soon as a second mode lands.

Concrete view / decision types stay where the consuming mode owns them
(e.g. :class:`~molexp.agent.modes.plan.schemas.PlanReviewView` /
:class:`~molexp.agent.modes.plan.schemas.ApprovalDecision`); the
protocol here is parameterized via :pydata:`TView` / :pydata:`TDecision`
so each mode binds its own pair.

The shipped concrete here is :class:`AutoApproveGatePolicy` — the safe
non-interactive default that returns the constant decision the caller
hands it. Concrete *interactive* gates ship under the consuming mode's
subpackage (e.g.
:class:`~molexp.agent.modes.plan.gates.PromptGatePolicy` for PlanMode)
because they bind the protocol to a specific view / decision pair. Per
``feedback_orthogonal_policies.md``, hot-swap belongs to the lifecycle
owner (e.g. ``PlanMode.set_gate_policy``); :func:`static_gate_policy_lookup`
just adapts a fixed policy into the ``Callable[[], GatePolicy]``
shape that frozen deps containers thread through their workflow.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Protocol, TypeVar, runtime_checkable

__all__ = [
    "AutoApproveGatePolicy",
    "GatePolicy",
    "static_gate_policy_lookup",
]


TView = TypeVar("TView")
TDecision = TypeVar("TDecision")


@runtime_checkable
class GatePolicy(Protocol[TView, TDecision]):
    """Approval gate consulted at a workflow's human-checkpoint node.

    A single method — :meth:`human_review` — accepts the mode-supplied
    review view and returns the mode-supplied decision payload. The
    workflow-orthogonal protocol lives here; concrete view / decision
    types are owned by each mode that uses the gate (PlanMode pairs
    :class:`~molexp.agent.modes.plan.schemas.PlanReviewView` with
    :class:`~molexp.agent.modes.plan.schemas.ApprovalDecision`).
    """

    async def human_review(self, view: TView) -> TDecision: ...


class AutoApproveGatePolicy(Generic[TView, TDecision]):
    """Always-return-the-given-decision gate — the safe non-interactive default.

    Construct with the mode's "approved" decision instance. The view is
    discarded; every call returns the same decision::

        gate = AutoApproveGatePolicy(ApprovalDecision(approved=True))

    Holding a singleton decision (instead of building one per call)
    keeps the policy stateless from the caller's perspective.
    """

    def __init__(self, decision: TDecision) -> None:
        self._decision = decision

    async def human_review(self, view: TView) -> TDecision:
        del view  # protocol-pinned name; not consulted by auto-approve
        return self._decision


def static_gate_policy_lookup(
    policy: GatePolicy[TView, TDecision],
) -> Callable[[], GatePolicy[TView, TDecision]]:
    """Wrap a fixed :class:`GatePolicy` as a no-arg lookup callable.

    Tests, scripts, and any caller that does not need mid-run policy
    swapping can use this to satisfy the
    ``Callable[[], GatePolicy]`` field that frozen deps containers
    thread through their workflow without writing a lambda. The
    closure captures ``policy`` once and returns the same instance on
    every call, so consumers get singleton semantics.
    """

    def _lookup() -> GatePolicy[TView, TDecision]:
        return policy

    return _lookup
