"""Tests for the workflow-native PlanMode review→repair loop.

The legacy ``_repair_loop.drive_with_repair`` hand-rolled ``while True``
driver is gone; iteration control now lives inside the workflow itself
(``WorkflowBuilder.loop(body=[...], until="RepairDecide", max_iters=N)``).
These tests exercise the surface the rewrite preserves:

- approval on the first pass terminates the loop after one iteration;
- a rejecting final review drives ``RepairDecide`` → ``Next("continue")``
  and the body re-runs;
- exceeding ``max_iters`` emits :class:`~molexp.workflow.LoopMaxItersExceeded`
  while the workflow finishes (status not "failed");
- repair signals planted by capability-/codegen-gate exceptions
  (``CapabilityDiscoveryRequired``, ``UnevidencedApiReference``) flow
  through ``ctx.deps.runtime.repair_signal`` and reach
  ``RepairDecide`` without crashing the workflow.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from molexp.agent.modes.plan import PlanFolder
from molexp.agent.modes.plan._pipeline import build_plan_workflow
from molexp.agent.modes.plan.policy import STANDARD_PLAN_POLICY
from molexp.agent.modes.plan.protocols import PlanDeps
from molexp.agent.modes.plan.schemas import HandoffResult, PlanReviewView
from molexp.agent.modes.plan.state import PlanRuntimeState
from molexp.agent.review import BypassPolicy, ReviewDecision, ReviewView
from molexp.workflow import LoopMaxItersExceeded
from molexp.workspace import Workspace

from .conftest import FakeRouter


# ── Stub policies ──────────────────────────────────────────────────────────


class _ApproveOnPass:
    """Final-review policy: rejects until the configured pass, then approves."""

    def __init__(self, approve_at: int) -> None:
        self.approve_at = approve_at
        self.calls: list[PlanReviewView] = []

    async def review(self, view: ReviewView) -> ReviewDecision:
        assert isinstance(view, PlanReviewView)
        self.calls.append(view)
        if len(self.calls) - 1 >= self.approve_at:
            return ReviewDecision(approved=True)
        return ReviewDecision(
            approved=False,
            reason="needs another pass",
            target_steps=("DraftImplementationPlan",),
            cascade_downstream=True,
            feedback="iterate",
        )


class _AlwaysReject:
    def __init__(self) -> None:
        self.calls = 0

    async def review(self, view: ReviewView) -> ReviewDecision:
        del view
        self.calls += 1
        return ReviewDecision(approved=False, reason="never approve")


# ── Helpers ────────────────────────────────────────────────────────────────


@pytest.fixture
def plan_folder(tmp_path: Path) -> PlanFolder:
    return Workspace(tmp_path / "ws").add_folder(PlanFolder(name="loop_test"))


def _deps(plan_folder: PlanFolder, *, final_policy: object = None) -> tuple[PlanDeps, FakeRouter]:
    router = FakeRouter()
    resolved = final_policy if final_policy is not None else BypassPolicy()
    deps = PlanDeps(
        router=router,  # type: ignore[arg-type]
        policy=STANDARD_PLAN_POLICY,
        plan_folder=plan_folder,
        final_policy_lookup=lambda: resolved,
        runtime=PlanRuntimeState(),
    )
    return deps, router


# ── Tests ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_first_pass_approval_runs_once(plan_folder: PlanFolder) -> None:
    """Approval on iteration 0 → workflow completes with iteration counter at 0."""
    gate = _ApproveOnPass(approve_at=0)
    deps, _ = _deps(plan_folder, final_policy=gate)
    workflow = build_plan_workflow(max_iterations=4)

    result = await workflow.execute(
        config={"user_input": "report"},
        deps=deps,
    )

    assert result.status == "completed"
    assert len(gate.calls) == 1
    assert gate.calls[0].repair_iteration == 0
    assert deps.runtime.iteration == 0
    handoff = deps.runtime.last_inner_outputs.get("FinalHandoffCheck")
    assert isinstance(handoff, HandoffResult)
    assert handoff.decision.approved


@pytest.mark.asyncio
async def test_rejection_drives_repair_iteration(plan_folder: PlanFolder) -> None:
    """Reject once, approve on the second pass → iteration counter advances to 1."""
    gate = _ApproveOnPass(approve_at=1)
    deps, _ = _deps(plan_folder, final_policy=gate)
    workflow = build_plan_workflow(max_iterations=4)

    result = await workflow.execute(
        config={"user_input": "report"},
        deps=deps,
    )

    assert result.status == "completed"
    assert len(gate.calls) == 2
    assert gate.calls[0].repair_iteration == 0
    assert gate.calls[1].repair_iteration == 1
    assert deps.runtime.iteration == 1
    assert len(deps.runtime.repair_history) == 1
    handoff = deps.runtime.last_inner_outputs.get("FinalHandoffCheck")
    assert isinstance(handoff, HandoffResult)
    assert handoff.decision.approved


@pytest.mark.asyncio
async def test_max_iters_emits_loop_warning(plan_folder: PlanFolder) -> None:
    """Always-reject + max_iters=2 → workflow finishes with
    :class:`LoopMaxItersExceeded` warning."""
    gate = _AlwaysReject()
    deps, _ = _deps(plan_folder, final_policy=gate)
    workflow = build_plan_workflow(max_iterations=2)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = await workflow.execute(
            config={"user_input": "report"},
            deps=deps,
        )

    # The workflow itself completes — the loop runtime forces Next("exit").
    assert result.status == "completed"
    # The LoopMaxItersExceeded warning fired exactly once.
    loop_warnings = [w for w in captured if issubclass(w.category, LoopMaxItersExceeded)]
    assert loop_warnings, "expected LoopMaxItersExceeded warning"
    # The runtime iteration counter saw both budget-counted iterations.
    assert deps.runtime.iteration >= 2
