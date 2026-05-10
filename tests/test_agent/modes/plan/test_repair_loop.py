"""Tests for ``_repair_loop.drive_with_repair`` — PlanMode review→repair driver.

Acceptance criteria covered:

- ``ac-002`` — :class:`PlanReviewView` carries iteration state across rounds.
- ``ac-007`` — ``GenerateTaskTests`` / ``GenerateTaskImplementations`` honour
  ``ctx.deps.repair_target_tasks`` (per-task LLM filter).
- ``ac-011`` — first-pass approval returns immediately without archiving.
- ``ac-012`` — rejection drives a partial-rerun shaped by the decision.
- ``ac-013`` — ``max_iterations`` cap surfaces ``RepairBudgetExceeded`` +
  rejected status.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.modes.plan import PlanWorkspaceHandle, RepairBudgetExceeded
from molexp.agent.modes.plan._repair_loop import drive_with_repair
from molexp.agent.modes.plan.protocols import PlanDeps, PlanGatePolicy
from molexp.agent.modes.plan.schemas import (
    ApprovalDecision,
    PlanReviewView,
)
from molexp.agent.policy import (
    AutoApproveGatePolicy,
    GatePolicy,
    static_gate_policy_lookup,
)
from molexp.workspace import Workspace

from .conftest import FakeProvider

# ── Stub gates used by the tests ───────────────────────────────────────────


class _ApproveOnPass(GatePolicy[PlanReviewView, ApprovalDecision]):
    """Records each ``human_review`` invocation; approves on the configured pass."""

    def __init__(self, approve_at: int) -> None:
        self.approve_at = approve_at
        self.calls: list[PlanReviewView] = []

    async def human_review(self, view: PlanReviewView) -> ApprovalDecision:
        self.calls.append(view)
        if len(self.calls) - 1 >= self.approve_at:
            return ApprovalDecision(approved=True)
        return ApprovalDecision(
            approved=False,
            reason="needs another pass",
            target_node_ids=("DraftImplementationPlan",),
            target_task_ids=("prepare",),
            cascade_downstream=True,
            feedback="iterate",
        )


class _AlwaysReject(GatePolicy[PlanReviewView, ApprovalDecision]):
    def __init__(self) -> None:
        self.calls = 0

    async def human_review(self, view: PlanReviewView) -> ApprovalDecision:
        del view
        self.calls += 1
        return ApprovalDecision(
            approved=False,
            reason="no",
            target_task_ids=("prepare",),
        )


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def repair_handle(tmp_path: Path) -> PlanWorkspaceHandle:
    return PlanWorkspaceHandle.materialize(Workspace(tmp_path / "ws"), plan_id="rep_loop")


def _build_deps(handle: PlanWorkspaceHandle, *, gate_policy: PlanGatePolicy | None = None) -> PlanDeps:
    from molexp.agent.modes.plan.policy import STANDARD_PLAN_POLICY

    resolved = (
        gate_policy
        if gate_policy is not None
        else AutoApproveGatePolicy(ApprovalDecision(approved=True))
    )
    return PlanDeps(
        router=FakeProvider(),  # type: ignore[arg-type]
        policy=STANDARD_PLAN_POLICY,
        workspace_handle=handle,
        gate_policy_lookup=static_gate_policy_lookup(resolved),
    )


# ── Tests ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_first_pass_approval(repair_handle: PlanWorkspaceHandle) -> None:
    """ac-011 — gate approves on iteration 0 → drive_with_repair runs PLAN_WORKFLOW
    exactly once and writes no archives."""
    gate = _ApproveOnPass(approve_at=0)
    deps = _build_deps(repair_handle, gate_policy=gate)

    result = await drive_with_repair(deps, "report", max_iterations=4)

    assert result.status == "completed"
    assert len(gate.calls) == 1
    assert gate.calls[0].repair_iteration == 0
    # No archive subdirectory exists — the loop never started a second round.
    repairs_root = repair_handle.root() / "repairs"
    if repairs_root.exists():
        assert list(repairs_root.glob("iter-*")) == []


@pytest.mark.asyncio
async def test_review_view_carries_iteration_state(
    repair_handle: PlanWorkspaceHandle,
) -> None:
    """ac-002 — second-iteration `PlanReviewView` carries `repair_iteration=1`."""
    gate = _ApproveOnPass(approve_at=1)
    deps = _build_deps(repair_handle, gate_policy=gate)

    result = await drive_with_repair(deps, "report", max_iterations=4)

    assert result.status == "completed"
    assert len(gate.calls) == 2
    assert gate.calls[0].repair_iteration == 0
    assert gate.calls[1].repair_iteration == 1


@pytest.mark.asyncio
async def test_partial_rerun_round(repair_handle: PlanWorkspaceHandle) -> None:
    """ac-012 — second iteration runs a subgraph shaped by the rejection.

    The first pass writes test_prepare.py, test_couple.py, test_isolate.py.
    The rejection asks for `target_task_ids=("prepare",)` so the second
    iteration MUST regenerate prepare's test/impl but reuse couple/isolate
    from the prior round (verified via mtime / content stability).
    """
    gate = _ApproveOnPass(approve_at=1)
    deps = _build_deps(repair_handle, gate_policy=gate)

    result = await drive_with_repair(deps, "report", max_iterations=4)

    assert result.status == "completed"
    # Archive of the first iteration was written.
    iter0 = repair_handle.repairs_dir(0)
    assert (iter0 / "tests" / "test_prepare.py").exists()
    assert (iter0 / "tests" / "test_couple.py").exists()
    # Manifest reflects one repair round.
    import yaml

    manifest_data = yaml.safe_load(repair_handle.manifest_path().read_text())
    assert manifest_data["repair_iterations"] == 1
    assert len(manifest_data["repair_history"]) == 1
    record = manifest_data["repair_history"][0]
    assert record["target_node_ids"] == ["DraftImplementationPlan"]
    assert record["target_task_ids"] == ["prepare"]
    assert record["cascade_downstream"] is True


@pytest.mark.asyncio
async def test_per_task_repair_filter(repair_handle: PlanWorkspaceHandle) -> None:
    """ac-007 — when repair_target_tasks=("prepare",), only prepare's test/impl
    files get fresh content; couple/isolate keep their iter-0 content."""
    gate = _ApproveOnPass(approve_at=1)
    deps = _build_deps(repair_handle, gate_policy=gate)

    await drive_with_repair(deps, "report", max_iterations=4)

    # After the loop ends, the live files reflect the second iteration's
    # work. Compare each per-task module against its iter-0 archive — only
    # `prepare` should differ.
    iter0 = repair_handle.repairs_dir(0)
    for task_id in ("couple", "isolate"):
        live = (repair_handle.tasks_pkg_dir() / f"{task_id}.py").read_text()
        archived = (iter0 / "src" / "experiment" / "tasks" / f"{task_id}.py").read_text()
        assert live == archived, f"{task_id}.py should be reused verbatim from iter-0"
        live_test = (repair_handle.tests_dir() / f"test_{task_id}.py").read_text()
        archived_test = (iter0 / "tests" / f"test_{task_id}.py").read_text()
        assert live_test == archived_test


@pytest.mark.asyncio
async def test_max_iterations_budget(repair_handle: PlanWorkspaceHandle) -> None:
    """ac-013 — exhausting max_iterations surfaces RepairBudgetExceeded
    and the returned WorkflowResult's HandoffResult has status=='rejected'."""
    gate = _AlwaysReject()
    deps = _build_deps(repair_handle, gate_policy=gate)

    with pytest.warns(RepairBudgetExceeded):
        result = await drive_with_repair(deps, "report", max_iterations=2)

    # The driver returns a WorkflowResult whose HumanReview output is the
    # last (rejected) handoff; status must be "rejected".
    handoff = result.outputs["HumanReview"]
    # ``handoff`` may be a HandoffResult (pydantic) or a dict (model_dump
    # in the back-compat path); the .status field is the same name.
    status = getattr(handoff, "status", None) or handoff["status"]  # type: ignore[index]
    assert status == "rejected"
    assert gate.calls == 2
