"""Tests for ``_repair_loop.drive_with_repair`` — PlanMode review→repair driver.

Acceptance criteria covered:

- ``ac-002`` — :class:`PlanReviewView` carries iteration state across rounds.
- ``ac-007`` — ``GenerateTaskTests`` / ``GenerateTaskImplementations`` honour
  ``ctx.deps.repair_target_tasks`` (per-task LLM filter).
- ``ac-011`` — first-pass approval returns immediately without archiving.
- ``ac-012`` — rejection drives a partial-rerun shaped by the decision.
- ``ac-013`` — ``max_iterations`` cap surfaces ``RepairBudgetExceeded`` +
  rejected status.
- ``PYDA-17`` — :class:`CapabilityDiscoveryRequired` and
  :class:`UnevidencedApiReference` are mapped to synthetic
  :class:`~molexp.agent.review.ReviewDecision` rows whose ``target_steps``
  follow the spec's escalation policy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.modes.plan import PlanFolder
from molexp.agent.modes.plan._repair_loop import RepairBudgetExceeded, drive_with_repair
from molexp.agent.modes.plan.protocols import PlanDeps
from molexp.agent.modes.plan.schemas import PlanReviewView
from molexp.agent.review import BypassPolicy, ReviewDecision, ReviewPolicy, ReviewView
from molexp.workspace import Workspace

from .conftest import FakeProvider

# ── Stub policies used by the tests ────────────────────────────────────────


class _ApproveOnPass:
    """Records each ``review`` invocation; approves on the configured pass."""

    def __init__(self, approve_at: int) -> None:
        self.approve_at = approve_at
        self.calls: list[PlanReviewView] = []

    async def review(self, view: ReviewView) -> ReviewDecision:
        # Plan-final hook receives PlanReviewView instances; record them
        # for the test assertions below.
        assert isinstance(view, PlanReviewView)
        self.calls.append(view)
        if len(self.calls) - 1 >= self.approve_at:
            return ReviewDecision(approved=True)
        return ReviewDecision(
            approved=False,
            reason="needs another pass",
            target_steps=("DraftImplementationPlan",),
            target_task_ids=("prepare",),
            cascade_downstream=True,
            feedback="iterate",
        )


class _AlwaysReject:
    def __init__(self) -> None:
        self.calls = 0

    async def review(self, view: ReviewView) -> ReviewDecision:
        del view
        self.calls += 1
        return ReviewDecision(
            approved=False,
            reason="no",
            target_task_ids=("prepare",),
        )


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def repair_handle(tmp_path: Path) -> PlanFolder:
    return Workspace(tmp_path / "ws").add_folder(PlanFolder(name="rep_loop"))


def _build_deps(
    handle: PlanFolder, *, final_policy: ReviewPolicy | None = None
) -> PlanDeps:
    from molexp.agent.modes.plan.policy import STANDARD_PLAN_POLICY

    resolved = final_policy if final_policy is not None else BypassPolicy()
    return PlanDeps(
        router=FakeProvider(),  # type: ignore[arg-type]
        policy=STANDARD_PLAN_POLICY,
        plan_folder=handle,
        final_policy_lookup=lambda: resolved,
    )


# ── Tests ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_first_pass_approval(repair_handle: PlanFolder) -> None:
    """ac-011 — policy approves on iteration 0 → drive_with_repair runs PLAN_WORKFLOW
    exactly once and writes no archives."""
    gate = _ApproveOnPass(approve_at=0)
    deps = _build_deps(repair_handle, final_policy=gate)

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
    repair_handle: PlanFolder,
) -> None:
    """ac-002 — second-iteration `PlanReviewView` carries `repair_iteration=1`."""
    gate = _ApproveOnPass(approve_at=1)
    deps = _build_deps(repair_handle, final_policy=gate)

    result = await drive_with_repair(deps, "report", max_iterations=4)

    assert result.status == "completed"
    assert len(gate.calls) == 2
    assert gate.calls[0].repair_iteration == 0
    assert gate.calls[1].repair_iteration == 1


@pytest.mark.asyncio
async def test_partial_rerun_round(repair_handle: PlanFolder) -> None:
    """ac-012 — second iteration runs a subgraph shaped by the rejection.

    The first pass writes test_prepare.py, test_couple.py, test_isolate.py.
    The rejection asks for `target_task_ids=("prepare",)` so the second
    iteration MUST regenerate prepare's test/impl but reuse couple/isolate
    from the prior round (verified via mtime / content stability).
    """
    gate = _ApproveOnPass(approve_at=1)
    deps = _build_deps(repair_handle, final_policy=gate)

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
    assert record["target_steps"] == ["DraftImplementationPlan"]
    assert record["target_task_ids"] == ["prepare"]
    assert record["cascade_downstream"] is True


@pytest.mark.asyncio
async def test_repair_feedback_is_injected_into_next_llm_round(
    repair_handle: PlanFolder,
) -> None:
    """Reviewer feedback becomes first-class repair context for the next LLM call."""
    from molexp.agent.modes.plan.policy import STANDARD_PLAN_POLICY

    router = FakeProvider()
    gate = _ApproveOnPass(approve_at=1)
    deps = PlanDeps(
        router=router,  # type: ignore[arg-type]
        policy=STANDARD_PLAN_POLICY,
        plan_folder=repair_handle,
        final_policy_lookup=lambda: gate,
    )

    await drive_with_repair(deps, "report", max_iterations=4)

    plan_prompts = [
        prompt
        for node_id, prompt in router.prompts
        if node_id.startswith("DraftImplementationPlan")
    ]
    assert len(plan_prompts) == 2
    assert "## Repair context (binding)" in plan_prompts[1]
    assert "Reviewer feedback:" in plan_prompts[1]
    assert "iterate" in plan_prompts[1]


@pytest.mark.asyncio
async def test_per_task_repair_filter(repair_handle: PlanFolder) -> None:
    """ac-007 — when repair_target_tasks=("prepare",), only prepare's test/impl
    files get fresh content; couple/isolate keep their iter-0 content."""
    gate = _ApproveOnPass(approve_at=1)
    deps = _build_deps(repair_handle, final_policy=gate)

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
async def test_max_iterations_budget(repair_handle: PlanFolder) -> None:
    """ac-013 — exhausting max_iterations surfaces RepairBudgetExceeded
    and the returned WorkflowResult's HandoffResult has status=='rejected'."""
    gate = _AlwaysReject()
    deps = _build_deps(repair_handle, final_policy=gate)

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


# ── Step-level rejection → subgraph rerun ────────────────────────────────


class _RejectStepOnce:
    """Step policy that rejects ``reject_step`` exactly once.

    All other steps approve unconditionally so the pipeline drives
    through the materialization once, hits the rejection, and then
    completes on the partial rerun.  ``call_log`` records the
    ``step_id`` of every consultation so tests can assert which steps
    fired in each iteration.
    """

    def __init__(self, *, reject_step: str) -> None:
        self.reject_step = reject_step
        self.rejected_count = 0
        self.call_log: list[str] = []

    async def review(self, view: ReviewView) -> ReviewDecision:
        self.call_log.append(view.step_id)
        if view.step_id == self.reject_step and self.rejected_count == 0:
            self.rejected_count += 1
            return ReviewDecision(
                approved=False,
                reason=f"step {self.reject_step} rejected once",
                target_steps=(self.reject_step,),
                cascade_downstream=True,
                feedback="please redo",
            )
        return ReviewDecision(approved=True)


@pytest.mark.asyncio
async def test_step_rejection_replays_only_rejected_step_and_downstream(
    repair_handle: PlanFolder,
) -> None:
    """A step-policy rejection of ``DraftImplementationPlan`` rebuilds a
    subgraph that recomputes that step plus its downstream cascade —
    earlier steps (``IngestReport`` / ``DraftReportDigest``) MUST NOT be
    rerun.  Observable signal: ``FakeProvider.calls`` for the LLM-bearing
    upstream nodes stays at exactly one call after the repair completes.
    """
    from molexp.agent.modes.plan.policy import STANDARD_PLAN_POLICY

    router = FakeProvider()
    step_policy = _RejectStepOnce(reject_step="DraftImplementationPlan")
    deps = PlanDeps(
        router=router,  # type: ignore[arg-type]
        policy=STANDARD_PLAN_POLICY,
        plan_folder=repair_handle,
        step_policy_lookup=lambda: step_policy,
    )

    result = await drive_with_repair(deps, "report", max_iterations=4)

    assert result.status == "completed"

    # DraftReportDigest is the only LLM-bearing step strictly upstream
    # of the rejected node; it should have been called exactly once
    # across both iterations.
    upstream_calls = [c for c in router.calls if c[0].startswith("DraftReportDigest")]
    assert len(upstream_calls) == 1, (
        f"DraftReportDigest should NOT be re-invoked on the partial rerun; "
        f"got {len(upstream_calls)} calls: {upstream_calls}"
    )

    # DraftImplementationPlan, by contrast, is the rejected step and
    # MUST run twice (initial reject + repair).
    rejected_calls = [c for c in router.calls if c[0].startswith("DraftImplementationPlan")]
    assert len(rejected_calls) == 2, (
        f"DraftImplementationPlan should run twice (initial + repair); "
        f"got {len(rejected_calls)} calls: {rejected_calls}"
    )

    # The step policy should record one rejection followed by approvals.
    assert step_policy.rejected_count == 1
    assert "DraftImplementationPlan" in step_policy.call_log


@pytest.mark.asyncio
async def test_step_rejection_archives_and_logs_decision(
    repair_handle: PlanFolder,
) -> None:
    """Step-level rejection persists a ``RepairIterationRecord`` and the
    archived iter-0 tree just like a plan-final rejection — so the audit
    trail is the same regardless of which hook flagged the run."""
    from molexp.agent.modes.plan.policy import STANDARD_PLAN_POLICY

    router = FakeProvider()
    step_policy = _RejectStepOnce(reject_step="DraftImplementationPlan")
    deps = PlanDeps(
        router=router,  # type: ignore[arg-type]
        policy=STANDARD_PLAN_POLICY,
        plan_folder=repair_handle,
        step_policy_lookup=lambda: step_policy,
    )

    await drive_with_repair(deps, "report", max_iterations=4)

    # iter-0 archive exists.
    iter0 = repair_handle.repairs_dir(0)
    assert (iter0 / "plan").exists()

    # Manifest carries a single repair_history row targeting the
    # rejected step with cascade_downstream=True.
    import yaml

    manifest_data = yaml.safe_load(repair_handle.manifest_path().read_text())
    assert manifest_data["repair_iterations"] == 1
    record = manifest_data["repair_history"][0]
    assert record["target_steps"] == ["DraftImplementationPlan"]
    assert record["cascade_downstream"] is True


# ── PYDA-17 — capability-exception → synthetic decision ───────────────────


def test_synthesize_decision_for_missing_capability() -> None:
    """``CapabilityDiscoveryRequired`` always re-runs both nodes and cascades."""
    from molexp.agent.modes.plan._repair_loop import _synthesize_capability_decision
    from molexp.agent.modes.plan.errors import CapabilityDiscoveryRequired

    exc = CapabilityDiscoveryRequired(
        "no probe configured",
        reason="no_probe",
        detail="set MOLEXP_MOLMCP_COMMAND",
    )
    decision = _synthesize_capability_decision(exc, unevidenced_count=0)

    assert decision.approved is False
    assert decision.target_steps == ("DraftCapabilityNeeds", "DiscoverCapabilities")
    assert decision.cascade_downstream is True


def test_synthesize_decision_for_unevidenced_first_iteration() -> None:
    """First ``UnevidencedApiReference`` re-runs only ``DiscoverCapabilities``."""
    from molexp.agent.modes.plan._repair_loop import _synthesize_capability_decision
    from molexp.agent.modes.plan.errors import UnevidencedApiReference

    exc = UnevidencedApiReference(
        "missing molpy.X.Y",
        refs=("molpy.X.Y",),
        reason="",
        detail="not in evidence batch",
    )
    decision = _synthesize_capability_decision(exc, unevidenced_count=0)
    assert decision.target_steps == ("DiscoverCapabilities",)
    assert decision.cascade_downstream is True


def test_synthesize_decision_for_unevidenced_second_iteration_escalates() -> None:
    """Second ``UnevidencedApiReference`` escalates to both nodes."""
    from molexp.agent.modes.plan._repair_loop import _synthesize_capability_decision
    from molexp.agent.modes.plan.errors import UnevidencedApiReference

    exc = UnevidencedApiReference(
        "missing molpy.X.Y again",
        refs=("molpy.X.Y",),
        detail="still wrong",
    )
    decision = _synthesize_capability_decision(exc, unevidenced_count=1)
    assert decision.target_steps == ("DraftCapabilityNeeds", "DiscoverCapabilities")


# ── End-to-end: capability-exception drives a repair iteration ────────────


class _ProbeFlipsAfterFirstFail:
    """Probe that raises CapabilityDiscoveryRequired on the first call to
    ``discover``, then succeeds with a skipped batch on subsequent calls.

    Exercises the full drive_with_repair recovery path: exception →
    archive → record → re-run from scratch → success.
    """

    def __init__(self) -> None:
        self.discover_calls = 0

    async def draft_needs(
        self,
        *,
        plan_brief: object,
        repair_context: object | None = None,
    ) -> object:
        del plan_brief, repair_context
        from molexp.agent.modes.plan.capability import CapabilityNeed, CapabilityNeedReport

        return CapabilityNeedReport(
            discovery_required=True,
            needs=(CapabilityNeed(task_id="prepare", capability="x"),),
        )

    async def discover(self, report: object, repair_context: object | None = None) -> object:
        del report, repair_context
        from molexp.agent.modes.plan.capability import CapabilityEvidenceBatch
        from molexp.agent.modes.plan.errors import CapabilityDiscoveryRequired

        self.discover_calls += 1
        if self.discover_calls == 1:
            raise CapabilityDiscoveryRequired(
                "first call fails",
                reason="injected",
                detail="for test",
            )
        # Subsequent calls: short-circuit downstream evidence checks.
        return CapabilityEvidenceBatch(discovery_skipped=True)


@pytest.mark.asyncio
async def test_capability_exception_drives_repair_iteration(
    repair_handle: PlanFolder,
) -> None:
    """End-to-end: a probe-raised CapabilityDiscoveryRequired triggers
    one repair iteration and the manifest records the synthetic decision."""
    deps = _build_deps(repair_handle)
    deps = deps.__class__(  # replace capability_probe via plain re-construction
        router=deps.router,
        policy=deps.policy,
        plan_folder=deps.plan_folder,
        step_policy_lookup=deps.step_policy_lookup,
        final_policy_lookup=deps.final_policy_lookup,
        capability_probe=_ProbeFlipsAfterFirstFail(),
    )

    result = await drive_with_repair(deps, "report", max_iterations=4)

    assert result.status == "completed"
    # One iteration was triggered by the exception.
    import yaml

    manifest_data = yaml.safe_load(repair_handle.manifest_path().read_text())
    assert manifest_data["repair_iterations"] >= 1
    # The first repair record is from the capability exception, with
    # the spec's target_steps mapping.
    record = manifest_data["repair_history"][0]
    assert "DraftCapabilityNeeds" in record["target_steps"]
    assert "DiscoverCapabilities" in record["target_steps"]
    assert record["cascade_downstream"] is True


@pytest.mark.asyncio
async def test_capability_exception_exhausts_budget(
    repair_handle: PlanFolder,
) -> None:
    """A probe that always raises eventually exhausts the budget and re-raises."""
    from molexp.agent.modes.plan.errors import CapabilityDiscoveryRequired

    class _AlwaysFails:
        async def draft_needs(self, **_kw: object) -> object:
            from molexp.agent.modes.plan.capability import CapabilityNeedReport

            return CapabilityNeedReport(discovery_required=True)

        async def discover(
            self,
            _report: object,
            repair_context: object | None = None,
        ) -> object:
            del repair_context
            raise CapabilityDiscoveryRequired("nope", reason="nope")

    deps = _build_deps(repair_handle)
    deps = deps.__class__(
        router=deps.router,
        policy=deps.policy,
        plan_folder=deps.plan_folder,
        step_policy_lookup=deps.step_policy_lookup,
        final_policy_lookup=deps.final_policy_lookup,
        capability_probe=_AlwaysFails(),
    )

    with pytest.raises(CapabilityDiscoveryRequired):
        await drive_with_repair(deps, "report", max_iterations=2)
