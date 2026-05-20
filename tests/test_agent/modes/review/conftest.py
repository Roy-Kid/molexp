"""Shared fixtures + test doubles for the ReviewMode test suite.

The suite never reaches a live LLM:

- ``ScriptedRouter`` is the minimal :class:`~molexp.agent.router.Router`
  stub — ReviewMode optionally asks it for a one-line natural-language
  summary; the verdict is structurally complete without any LLM call.
- ``NoTextRouter`` is a router *without* ``complete_text`` to prove the
  summary degrades gracefully.
- The plan-graph builders construct a satisfying plan, a dropped-output
  plan, and a lost-evidence plan against one shared :class:`IntentSpec`
  / :class:`CapabilityGraph`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.harness.harness import AgentHarness
from molexp.agent.harness.session import Session
from molexp.agent.harness.session_storage import InMemorySessionStorage
from molexp.agent.modes._planning import (
    ApprovalGate,
    CapabilityGraph,
    CapabilityNode,
    EvidenceState,
    IntentConstraint,
    IntentSpec,
    PlanCheck,
    PlanGraph,
    PlanState,
    PlanStep,
    PlanStepArtifact,
    PlanStepInput,
    PlanStepIO,
    ResourceBudget,
    RetryPolicy,
    RiskLevel,
    SuccessCriterion,
)
from molexp.agent.modes.plan.plan_folder import PlanFolder
from molexp.agent.router import ModelTier, RouterTextResult
from molexp.agent.types import UsageBreakdown
from molexp.workspace import Workspace

# ── router stubs ─────────────────────────────────────────────────────────


class ScriptedRouter:
    """Minimal :class:`~molexp.agent.router.Router` stub for ReviewMode.

    ReviewMode only asks the router for an optional one-line summary;
    ``complete_text`` echoes a canned line.
    """

    def __init__(self, *, summary: str = "review summary") -> None:
        self._summary = summary
        self.calls: list[str] = []

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[object, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        self.calls.append(prompt)
        return RouterTextResult(text=self._summary)

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


class NoTextRouter:
    """A router stub *without* ``complete_text`` — proves graceful degrade.

    ReviewMode falls back to a deterministic summary when the router
    cannot complete text.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


# ── intent / capability fixtures ─────────────────────────────────────────


def make_intent() -> IntentSpec:
    """A shared :class:`IntentSpec` the review judges every plan against."""
    return IntentSpec(
        objective="produce a charge-density report for the molecule",
        non_goals=("retrain any model", "write to the production database"),
        required_outputs=("report.pdf", "density.npz"),
        constraints=(IntentConstraint(kind="time", detail="finish within one hour"),),
        assumptions=("the input geometry is already optimized",),
        missing_information=(),
        success_criteria=(
            SuccessCriterion(summary="report.pdf is generated", verifiable=True),
            SuccessCriterion(summary="density.npz is generated", verifiable=True),
        ),
        allowed_side_effects=("write files under the run directory",),
        budget=ResourceBudget(max_cost_usd=10.0, max_wall_seconds=3600.0, model_tier=None),
        risk_level=RiskLevel.low,
    )


def make_capability_graph(*, all_evidenced: bool = True) -> CapabilityGraph:
    """A :class:`CapabilityGraph` with one node per plan step.

    When ``all_evidenced`` is ``False`` the ``cap_render`` node is left
    :data:`EvidenceState.missing` to exercise the lost-evidence path.
    """
    render_state = EvidenceState.evidenced if all_evidenced else EvidenceState.missing
    return CapabilityGraph(
        nodes=(
            CapabilityNode(
                id="cap_compute",
                capability="compute the electron density",
                evidence_state=EvidenceState.evidenced,
                confidence=0.9,
                api_refs=("molpy.compute.density",),
                usage_limits=(),
                needs_user_confirmation=False,
            ),
            CapabilityNode(
                id="cap_render",
                capability="render the density report",
                evidence_state=render_state,
                confidence=0.8,
                api_refs=("molpy.report.render",),
                usage_limits=(),
                needs_user_confirmation=False,
            ),
        ),
        edges=(),
    )


# ── plan-step builders ───────────────────────────────────────────────────


def make_step(
    step_id: str,
    *,
    depends_on: tuple[str, ...] = (),
    inputs: tuple[PlanStepInput, ...] = (),
    outputs: tuple[str, ...] = (),
    artifacts: tuple[PlanStepArtifact, ...] = (),
    capability_id: str | None = None,
    checks: tuple[PlanCheck, ...] = (),
) -> PlanStep:
    """Build one :class:`PlanStep` for review-plan fixtures."""
    return PlanStep(
        id=step_id,
        depends_on=depends_on,
        io=PlanStepIO(inputs=inputs, outputs=outputs),
        artifacts=artifacts,
        capability_id=capability_id,
        tool_binding=None,
        checks=checks,
        retry_policy=RetryPolicy(max_attempts=1, on=()),
        rollback=None,
        approval_gate=ApprovalGate.approve_direction,
        estimated_cost_usd=None,
        risk_level=RiskLevel.low,
        unknowns=(),
    )


def _compute_step(*, passing_check: bool = True) -> PlanStep:
    """The ``compute`` step — produces ``density.npz``."""
    check = PlanCheck(
        name="density_nonempty",
        description="density.npz is non-empty",
        blocking=True,
    )
    return make_step(
        "compute",
        outputs=("density",),
        artifacts=(PlanStepArtifact(path="density.npz", description="electron density"),),
        capability_id="cap_compute",
        checks=(check,) if passing_check else (),
    )


def _render_step(*, capability_id: str | None = "cap_render") -> PlanStep:
    """The ``render`` step — produces ``report.pdf``."""
    return make_step(
        "render",
        depends_on=("compute",),
        inputs=(PlanStepInput(name="density", source_step="compute"),),
        outputs=("report",),
        artifacts=(PlanStepArtifact(path="report.pdf", description="charge-density report"),),
        capability_id=capability_id,
    )


def make_satisfying_plan(*, state: PlanState = PlanState.approved) -> PlanGraph:
    """A :class:`PlanGraph` whose artefacts cover every required output."""
    return PlanGraph(
        plan_id="satisfying-plan",
        intent_ref="charge-density-intent",
        steps=(_compute_step(), _render_step()),
        state=state,
        compiled_contract_ref=None,
        notes="",
    )


def make_dropped_output_plan(*, state: PlanState = PlanState.approved) -> PlanGraph:
    """A :class:`PlanGraph` missing the ``report.pdf`` required output."""
    return PlanGraph(
        plan_id="dropped-output-plan",
        intent_ref="charge-density-intent",
        steps=(_compute_step(),),
        state=state,
        compiled_contract_ref=None,
        notes="",
    )


def make_lost_evidence_plan(*, state: PlanState = PlanState.approved) -> PlanGraph:
    """A :class:`PlanGraph` whose ``render`` step binds a missing capability."""
    return PlanGraph(
        plan_id="lost-evidence-plan",
        intent_ref="charge-density-intent",
        steps=(_compute_step(), _render_step(capability_id="cap_render")),
        state=state,
        compiled_contract_ref=None,
        notes="",
    )


def make_unmet_check_plan() -> PlanGraph:
    """A :class:`PlanGraph` in ``ready_for_run`` with an unmet blocking check.

    The ``compute`` step carries no checks at all, yet the plan claims
    :data:`PlanState.ready_for_run` — a lifecycle inconsistency: the
    intent demands a verifiable density and no check backs it.
    """
    return PlanGraph(
        plan_id="unmet-check-plan",
        intent_ref="charge-density-intent",
        steps=(_compute_step(passing_check=False), _render_step()),
        state=PlanState.ready_for_run,
        compiled_contract_ref=None,
        notes="",
    )


# ── workspace / plan-folder fixtures ─────────────────────────────────────


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """A fresh workspace under ``tmp_path/lab``."""
    return Workspace(tmp_path / "lab")


@pytest.fixture
def plan_folder(workspace: Workspace) -> PlanFolder:
    """A mounted :class:`PlanFolder` anchored at the workspace root."""
    folder = workspace.add_folder(PlanFolder(name="review-plan"))
    return folder  # type: ignore[return-value]


@pytest.fixture
def intent() -> IntentSpec:
    """The shared :class:`IntentSpec` fixture."""
    return make_intent()


@pytest.fixture
def capabilities() -> CapabilityGraph:
    """The shared all-evidenced :class:`CapabilityGraph` fixture."""
    return make_capability_graph()


# ── harness builder ──────────────────────────────────────────────────────


def make_harness(router: object) -> tuple[AgentHarness, list[object]]:
    """Build an :class:`AgentHarness` + the event-sink list it writes to."""
    sink_events: list[object] = []

    async def sink(event: object) -> None:
        sink_events.append(event)

    session = Session(storage=InMemorySessionStorage(), session_id="review-test")
    harness = AgentHarness(
        session=session,
        event_sink=sink,
        router=router,  # type: ignore[arg-type]
    )
    return harness, sink_events


async def drain(mode: object, harness: object, *, user_input: str = "review") -> list[object]:
    """Drain a mode's event stream into a list."""
    events: list[object] = []
    async for event in mode.run(harness=harness, user_input=user_input):  # type: ignore[attr-defined]
        events.append(event)
    return events


__all__ = [
    "NoTextRouter",
    "ScriptedRouter",
    "drain",
    "make_capability_graph",
    "make_dropped_output_plan",
    "make_harness",
    "make_intent",
    "make_lost_evidence_plan",
    "make_satisfying_plan",
    "make_step",
    "make_unmet_check_plan",
]
