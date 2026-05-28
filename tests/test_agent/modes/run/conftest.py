"""Shared fixtures + test doubles for the RunMode test suite.

The suite never reaches a live HPC scheduler or a live LLM:

- ``StubWorkflow`` duck-types :class:`molexp.workflow.Workflow.execute`
  with a scripted :class:`~molexp.workflow.WorkflowResult` (or a raised
  exception) — RunMode drives it through the public API surface.
- ``ScriptedRouter`` is the minimal :class:`~molexp.agent.router.Router`
  stub (RunMode makes no model call of its own; the router is present
  only for ``snapshot_usage`` / ``clear_usage``).
- ``approving_harness`` / ``blocking_harness`` register a
  :class:`~molexp.agent.review.ReviewPolicy`-shaped ``before_approval``
  hook so the ``approve_execution`` gate clears (or does not).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.agent.hooks import HookContext, HookPoint
from molexp.agent.modes._planning import (
    ApprovalGate,
    IsolatedTestSketch,
    PlanGraph,
    PlanState,
    PlanStep,
    PlanStepInput,
    PlanStepIO,
    RetryPolicy,
    RiskLevel,
)
from molexp.agent.modes.author.handoff import MaterializedWorkspaceHandoff
from molexp.agent.modes.plan.plan_folder import PlanFolder
from molexp.agent.review import ReviewDecision
from molexp.agent.router import ModelTier, RouterTextResult
from molexp.agent.runtime import AgentHarness
from molexp.agent.session import Session
from molexp.agent.session_storage import InMemorySessionStorage
from molexp.agent.types import UsageBreakdown
from molexp.workflow import ValidationReport, WorkflowResult
from molexp.workspace import Experiment, Workspace

# ── workflow stub ────────────────────────────────────────────────────────


class StubWorkflow:
    """A scripted :class:`molexp.workflow.Workflow.execute` stand-in.

    RunMode drives the materialized workflow purely through ``execute`` —
    this stub records the call and returns a scripted
    :class:`WorkflowResult` or raises a scripted exception. It is *not* a
    :class:`Workflow` subclass; the loader's type assertion is exercised
    separately with a real ``Workflow`` fixture.
    """

    def __init__(
        self,
        *,
        result: WorkflowResult | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self._result = result
        self._raises = raises
        self.execute_calls: list[dict[str, object]] = []

    async def execute(self, **kwargs: object) -> WorkflowResult:
        self.execute_calls.append(dict(kwargs))
        if self._raises is not None:
            raise self._raises
        assert self._result is not None, "StubWorkflow needs a result or a raises"
        return self._result


def passing_result(step_ids: tuple[str, ...]) -> WorkflowResult:
    """A ``completed`` :class:`WorkflowResult` with one output per step."""
    return WorkflowResult(
        status="completed",
        outputs={sid: {"ok": True} for sid in step_ids},
        run_id="run-stub",
        execution_id="exec-stub",
    )


def failing_result(succeeded: tuple[str, ...]) -> WorkflowResult:
    """A ``failed`` :class:`WorkflowResult` carrying only ``succeeded`` outputs."""
    return WorkflowResult(
        status="failed",
        outputs={sid: {"ok": True} for sid in succeeded},
        run_id="run-stub",
        execution_id="exec-stub",
    )


# ── router stub ──────────────────────────────────────────────────────────


class ScriptedRouter:
    """Minimal :class:`~molexp.agent.router.Router` stub for RunMode.

    RunMode makes no model call of its own; the router is present only so
    ``snapshot_usage`` / ``clear_usage`` resolve.
    """

    def __init__(self) -> None:
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
        return RouterTextResult(text=f"echo:{prompt}")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


# ── plan-graph builders ──────────────────────────────────────────────────


def make_step(
    step_id: str,
    *,
    depends_on: tuple[str, ...] = (),
    inputs: tuple[PlanStepInput, ...] = (),
    outputs: tuple[str, ...] = (),
    retry_policy: RetryPolicy | None = None,
) -> PlanStep:
    """Build one :class:`PlanStep` for plan-graph fixtures."""
    return PlanStep(
        id=step_id,
        depends_on=depends_on,
        io=PlanStepIO(inputs=inputs, outputs=outputs),
        artifacts=(),
        api_refs=("molpy.System",),
        composition_notes="fixture step",
        checks=(),
        retry_policy=retry_policy or RetryPolicy(max_attempts=1, on=()),
        rollback=None,
        approval_gate=ApprovalGate.approve_execution,
        estimated_cost_usd=None,
        risk_level=RiskLevel.low,
        unknowns=(),
        test_sketch=IsolatedTestSketch(
            is_isolated_testable=True,
            synthetic_inputs=(),
            assertion_sketch=(),
            rationale="",
        ),
    )


def make_plan_graph(
    *,
    plan_id: str = "demo-plan",
    state: PlanState = PlanState.ready_for_run,
) -> PlanGraph:
    """A two-step :class:`PlanGraph` (``prepare → run``) in ``ready_for_run``."""
    prepare = make_step("prepare", outputs=("payload",))
    run = make_step(
        "run",
        depends_on=("prepare",),
        inputs=(PlanStepInput(name="payload", source_step="prepare"),),
        outputs=("result",),
    )
    return PlanGraph(
        plan_id=plan_id,
        intent_ref="intent-1",
        steps=(prepare, run),
        state=state,
        compiled_contract_ref="wf-1",
        notes="",
    )


def make_handoff(
    *,
    plan_id: str = "demo-plan",
    workspace_path: Path,
    entrypoint_module: str = "experiment.workflow",
    entrypoint_symbol: str = "create_workflow",
    source_root: Path | None = None,
) -> MaterializedWorkspaceHandoff:
    """A :class:`MaterializedWorkspaceHandoff` for RunMode fixtures."""
    return MaterializedWorkspaceHandoff(
        plan_id=plan_id,
        plan_graph=make_plan_graph(plan_id=plan_id),
        experiment_workspace_path=workspace_path,
        workflow_yaml_path=workspace_path / "ir" / "workflow.yaml",
        entrypoint_module=entrypoint_module,
        entrypoint_symbol=entrypoint_symbol,
        source_root=source_root or workspace_path / "src",
        validation_report_snapshot=ValidationReport(ok=True, issues=()),
        materialization_approved_at=datetime.now(UTC),
    )


# ── workspace fixtures ───────────────────────────────────────────────────


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """A fresh workspace under ``tmp_path/lab``."""
    return Workspace(tmp_path / "lab")


@pytest.fixture
def experiment(workspace: Workspace) -> Experiment:
    """A workspace ``Experiment`` to host RunMode's workspace ``Run``."""
    return workspace.add_project("proj").add_experiment("exp")


@pytest.fixture
def plan_folder(workspace: Workspace) -> PlanFolder:
    """A mounted :class:`PlanFolder` in state ``ready_for_run``."""
    folder = workspace.add_folder(PlanFolder(name="demo-plan"))
    for dst in (
        PlanState.exploring,
        PlanState.draft_plan,
        PlanState.awaiting_approval,
        PlanState.approved,
        PlanState.materializing,
        PlanState.validating,
        PlanState.ready_for_run,
    ):
        folder.transition_to(dst)
    folder.save()
    return folder  # type: ignore[return-value]


# ── harness builders ─────────────────────────────────────────────────────


def make_harness(
    router: object,
    *,
    approval: ReviewDecision | None = None,
) -> tuple[AgentHarness, list[object]]:
    """Build an :class:`AgentHarness`; optionally register an approval hook.

    When ``approval`` is supplied, a ``before_approval`` hook returning it
    is registered — so the ``approve_execution`` gate resolves to that
    decision. When ``approval`` is ``None``, no hook is registered and the
    harness defaults to approving (its documented no-hook behaviour).
    """
    sink_events: list[object] = []

    async def sink(event: object) -> None:
        sink_events.append(event)

    session = Session(storage=InMemorySessionStorage(), session_id="run-test")
    harness = AgentHarness(
        session=session,
        event_sink=sink,
        router=router,  # type: ignore[arg-type]
    )
    if approval is not None:

        async def _approval_hook(_ctx: HookContext) -> ReviewDecision:
            return approval

        harness.hooks.register(HookPoint.before_approval, _approval_hook)
    return harness, sink_events


def approving_decision() -> ReviewDecision:
    """An approving :class:`ReviewDecision` for the ``approve_execution`` gate."""
    return ReviewDecision(approved=True, reason="reviewer approved execution")


def blocking_decision() -> ReviewDecision:
    """A blocking :class:`ReviewDecision` for the ``approve_execution`` gate."""
    return ReviewDecision(approved=False, reason="reviewer blocked execution")


async def drain(mode: object, harness: object) -> list[object]:
    """Drain a mode's event stream into a list."""
    events: list[object] = []
    async for event in mode.run(harness=harness, user_input="execute"):  # type: ignore[attr-defined]
        events.append(event)
    return events


# Re-export the most-used names so test modules import from one place.
__all__ = [
    "ScriptedRouter",
    "StubWorkflow",
    "approving_decision",
    "blocking_decision",
    "drain",
    "failing_result",
    "make_handoff",
    "make_harness",
    "make_plan_graph",
    "make_step",
    "passing_result",
]
