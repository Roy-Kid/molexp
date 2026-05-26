"""Shared fixtures + test doubles for the AuthorMode test suite.

The suite never reaches a live LLM or a real MCP server: a
``ScriptedRouter`` feeds canned structured responses keyed by schema,
and a real :class:`~molexp.agent.harness.execution_env.LocalExecutionEnv`
runs the generated tests in genuine subprocesses (the debug loop's
isolation floor).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel

from molexp.agent.harness.execution_env import LocalExecutionEnv
from molexp.agent.harness.harness import AgentHarness
from molexp.agent.harness.session import Session
from molexp.agent.harness.session_storage import InMemorySessionStorage
from molexp.agent.modes._planning import (
    ApprovalGate,
    IntentSpec,
    IsolatedTestSketch,
    PlanGraph,
    PlanState,
    PlanStep,
    PlanStepInput,
    PlanStepIO,
    ResourceBudget,
    RetryPolicy,
    RiskLevel,
)
from molexp.agent.modes.plan.handoff import ApprovedPlanHandoff
from molexp.agent.modes.plan.plan_folder import PlanFolder
from molexp.agent.router import ModelTier, RouterTextResult
from molexp.agent.types import UsageBreakdown
from molexp.workspace import Workspace


class ScriptedRouter:
    """A :class:`~molexp.agent.router.Router` stub for the structured path.

    ``complete_structured`` returns the next scripted response whose type
    matches the requested ``schema`` (FIFO per schema). When a
    ``schema_factory`` is registered for a schema, it is called per
    request instead (used to generate per-task source). Every call is
    recorded on ``calls``.
    """

    def __init__(self, responses: Sequence[BaseModel] = ()) -> None:
        self._responses: list[BaseModel] = list(responses)
        self._factories: dict[type, object] = {}
        self.calls: list[dict[str, object]] = []

    def register_factory(self, schema: type, factory: object) -> None:
        """Register a callable ``factory(node_id) -> schema`` for ``schema``."""
        self._factories[schema] = factory

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[object, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        return RouterTextResult(text=f"echo:{prompt}")

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[BaseModel],
        node_id: str = "",
    ) -> BaseModel:
        self.calls.append({"tier": tier, "schema": schema, "node_id": node_id, "user": user})
        factory = self._factories.get(schema)
        if factory is not None:
            return factory(node_id)  # type: ignore[operator,no-any-return]
        for index, response in enumerate(self._responses):
            if isinstance(response, schema):
                return self._responses.pop(index)
        raise AssertionError(
            f"ScriptedRouter has no scripted {schema.__name__} response (node_id={node_id!r})"
        )

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def make_intent() -> IntentSpec:
    """A minimal :class:`IntentSpec` for handoff fixtures."""
    return IntentSpec(
        objective="materialize a two-step pipeline",
        non_goals=(),
        required_outputs=("result",),
        constraints=(),
        assumptions=(),
        missing_information=(),
        success_criteria=(),
        allowed_side_effects=(),
        budget=ResourceBudget(max_cost_usd=None, max_wall_seconds=None, model_tier=None),
        risk_level=RiskLevel.low,
    )


def make_step(
    step_id: str,
    *,
    depends_on: tuple[str, ...] = (),
    inputs: tuple[PlanStepInput, ...] = (),
    outputs: tuple[str, ...] = (),
    api_refs: tuple[str, ...] = ("molpy.System",),
    composition_notes: str = "test fixture step",
) -> PlanStep:
    """Build one :class:`PlanStep` for plan-graph fixtures."""
    return PlanStep(
        id=step_id,
        depends_on=depends_on,
        io=PlanStepIO(inputs=inputs, outputs=outputs),
        artifacts=(),
        api_refs=api_refs,
        composition_notes=composition_notes,
        checks=(),
        retry_policy=RetryPolicy(max_attempts=1, on=()),
        rollback=None,
        approval_gate=ApprovalGate.approve_materialization,
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
    state: PlanState = PlanState.approved,
) -> PlanGraph:
    """A simple two-step approved :class:`PlanGraph` (``prepare → run``)."""
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
        compiled_contract_ref=None,
        notes="",
    )


@pytest.fixture
def plan_folder(tmp_path: Path) -> PlanFolder:
    """A mounted :class:`PlanFolder` in state ``approved``."""
    ws = Workspace(tmp_path / "lab")
    folder = ws.add_folder(PlanFolder(name="demo-plan"))
    # intake -> exploring -> draft_plan -> awaiting_approval -> approved
    folder.transition_to(PlanState.exploring)
    folder.transition_to(PlanState.draft_plan)
    folder.transition_to(PlanState.awaiting_approval)
    folder.transition_to(PlanState.approved)
    folder.save()
    return folder  # type: ignore[return-value]


@pytest.fixture
def approved_handoff(plan_folder: PlanFolder) -> ApprovedPlanHandoff:
    """An :class:`ApprovedPlanHandoff` bound to the ``plan_folder`` fixture."""
    return ApprovedPlanHandoff(
        plan_id=plan_folder.plan_id,
        intent=make_intent(),
        plan_graph=make_plan_graph(plan_id=plan_folder.plan_id),
        plan_folder_path=Path(str(plan_folder.path())),
        direction_approved_at=datetime.now(UTC),
    )


def make_harness(
    router: object,
    *,
    scratch_dir: Path,
    session: Session | None = None,
) -> tuple[AgentHarness, list[object]]:
    """Build an :class:`AgentHarness` with a real :class:`LocalExecutionEnv`."""
    sink_events: list[object] = []

    async def sink(event: object) -> None:
        sink_events.append(event)

    sess = session or Session(storage=InMemorySessionStorage(), session_id="author-test")
    harness = AgentHarness(
        session=sess,
        event_sink=sink,
        router=router,  # type: ignore[arg-type]
        execution_env=LocalExecutionEnv(scratch_dir=scratch_dir),
    )
    return harness, sink_events
