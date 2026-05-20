"""RED-phase tests for the plan-graph cluster of
``molexp.agent.modes._planning``.

The package does not exist yet; these tests fail at collection until the
implementation lands.

Covers, per the testing rules:

- Basics  — full valid construction + ``model_dump(mode="json")`` round-trip.
- Edge cases — ``extra="forbid"`` rejects an unknown field;
  ``RetryPolicy.max_attempts < 1`` is rejected.
- Immutability — ``frozen=True`` rejects attribute assignment.
- Logic — ``step_by_id`` hit/miss; ``downstream_of`` transitive
  dependents; ``is_acyclic`` True / False.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from molexp.agent.modes._planning import (
    ApprovalGate,
    PlanCheck,
    PlanGraph,
    PlanState,
    PlanStep,
    PlanStepArtifact,
    PlanStepInput,
    PlanStepIO,
    RetryPolicy,
    RiskLevel,
)

# --------------------------------------------------------------------------
# fixtures (hand-built; no LLM, no I/O)
# --------------------------------------------------------------------------


def _step(step_id: str, *, depends_on: tuple[str, ...] = ()) -> PlanStep:
    return PlanStep(
        id=step_id,
        depends_on=depends_on,
        io=PlanStepIO(
            inputs=(PlanStepInput(name="raw", source_step=None),),
            outputs=("result",),
        ),
        artifacts=(PlanStepArtifact(path=f"{step_id}.json", description="step output"),),
        capability_id=f"cap::{step_id}",
        tool_binding=f"tool::{step_id}",
        checks=(PlanCheck(name="schema", description="output is valid", blocking=True),),
        retry_policy=RetryPolicy(max_attempts=2, on=("timeout",)),
        rollback=None,
        approval_gate=ApprovalGate.approve_execution,
        estimated_cost_usd=1.25,
        risk_level=RiskLevel.low,
        unknowns=(),
    )


def _linear_graph() -> PlanGraph:
    # a -> b -> c
    return PlanGraph(
        plan_id="plan-1",
        intent_ref="intent-1",
        steps=(
            _step("a"),
            _step("b", depends_on=("a",)),
            _step("c", depends_on=("b",)),
        ),
        state=PlanState.draft_plan,
        compiled_contract_ref=None,
        notes="linear pipeline",
    )


# --------------------------------------------------------------------------
# basics — leaf models
# --------------------------------------------------------------------------


def test_plan_step_input_construction() -> None:
    item = PlanStepInput(name="features", source_step="extract")
    assert item.name == "features"
    assert item.source_step == "extract"


def test_plan_step_input_accepts_none_source() -> None:
    item = PlanStepInput(name="features", source_step=None)
    assert item.source_step is None


def test_plan_step_io_construction() -> None:
    io = PlanStepIO(
        inputs=(PlanStepInput(name="raw", source_step=None),),
        outputs=("clean",),
    )
    assert io.outputs == ("clean",)
    assert io.inputs[0].name == "raw"


def test_plan_step_artifact_construction() -> None:
    artifact = PlanStepArtifact(path="model.pt", description="trained weights")
    assert artifact.path == "model.pt"
    assert artifact.description == "trained weights"


def test_plan_check_construction() -> None:
    check = PlanCheck(name="acc", description="accuracy gate", blocking=True)
    assert check.name == "acc"
    assert check.blocking is True


def test_retry_policy_default_max_attempts_is_one() -> None:
    policy = RetryPolicy(on=())
    assert policy.max_attempts == 1


def test_retry_policy_explicit_construction() -> None:
    policy = RetryPolicy(max_attempts=3, on=("timeout", "rate_limit"))
    assert policy.max_attempts == 3
    assert policy.on == ("timeout", "rate_limit")


# --------------------------------------------------------------------------
# basics — PlanStep / PlanGraph construction
# --------------------------------------------------------------------------


def test_plan_step_full_construction() -> None:
    step = _step("a")
    assert step.id == "a"
    assert step.capability_id == "cap::a"
    assert step.approval_gate is ApprovalGate.approve_execution
    assert step.retry_policy.max_attempts == 2
    assert step.risk_level is RiskLevel.low


def test_plan_graph_full_construction() -> None:
    graph = _linear_graph()
    assert graph.plan_id == "plan-1"
    assert len(graph.steps) == 3
    assert graph.state is PlanState.draft_plan
    assert graph.notes == "linear pipeline"


# --------------------------------------------------------------------------
# basics — JSON round-trip
# --------------------------------------------------------------------------


def test_plan_graph_json_round_trip() -> None:
    graph = _linear_graph()
    dumped = graph.model_dump(mode="json")
    restored = PlanGraph.model_validate(dumped)
    assert restored == graph


def test_plan_graph_json_dump_is_jsonable() -> None:
    dumped = _linear_graph().model_dump(mode="json")
    assert dumped["state"] == "draft_plan"
    assert isinstance(dumped["steps"], list)
    assert dumped["steps"][0]["approval_gate"] == "approve_execution"


# --------------------------------------------------------------------------
# edge cases — extra="forbid"
# --------------------------------------------------------------------------


def test_plan_step_input_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        PlanStepInput(name="x", source_step=None, kind="data")  # type: ignore[call-arg]


def test_plan_check_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        PlanCheck(name="c", description="d", blocking=True, severity="high")  # type: ignore[call-arg]


def test_retry_policy_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=1, on=(), backoff="exp")  # type: ignore[call-arg]


def test_plan_step_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        PlanStep(
            id="a",
            depends_on=(),
            io=PlanStepIO(inputs=(), outputs=()),
            artifacts=(),
            capability_id=None,
            tool_binding=None,
            checks=(),
            retry_policy=RetryPolicy(on=()),
            rollback=None,
            approval_gate=ApprovalGate.approve_direction,
            estimated_cost_usd=None,
            risk_level=RiskLevel.low,
            unknowns=(),
            owner="bob",  # type: ignore[call-arg]
        )


def test_plan_graph_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        PlanGraph(
            plan_id="p",
            intent_ref=None,
            steps=(),
            state=PlanState.intake,
            compiled_contract_ref=None,
            notes="",
            version=2,  # type: ignore[call-arg]
        )


# --------------------------------------------------------------------------
# edge cases — RetryPolicy.max_attempts >= 1
# --------------------------------------------------------------------------


def test_retry_policy_rejects_zero_max_attempts() -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=0, on=())


def test_retry_policy_rejects_negative_max_attempts() -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=-1, on=())


# --------------------------------------------------------------------------
# immutability — frozen=True
# --------------------------------------------------------------------------


def test_retry_policy_is_frozen() -> None:
    policy = RetryPolicy(max_attempts=2, on=())
    with pytest.raises(ValidationError):
        policy.max_attempts = 5  # type: ignore[misc]


def test_plan_step_is_frozen() -> None:
    step = _step("a")
    with pytest.raises(ValidationError):
        step.id = "b"  # type: ignore[misc]


def test_plan_graph_is_frozen() -> None:
    graph = _linear_graph()
    with pytest.raises(ValidationError):
        graph.plan_id = "other"  # type: ignore[misc]


# --------------------------------------------------------------------------
# logic — step_by_id
# --------------------------------------------------------------------------


def test_step_by_id_returns_matching_step() -> None:
    graph = _linear_graph()
    found = graph.step_by_id("b")
    assert found is not None
    assert found.id == "b"


def test_step_by_id_returns_none_for_unknown() -> None:
    assert _linear_graph().step_by_id("missing") is None


# --------------------------------------------------------------------------
# logic — downstream_of (transitive dependents)
# --------------------------------------------------------------------------


def test_downstream_of_returns_transitive_dependents() -> None:
    # a -> b -> c : everything downstream of "a" is {b, c}.
    downstream = _linear_graph().downstream_of("a")
    assert set(downstream) == {"b", "c"}


def test_downstream_of_direct_dependent_only() -> None:
    downstream = _linear_graph().downstream_of("b")
    assert set(downstream) == {"c"}


def test_downstream_of_leaf_step_is_empty() -> None:
    assert _linear_graph().downstream_of("c") == ()


# --------------------------------------------------------------------------
# logic — is_acyclic
# --------------------------------------------------------------------------


def test_is_acyclic_true_for_dag() -> None:
    assert _linear_graph().is_acyclic() is True


def test_is_acyclic_false_for_cyclic_graph() -> None:
    # a -> b and b -> a forms a 2-node cycle.
    cyclic = PlanGraph(
        plan_id="plan-cycle",
        intent_ref=None,
        steps=(
            _step("a", depends_on=("b",)),
            _step("b", depends_on=("a",)),
        ),
        state=PlanState.draft_plan,
        compiled_contract_ref=None,
        notes="cyclic",
    )
    assert cyclic.is_acyclic() is False


def test_is_acyclic_true_for_empty_graph() -> None:
    empty = PlanGraph(
        plan_id="plan-empty",
        intent_ref=None,
        steps=(),
        state=PlanState.intake,
        compiled_contract_ref=None,
        notes="",
    )
    assert empty.is_acyclic() is True
