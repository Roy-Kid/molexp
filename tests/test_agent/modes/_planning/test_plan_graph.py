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
        api_refs=("molpy.System",),
        composition_notes="step",
        checks=(PlanCheck(name="schema", description="output is valid", blocking=True),),
        retry_policy=RetryPolicy(max_attempts=2, on=("timeout",)),
        rollback=None,
        approval_gate=ApprovalGate.approve_execution,
        estimated_cost_usd=1.25,
        risk_level=RiskLevel.low,
        unknowns=(),
        test_sketch=IsolatedTestSketch(
            is_isolated_testable=True,
            synthetic_inputs=(),
            assertion_sketch=(),
            rationale="",
        ),
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
    assert step.api_refs == ("molpy.System",)
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
            api_refs=("molpy.System",),
            composition_notes="step",
            checks=(),
            retry_policy=RetryPolicy(on=()),
            rollback=None,
            approval_gate=ApprovalGate.approve_direction,
            estimated_cost_usd=None,
            risk_level=RiskLevel.low,
            unknowns=(),
            test_sketch=IsolatedTestSketch(
                is_isolated_testable=True,
                synthetic_inputs=(),
                assertion_sketch=(),
                rationale="",
            ),
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


# ==========================================================================
# RED — testability-driven decomposition (spec ac-001, ac-002)
#
# Two not-yet-existing contract pieces:
#   1. IsolatedTestSketch — new frozen-pydantic model in plan_graph.py.
#   2. PlanStep gains a REQUIRED field test_sketch: IsolatedTestSketch
#      (breaking change, no default, no backward compatibility).
#
# These tests fail RED until the implementation lands:
#   - the IsolatedTestSketch import below raises ImportError at collection;
#   - the required-field test would still fail even if the import were
#     stubbed, since current PlanStep has no test_sketch field.
# ==========================================================================

from molexp.agent.modes._planning import IsolatedTestSketch  # noqa: E402


def _test_sketch(*, is_isolated_testable: bool = True) -> IsolatedTestSketch:
    """Build a valid IsolatedTestSketch for use in PlanStep fixtures."""
    return IsolatedTestSketch(
        is_isolated_testable=is_isolated_testable,
        synthetic_inputs=("a 3-atom synthetic molecule with explicit coords",),
        assertion_sketch=("output atom count equals input atom count",),
        rationale="bound to a primitive symbol; synthetic input is cheap to build",
    )


def _testable_step(step_id: str, *, depends_on: tuple[str, ...] = ()) -> PlanStep:
    """Build a fully valid PlanStep including the required test_sketch."""
    return PlanStep(
        id=step_id,
        depends_on=depends_on,
        io=PlanStepIO(
            inputs=(PlanStepInput(name="raw", source_step=None),),
            outputs=("result",),
        ),
        artifacts=(PlanStepArtifact(path=f"{step_id}.json", description="step output"),),
        api_refs=("molpy.System",),
        composition_notes="step",
        checks=(PlanCheck(name="schema", description="output is valid", blocking=True),),
        retry_policy=RetryPolicy(max_attempts=2, on=("timeout",)),
        rollback=None,
        approval_gate=ApprovalGate.approve_execution,
        estimated_cost_usd=1.25,
        risk_level=RiskLevel.low,
        unknowns=(),
        test_sketch=_test_sketch(),
    )


# --------------------------------------------------------------------------
# happy path — IsolatedTestSketch construction
# --------------------------------------------------------------------------


def test_isolated_test_sketch_full_construction() -> None:
    sketch = IsolatedTestSketch(
        is_isolated_testable=True,
        synthetic_inputs=("a tiny synthetic graph",),
        assertion_sketch=("returns a non-empty result",),
        rationale="primitive symbol, cheap synthetic input",
    )
    assert sketch.is_isolated_testable is True
    assert sketch.synthetic_inputs == ("a tiny synthetic graph",)
    assert sketch.assertion_sketch == ("returns a non-empty result",)
    assert sketch.rationale == "primitive symbol, cheap synthetic input"


def test_isolated_test_sketch_not_testable_construction() -> None:
    sketch = IsolatedTestSketch(
        is_isolated_testable=False,
        synthetic_inputs=(),
        assertion_sketch=(),
        rationale="needs the real upstream trajectory; cannot synthesize cheaply",
    )
    assert sketch.is_isolated_testable is False
    assert sketch.synthetic_inputs == ()
    assert sketch.assertion_sketch == ()


# --------------------------------------------------------------------------
# happy path — IsolatedTestSketch JSON round-trip
# --------------------------------------------------------------------------


def test_isolated_test_sketch_json_round_trip() -> None:
    sketch = _test_sketch()
    dumped = sketch.model_dump(mode="json")
    restored = IsolatedTestSketch.model_validate(dumped)
    assert restored == sketch


def test_isolated_test_sketch_json_dump_is_jsonable() -> None:
    dumped = _test_sketch().model_dump(mode="json")
    assert dumped["is_isolated_testable"] is True
    assert isinstance(dumped["synthetic_inputs"], list)
    assert isinstance(dumped["assertion_sketch"], list)
    assert isinstance(dumped["rationale"], str)


# --------------------------------------------------------------------------
# happy path — PlanStep carries a test_sketch (ac-001)
# --------------------------------------------------------------------------


def test_plan_step_construction_with_test_sketch() -> None:
    step = _testable_step("a")
    assert isinstance(step.test_sketch, IsolatedTestSketch)
    assert step.test_sketch.is_isolated_testable is True


def test_plan_step_with_test_sketch_json_round_trip() -> None:
    step = _testable_step("a")
    dumped = step.model_dump(mode="json")
    restored = PlanStep.model_validate(dumped)
    assert restored == step
    assert restored.test_sketch == step.test_sketch


# --------------------------------------------------------------------------
# immutability — IsolatedTestSketch is frozen
# --------------------------------------------------------------------------


def test_isolated_test_sketch_is_frozen() -> None:
    sketch = _test_sketch()
    with pytest.raises(ValidationError):
        sketch.is_isolated_testable = False  # type: ignore[misc]


# --------------------------------------------------------------------------
# edge — IsolatedTestSketch rejects unknown field (extra="forbid")
# --------------------------------------------------------------------------


def test_isolated_test_sketch_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        IsolatedTestSketch(
            is_isolated_testable=True,
            synthetic_inputs=(),
            assertion_sketch=(),
            rationale="r",
            confidence="high",  # type: ignore[call-arg]
        )


# --------------------------------------------------------------------------
# required-field — PlanStep without test_sketch raises ValidationError (ac-002)
# --------------------------------------------------------------------------


def test_plan_step_without_test_sketch_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        PlanStep(
            id="a",
            depends_on=(),
            io=PlanStepIO(
                inputs=(PlanStepInput(name="raw", source_step=None),),
                outputs=("result",),
            ),
            artifacts=(),
            api_refs=("molpy.System",),
            composition_notes="step",
            checks=(),
            retry_policy=RetryPolicy(on=()),
            rollback=None,
            approval_gate=ApprovalGate.approve_execution,
            estimated_cost_usd=None,
            risk_level=RiskLevel.low,
            unknowns=(),
        )  # type: ignore[call-arg]


# ── inputs/outputs accept free-form names — codegen sanitises ─────────


@pytest.mark.parametrize(
    "name",
    [
        "data.peo",  # filename with dot
        "path/to/file",  # path
        "with-hyphen",  # hyphen
        "peo_chain",  # plain identifier
        "data_peo",  # snake_case
        "atomistic",  # word
    ],
)
def test_plan_step_io_accepts_free_form_names(name: str) -> None:
    """Plan-side names are free-form; codegen sanitises to identifiers."""
    PlanStepIO(
        inputs=(PlanStepInput(name=name, source_step=None),),
        outputs=(name,),
    )
