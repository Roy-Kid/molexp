"""RED-phase tests for the diff cluster of
``molexp.agent.modes._planning``.

The package does not exist yet; these tests fail at collection until the
implementation lands.

Covers, per the testing rules:

- Basics  — enum membership; full valid ``PlanDiff`` construction +
  ``model_dump(mode="json")`` round-trip.
- Edge cases — ``extra="forbid"`` rejects an unknown field;
  ``PlanNodeOp`` cross-field validation (remove-with-step and
  add/replace-without-step both raise ``ValidationError``).
- Immutability — ``frozen=True`` rejects attribute assignment;
  ``PlanDiff.apply`` never mutates its input graph.
- Logic — ``PlanDiff.apply`` performs add / remove / replace and
  preserves the order of unaffected steps.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from molexp.agent.modes._planning import (
    ApprovalGate,
    DiffOpKind,
    IsolatedTestSketch,
    PlanCheck,
    PlanDiff,
    PlanGraph,
    PlanNodeOp,
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
        artifacts=(PlanStepArtifact(path=f"{step_id}.json", description="output"),),
        api_refs=("molpy.System",),
        composition_notes="step",
        checks=(PlanCheck(name="schema", description="valid", blocking=True),),
        retry_policy=RetryPolicy(on=()),
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


def _graph() -> PlanGraph:
    # a, b, c in declaration order
    return PlanGraph(
        plan_id="plan-1",
        intent_ref=None,
        steps=(_step("a"), _step("b"), _step("c")),
        state=PlanState.draft_plan,
        compiled_contract_ref=None,
        notes="",
    )


def _diff(*operations: PlanNodeOp) -> PlanDiff:
    return PlanDiff(
        failed_invariant="acyclic",
        affected_nodes=tuple(op.node_id for op in operations),
        operations=operations,
        rationale="repair",
        reused=(),
        invalidated=(),
    )


# --------------------------------------------------------------------------
# basics — enums
# --------------------------------------------------------------------------


def test_approval_gate_members() -> None:
    assert {m.value for m in ApprovalGate} == {
        "approve_direction",
        "approve_materialization",
        "approve_execution",
    }


def test_diff_op_kind_members() -> None:
    assert {m.value for m in DiffOpKind} == {"add", "remove", "replace"}


# --------------------------------------------------------------------------
# basics — PlanNodeOp construction
# --------------------------------------------------------------------------


def test_plan_node_op_add_requires_step() -> None:
    op = PlanNodeOp(kind=DiffOpKind.add, node_id="d", step=_step("d"))
    assert op.kind is DiffOpKind.add
    assert op.step is not None
    assert op.step.id == "d"


def test_plan_node_op_remove_has_no_step() -> None:
    op = PlanNodeOp(kind=DiffOpKind.remove, node_id="b", step=None)
    assert op.kind is DiffOpKind.remove
    assert op.step is None


def test_plan_node_op_replace_requires_step() -> None:
    op = PlanNodeOp(kind=DiffOpKind.replace, node_id="b", step=_step("b"))
    assert op.kind is DiffOpKind.replace
    assert op.step is not None


# --------------------------------------------------------------------------
# basics — PlanDiff construction + JSON round-trip
# --------------------------------------------------------------------------


def test_plan_diff_full_construction() -> None:
    diff = _diff(
        PlanNodeOp(kind=DiffOpKind.remove, node_id="b", step=None),
    )
    assert diff.failed_invariant == "acyclic"
    assert diff.affected_nodes == ("b",)
    assert len(diff.operations) == 1


def test_plan_diff_json_round_trip() -> None:
    diff = _diff(
        PlanNodeOp(kind=DiffOpKind.add, node_id="d", step=_step("d")),
        PlanNodeOp(kind=DiffOpKind.remove, node_id="a", step=None),
    )
    dumped = diff.model_dump(mode="json")
    restored = PlanDiff.model_validate(dumped)
    assert restored == diff


def test_plan_diff_json_dump_is_jsonable() -> None:
    diff = _diff(PlanNodeOp(kind=DiffOpKind.remove, node_id="a", step=None))
    dumped = diff.model_dump(mode="json")
    assert dumped["operations"][0]["kind"] == "remove"
    assert dumped["operations"][0]["step"] is None


# --------------------------------------------------------------------------
# edge cases — extra="forbid"
# --------------------------------------------------------------------------


def test_plan_node_op_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        PlanNodeOp(
            kind=DiffOpKind.remove,
            node_id="b",
            step=None,
            reason="x",  # type: ignore[call-arg]
        )


def test_plan_diff_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        PlanDiff(
            failed_invariant="acyclic",
            affected_nodes=(),
            operations=(),
            rationale="r",
            reused=(),
            invalidated=(),
            author="bob",  # type: ignore[call-arg]
        )


# --------------------------------------------------------------------------
# edge cases — PlanNodeOp cross-field validation
# --------------------------------------------------------------------------


def test_plan_node_op_remove_with_step_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PlanNodeOp(kind=DiffOpKind.remove, node_id="b", step=_step("b"))


def test_plan_node_op_add_without_step_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PlanNodeOp(kind=DiffOpKind.add, node_id="d", step=None)


def test_plan_node_op_replace_without_step_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PlanNodeOp(kind=DiffOpKind.replace, node_id="b", step=None)


# --------------------------------------------------------------------------
# immutability — frozen=True
# --------------------------------------------------------------------------


def test_plan_node_op_is_frozen() -> None:
    op = PlanNodeOp(kind=DiffOpKind.remove, node_id="b", step=None)
    with pytest.raises(ValidationError):
        op.node_id = "c"  # type: ignore[misc]


def test_plan_diff_is_frozen() -> None:
    diff = _diff(PlanNodeOp(kind=DiffOpKind.remove, node_id="b", step=None))
    with pytest.raises(ValidationError):
        diff.rationale = "changed"  # type: ignore[misc]


# --------------------------------------------------------------------------
# logic — PlanDiff.apply add / remove / replace
# --------------------------------------------------------------------------


def test_apply_add_appends_step() -> None:
    graph = _graph()
    diff = _diff(PlanNodeOp(kind=DiffOpKind.add, node_id="d", step=_step("d")))
    result = diff.apply(graph)
    assert tuple(s.id for s in result.steps) == ("a", "b", "c", "d")


def test_apply_remove_drops_matching_step() -> None:
    graph = _graph()
    diff = _diff(PlanNodeOp(kind=DiffOpKind.remove, node_id="b", step=None))
    result = diff.apply(graph)
    assert tuple(s.id for s in result.steps) == ("a", "c")


def test_apply_replace_swaps_matching_step() -> None:
    graph = _graph()
    replacement = _step("b", depends_on=("a",))
    diff = _diff(PlanNodeOp(kind=DiffOpKind.replace, node_id="b", step=replacement))
    result = diff.apply(graph)
    assert tuple(s.id for s in result.steps) == ("a", "b", "c")
    swapped = result.step_by_id("b")
    assert swapped is not None
    assert swapped.depends_on == ("a",)


def test_apply_preserves_order_of_unaffected_steps() -> None:
    graph = _graph()
    diff = _diff(
        PlanNodeOp(kind=DiffOpKind.replace, node_id="a", step=_step("a")),
        PlanNodeOp(kind=DiffOpKind.add, node_id="d", step=_step("d")),
    )
    result = diff.apply(graph)
    # b and c keep their relative position; a stays first, d appended last.
    assert tuple(s.id for s in result.steps) == ("a", "b", "c", "d")


def test_apply_returns_new_plan_graph_instance() -> None:
    graph = _graph()
    diff = _diff(PlanNodeOp(kind=DiffOpKind.remove, node_id="b", step=None))
    result = diff.apply(graph)
    assert isinstance(result, PlanGraph)
    assert result is not graph


# --------------------------------------------------------------------------
# immutability — apply never mutates its input
# --------------------------------------------------------------------------


def test_apply_does_not_mutate_input_graph() -> None:
    graph = _graph()
    before = tuple(s.id for s in graph.steps)
    diff = _diff(
        PlanNodeOp(kind=DiffOpKind.remove, node_id="b", step=None),
        PlanNodeOp(kind=DiffOpKind.add, node_id="d", step=_step("d")),
    )
    diff.apply(graph)
    assert tuple(s.id for s in graph.steps) == before
    assert before == ("a", "b", "c")
