"""Tests for the relocated deterministic ``WorkflowContract`` normalizer (ac-003)."""

from __future__ import annotations

from molexp.agent.modes.author.contract_normalize import (
    derive_dependencies,
    normalize_contract,
)
from molexp.workflow import TaskInputSpec, TaskIO, TaskOutputSpec, WorkflowContract


def _contract(*task_io: TaskIO) -> WorkflowContract:
    return WorkflowContract(workflow_id="wf_test", task_io=task_io)


def test_coerce_long_form_source_to_bare_task_id() -> None:
    contract = _contract(
        TaskIO(task_id="a", outputs=(TaskOutputSpec(name="x", type="object"),)),
        TaskIO(
            task_id="b",
            inputs=(TaskInputSpec(name="x", type="object", source="a.x"),),
        ),
    )
    report = normalize_contract(contract)
    b_io = next(t for t in report.contract.task_io if t.task_id == "b")
    assert b_io.inputs[0].source == "a"
    assert report.ok


def test_infer_missing_source_from_unique_upstream_output() -> None:
    contract = _contract(
        TaskIO(task_id="a", outputs=(TaskOutputSpec(name="payload", type="object"),)),
        TaskIO(
            task_id="b",
            inputs=(TaskInputSpec(name="payload", type="object", source=None),),
        ),
    )
    report = normalize_contract(contract)
    b_io = next(t for t in report.contract.task_io if t.task_id == "b")
    assert b_io.inputs[0].source == "a"
    assert report.ok


def test_unresolvable_source_is_reported_not_dropped() -> None:
    contract = _contract(
        TaskIO(
            task_id="b",
            inputs=(TaskInputSpec(name="payload", type="object", source=None),),
        ),
    )
    report = normalize_contract(contract)
    assert not report.ok
    assert any(issue.kind == "unresolved_source" for issue in report.issues)


def test_derive_dependencies_returns_ordered_distinct_sources() -> None:
    contract = _contract(
        TaskIO(task_id="a", outputs=(TaskOutputSpec(name="x", type="object"),)),
        TaskIO(task_id="b", outputs=(TaskOutputSpec(name="y", type="object"),)),
        TaskIO(
            task_id="c",
            inputs=(
                TaskInputSpec(name="x", type="object", source="a"),
                TaskInputSpec(name="y", type="object", source="b"),
            ),
        ),
    )
    deps = derive_dependencies(contract)
    assert deps["c"] == ("a", "b")
    assert deps["a"] == ()


def test_normalize_does_not_mutate_input_contract() -> None:
    contract = _contract(
        TaskIO(task_id="a", outputs=(TaskOutputSpec(name="x", type="object"),)),
        TaskIO(
            task_id="b",
            inputs=(TaskInputSpec(name="x", type="object", source="a.x"),),
        ),
    )
    normalize_contract(contract)
    assert contract.task_io[1].inputs[0].source == "a.x"


def test_dangling_produced_by_is_reported() -> None:
    from molexp.workflow import ArtifactDecl

    contract = _contract(
        TaskIO(
            task_id="a",
            artifacts=(ArtifactDecl(path="out.txt", produced_by="ghost"),),
        ),
    )
    report = normalize_contract(contract)
    assert any(issue.kind == "dangling_produced_by" for issue in report.issues)
