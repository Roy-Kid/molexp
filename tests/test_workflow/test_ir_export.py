"""Tests for the full-graph IR + Mermaid export.

Covers :meth:`Workflow.to_ir` (→ :class:`WorkflowGraphIR`) and
:meth:`Workflow.to_mermaid`. Unlike the DAG-only wire IR
(:meth:`Workflow.to_dict`), this surface captures the complete compiled
topology — entries, control edges, branch routes, loops, parallels — and
serializes decorator-defined workflows that carry no ``task_type`` slug.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from molexp.workflow import (
    GraphLoopIR,
    GraphParallelIR,
    GraphTaskIR,
    WorkflowBuilder,
    WorkflowGraphIR,
)


def _branchy_builder() -> WorkflowBuilder:
    wf = WorkflowBuilder(name="pipeline", mode="batch", version="3", entry="fetch")

    @wf.task
    async def fetch(ctx):
        return 1

    @wf.task(depends_on=["fetch"], routes={"ok": "publish", "fail": "rollback"})
    async def validate(ctx):
        return 2

    @wf.task(depends_on=["validate"])
    async def publish(ctx):
        return 3

    @wf.task(depends_on=["validate"])
    async def rollback(ctx):
        return 4

    @wf.actor(depends_on=["fetch"])
    async def stream(ctx):
        yield 1

    return wf


# ── to_ir ────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_to_ir_returns_workflow_graph_ir_with_metadata():
    spec = _branchy_builder().build()
    ir = spec.to_ir()
    assert isinstance(ir, WorkflowGraphIR)
    assert ir.name == "pipeline"
    assert ir.workflow_id == spec.workflow_id
    assert ir.version == "3"
    assert ir.mode == "batch"


@pytest.mark.unit
def test_to_ir_captures_all_tasks_without_requiring_task_type_slug():
    """Decorator tasks have no slug; the full IR exports them anyway."""
    ir = _branchy_builder().build().to_ir()
    names = [t.name for t in ir.tasks]
    assert names == ["fetch", "validate", "publish", "rollback", "stream"]
    assert all(t.task_type is None for t in ir.tasks)


@pytest.mark.unit
def test_to_ir_marks_actor_nodes():
    ir = _branchy_builder().build().to_ir()
    by_name = {t.name: t for t in ir.tasks}
    assert by_name["stream"].is_actor is True
    assert by_name["fetch"].is_actor is False


@pytest.mark.unit
def test_to_ir_captures_dependencies_entries_and_branches():
    ir = _branchy_builder().build().to_ir()
    by_name = {t.name: t for t in ir.tasks}
    assert by_name["validate"].depends_on == ("fetch",)
    assert ir.entries == ("fetch",)
    assert ("validate", "ok", "publish") in ir.branch_edges
    assert ("validate", "fail", "rollback") in ir.branch_edges


@pytest.mark.unit
def test_to_ir_captures_control_edges():
    wf = WorkflowBuilder(name="cf")

    @wf.task
    async def a(ctx):
        return 1

    @wf.task
    async def b(ctx):
        return 2

    wf.entry("a")
    wf.control(src="a", to="b")
    ir = wf.build().to_ir()
    assert ("a", "b") in ir.control_edges


@pytest.mark.unit
def test_to_ir_captures_loops_and_parallels():
    wf = WorkflowBuilder(name="lp")

    @wf.task
    async def seed(ctx):
        return 0

    @wf.task(depends_on=["seed"])
    async def compute(ctx):
        return 1

    @wf.task(depends_on=["compute"])
    async def check_done(ctx):
        return 2

    @wf.task
    async def items(ctx):
        return [1, 2]

    @wf.task
    async def process(ctx):
        return 3

    wf.loop(body=["compute"], until="check_done", max_iters=10)
    wf.parallel(map_over="items", body="process", join="check_done", max_concurrency=4)
    ir = wf.build().to_ir()

    assert ir.loops == (
        GraphLoopIR(body=("compute",), until="check_done", max_iters=10, on_exit="_end"),
    )
    assert ir.parallels == (
        GraphParallelIR(map_over="items", body="process", join="check_done", max_concurrency=4),
    )


@pytest.mark.unit
def test_to_ir_carries_config_for_oop_tasks():
    from molexp.workflow import Task

    class Adder(Task):
        async def execute(self, ctx):
            return 1

    wf = WorkflowBuilder(name="oop")
    wf.add(Adder(), name="adder", task_type="core.add", config={"value": 10})
    ir = wf.build().to_ir()
    adder = next(t for t in ir.tasks if t.name == "adder")
    assert adder.task_type == "core.add"
    assert adder.config == {"value": 10}


# ── IR immutability + JSON round-trip ────────────────────────────────────────


@pytest.mark.unit
def test_workflow_graph_ir_is_frozen():
    ir = _branchy_builder().build().to_ir()
    with pytest.raises(ValidationError):
        ir.name = "mutated"  # type: ignore[misc]


@pytest.mark.unit
def test_workflow_graph_ir_json_round_trip_is_exact():
    ir = _branchy_builder().build().to_ir()
    restored = WorkflowGraphIR.model_validate_json(ir.model_dump_json())
    assert restored == ir


@pytest.mark.unit
def test_graph_task_ir_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        GraphTaskIR(name="x", bogus=1)  # type: ignore[call-arg]


# ── to_mermaid ───────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_to_mermaid_emits_flowchart_header_and_trailing_newline():
    out = _branchy_builder().build().to_mermaid()
    assert out.startswith("flowchart LR\n")
    assert out.endswith("\n")


@pytest.mark.unit
def test_to_mermaid_honours_direction():
    out = _branchy_builder().build().to_mermaid(direction="TD")
    assert out.startswith("flowchart TD\n")


@pytest.mark.unit
def test_to_mermaid_renders_nodes_dependencies_and_entry_marker():
    out = _branchy_builder().build().to_mermaid()
    assert 'n_fetch["fetch"]' in out
    assert "__start((start))" in out
    assert "__start --> n_fetch" in out
    assert "n_fetch --> n_validate" in out


@pytest.mark.unit
def test_to_mermaid_renders_actor_as_stadium():
    out = _branchy_builder().build().to_mermaid()
    assert 'n_stream(["stream"])' in out


@pytest.mark.unit
def test_to_mermaid_labels_branch_routes_and_suppresses_plain_duplicate():
    out = _branchy_builder().build().to_mermaid()
    assert "n_validate -->|ok| n_publish" in out
    assert "n_validate -->|fail| n_rollback" in out
    # The plain dependency edge on the same pair is suppressed in favour of
    # the labeled branch edge.
    assert "n_validate --> n_publish" not in out
    assert "n_validate --> n_rollback" not in out


@pytest.mark.unit
def test_to_mermaid_renders_loop_and_parallel_edges():
    wf = WorkflowBuilder(name="lp")

    @wf.task
    async def compute(ctx):
        return 1

    @wf.task(depends_on=["compute"])
    async def check_done(ctx):
        return 2

    @wf.task
    async def items(ctx):
        return [1]

    @wf.task
    async def process(ctx):
        return 3

    wf.loop(body=["compute"], until="check_done", max_iters=5)
    wf.parallel(map_over="items", body="process", join="check_done", max_concurrency=3)
    out = wf.build().to_mermaid()
    assert "n_check_done -.->|continue| n_compute" in out
    assert "n_check_done -->|exit| n__end" in out
    assert "n_items -->|fan-out x3| n_process" in out
    assert "n_process -->|join| n_check_done" in out


@pytest.mark.unit
def test_to_mermaid_sanitizes_unsafe_task_names():
    wf = WorkflowBuilder(name="x")
    wf.add(_Noop(), name="step-one.v2", task_type="t", config={})
    out = wf.build().to_mermaid()
    assert "n_step_one_v2" in out
    # Original name preserved inside the display label.
    assert "step-one.v2" in out


class _Noop:
    async def execute(self, ctx):
        return None
