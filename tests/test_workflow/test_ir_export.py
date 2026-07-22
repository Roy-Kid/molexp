"""Full-graph IR + Mermaid export — ``CompiledWorkflow.to_graph_ir`` /
``to_graph_mermaid`` (``molexp.workflow.ir``).

Unlike the DAG-only wire IR (``Workflow.to_ir``, covered by
``test_ir_roundtrip`` / ``test_codec``), this surface captures the complete
compiled topology — entries, control edges, branch routes, loops, parallels —
and serializes decorator-defined workflows that carry no ``task_type`` slug.
"""

from __future__ import annotations

import pytest

from molexp.workflow import (
    GraphLoopIR,
    GraphParallelIR,
    WorkflowCompiler,
    WorkflowGraphIR,
)


def _branchy_builder() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="pipeline", mode="batch", version="3", entry="fetch")

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


class _Noop:
    async def execute(self, ctx):
        return None


class TestToGraphIR:
    @pytest.mark.unit
    def test_exports_all_nodes_without_slug_and_marks_actors(self):
        """Decorator tasks carry no slug yet all export (in order, ``task_type``
        None), and actor nodes are flagged ``is_actor``."""
        spec = _branchy_builder().compile()
        ir = spec.to_graph_ir()

        assert isinstance(ir, WorkflowGraphIR)
        assert ir.name == "pipeline"
        assert ir.workflow_id == spec.workflow_id

        by_name = {t.name: t for t in ir.tasks}
        assert [t.name for t in ir.tasks] == ["fetch", "validate", "publish", "rollback", "stream"]
        assert all(t.task_type is None for t in ir.tasks)
        assert by_name["stream"].is_actor is True
        assert by_name["fetch"].is_actor is False

    @pytest.mark.unit
    def test_captures_entries_dependencies_and_branch_routes(self):
        ir = _branchy_builder().compile().to_graph_ir()
        by_name = {t.name: t for t in ir.tasks}
        assert by_name["validate"].depends_on == ("fetch",)
        assert ir.entries == ("fetch",)
        assert ("validate", "ok", "publish") in ir.branch_edges
        assert ("validate", "fail", "rollback") in ir.branch_edges

    @pytest.mark.unit
    def test_captures_control_edges(self):
        wf = WorkflowCompiler(name="cf")

        @wf.task
        async def a(ctx):
            return 1

        @wf.task
        async def b(ctx):
            return 2

        wf.entry("a")
        wf.control(src="a", to="b")
        ir = wf.compile().to_graph_ir()
        assert ("a", "b") in ir.control_edges

    @pytest.mark.unit
    def test_captures_loops_and_parallels(self):
        wf = WorkflowCompiler(name="lp")

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

        # A parallel-join and a loop-until cannot be fused onto one node, so
        # the join rides a distinct task; the IR features asserted (loop +
        # parallel captured) are unchanged.
        @wf.task(depends_on=["items"])
        async def gather(ctx):
            return 4

        wf.loop(body=["compute"], until="check_done", max_iters=10)
        wf.parallel(map_over="items", body="process", join="gather", max_concurrency=4)
        ir = wf.compile().to_graph_ir()

        assert ir.loops == (
            GraphLoopIR(body=("compute",), until="check_done", max_iters=10, on_exit="_end"),
        )
        assert ir.parallels == (
            GraphParallelIR(map_over="items", body="process", join="gather", max_concurrency=4),
        )

    @pytest.mark.unit
    def test_projects_unified_kind_tagged_edge_set(self):
        """``build_workflow_graph_ir`` projects the split collections into one
        ``kind``-tagged edge set: depends_on→data, branch routes→branch(+condition)."""
        ir = _branchy_builder().compile().to_graph_ir()
        data_edges = {(e.source, e.target) for e in ir.edges if e.kind == "data"}
        branch_edges = {(e.source, e.condition, e.target) for e in ir.edges if e.kind == "branch"}
        assert ("fetch", "validate") in data_edges
        assert ("validate", "ok", "publish") in branch_edges
        assert ("validate", "fail", "rollback") in branch_edges
        assert {e.kind for e in ir.edges} <= {"data", "control", "branch", "loop", "parallel"}

    @pytest.mark.unit
    def test_parallel_fanout_in_unified_edges_and_omitted_from_wire_ir(self):
        """ac-001/002: to_graph_ir tags both map_over→body and body→join edges
        kind="parallel"; to_ir's data-DAG wire format omits them (by-design)."""
        wf = WorkflowCompiler(name="par")

        @wf.task
        async def items(ctx):
            return [1, 2]

        @wf.task
        async def process(ctx):
            return 3

        @wf.task(depends_on=["items"])
        async def gather(ctx):
            return 4

        wf.parallel(map_over="items", body="process", join="gather")
        compiled = wf.compile()

        parallel_pairs = {
            (e.source, e.target) for e in compiled.to_graph_ir().edges if e.kind == "parallel"
        }
        assert ("items", "process") in parallel_pairs
        assert ("process", "gather") in parallel_pairs

        # The data-DAG wire format (to_ir) deliberately omits the parallel
        # fan-out: it never tags an edge "parallel", and the map_over→body edge
        # is absent.
        links = compiled.to_ir(strict=False).get("links", [])
        assert all(link.get("kind") != "parallel" for link in links if isinstance(link, dict))
        data_pairs = {(link["source"], link["target"]) for link in links if isinstance(link, dict)}
        assert ("items", "process") not in data_pairs

    @pytest.mark.unit
    def test_carries_node_position_from_wire_ir(self):
        """A position set on the wire IR survives into the full graph IR's nodes
        (exposed as a GraphNodePosition)."""
        from molexp.workflow import CompiledWorkflow, GraphNodePosition

        ir = {
            "workflow_id": "workflow_00000000",
            "name": "p",
            "task_configs": [
                {
                    "task_id": "k",
                    "task_type": "core.constant",
                    "config": {"value": 1},
                    "status": "pending",
                    "position": {"x": 12.5, "y": -3.0},
                }
            ],
            "links": [],
            "metadata": {"label": None, "description": None, "tags": [], "custom": {}},
        }
        graph_ir = CompiledWorkflow.from_ir(ir).to_graph_ir()
        node = next(t for t in graph_ir.tasks if t.name == "k")
        assert node.position == GraphNodePosition(x=12.5, y=-3.0)

    @pytest.mark.unit
    def test_carries_config_for_registered_oop_task(self):
        from molexp.workflow import Task
        from molexp.workflow.registry import default_registry

        class Adder(Task):
            def __init__(self, value: int = 0) -> None:
                self.value = value

            async def execute(self, ctx):
                return 1

        # Slug lives with the type, declared once; resolved at compile time.
        default_registry.register("test.adder", Adder)

        wf = WorkflowCompiler(name="oop")
        # Config is the instance's captured __init__ args — IR carries them.
        wf.add(Adder(value=10), name="adder")
        ir = wf.compile().to_graph_ir()
        adder = next(t for t in ir.tasks if t.name == "adder")
        assert adder.task_type == "test.adder"
        assert adder.config == {"value": 10}

    @pytest.mark.unit
    def test_json_round_trip_is_exact(self):
        ir = _branchy_builder().compile().to_graph_ir()
        restored = WorkflowGraphIR.model_validate_json(ir.model_dump_json())
        assert restored == ir

    @pytest.mark.unit
    def test_embeds_subworkflow_inner_graph(self):
        """A SubWorkflow node exposes the full inner WorkflowGraphIR under
        ``GraphTaskIR.subworkflow`` (UI drill-down); ordinary nodes carry
        ``subworkflow=None``. The embedding round-trips through JSON."""
        from molexp.workflow import SubWorkflow

        inner = WorkflowCompiler(name="inner")

        @inner.task
        async def load(ctx):
            return ctx.inputs

        @inner.task(depends_on=["load"])
        async def scale(ctx):
            return ctx.inputs

        outer = (
            WorkflowCompiler(name="outer")
            .add(SubWorkflow(inner), name="sub")
            .add(_Noop(), name="after", depends_on=["sub"])
            .compile()
        )
        ir = outer.to_graph_ir()
        by_name = {t.name: t for t in ir.tasks}

        assert by_name["after"].subworkflow is None

        sub_ir = by_name["sub"].subworkflow
        assert isinstance(sub_ir, WorkflowGraphIR)
        assert sub_ir.name == "inner"
        assert {t.name for t in sub_ir.tasks} == {"load", "scale"}

        # Round-trips through JSON (the wire contract for the UI).
        dumped = ir.model_dump(mode="json")
        sub_dumped = next(t for t in dumped["tasks"] if t["name"] == "sub")["subworkflow"]
        assert sub_dumped["name"] == "inner"
        assert WorkflowGraphIR.model_validate(dumped) == ir


class TestToGraphMermaid:
    @pytest.mark.unit
    def test_renders_nodes_entry_dependencies_and_branch_routes(self):
        """Nodes + start marker + dependency edges render; branch routes carry
        their label and suppress the plain duplicate edge on the same pair."""
        out = _branchy_builder().compile().to_graph_mermaid()

        assert 'n_fetch["fetch"]' in out
        assert "__start((start))" in out
        assert "__start --> n_fetch" in out
        assert "n_fetch --> n_validate" in out

        assert "n_validate -->|ok| n_publish" in out
        assert "n_validate -->|fail| n_rollback" in out
        assert "n_validate --> n_publish" not in out
        assert "n_validate --> n_rollback" not in out
