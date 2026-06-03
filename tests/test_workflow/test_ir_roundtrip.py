"""IR (JSON) ↔ Workflow round-trip tests.

These verify that:
- ``CompiledWorkflow.from_ir(d)`` produces a runnable spec.
- ``Workflow.to_ir()`` is the inverse for IR-built specs.
- Python-built specs marked with ``task_type=`` slugs round-trip too.
- The recomputed ``workflow_id`` (topology hash) is preserved.
"""

from __future__ import annotations

import pytest

from molexp.workflow import (
    CompiledWorkflow,
    TaskTypeRegistry,
    WorkflowCompiler,
    WorkflowRuntime,
    default_codec,
    default_registry,
)


@pytest.fixture
def registry() -> TaskTypeRegistry:
    """Use the module-level registry; the demo slugs are pre-registered."""
    return default_registry


def _ir_constant_add() -> dict:
    """`a = 2`, `b = 3`, `c = a + b` — a tiny diamond-less DAG."""
    return {
        "workflow_id": "workflow_00000000",
        "name": "constant_add",
        "task_configs": [
            {
                "task_id": "a",
                "task_type": "core.constant",
                "config": {"value": 2},
                "status": "pending",
            },
            {
                "task_id": "b",
                "task_type": "core.constant",
                "config": {"value": 3},
                "status": "pending",
            },
            {"task_id": "c", "task_type": "core.add", "config": {}, "status": "pending"},
        ],
        "links": [
            {"source": "a", "target": "c", "mapping": {}, "status": "pending"},
            {"source": "b", "target": "c", "mapping": {}, "status": "pending"},
        ],
        "metadata": {"label": None, "description": None, "tags": [], "custom": {}},
    }


class TestFromDict:
    def test_topology_is_recovered(self, registry: TaskTypeRegistry) -> None:
        spec = CompiledWorkflow.from_ir(_ir_constant_add(), registry=registry)
        names = {t.name for t in spec._tasks}
        assert names == {"a", "b", "c"}

        c = next(t for t in spec._tasks if t.name == "c")
        assert sorted(c.depends_on) == ["a", "b"]

    def test_unknown_slug_raises(self, registry: TaskTypeRegistry) -> None:
        ir = _ir_constant_add()
        ir["task_configs"][0]["task_type"] = "nonexistent.thing"
        with pytest.raises(KeyError, match="nonexistent.thing"):  # noqa: RUF043
            CompiledWorkflow.from_ir(ir, registry=registry)

    def test_dangling_link_raises(self, registry: TaskTypeRegistry) -> None:
        ir = _ir_constant_add()
        ir["links"].append({"source": "ghost", "target": "c", "mapping": {}, "status": "pending"})
        with pytest.raises(ValueError, match="ghost"):
            CompiledWorkflow.from_ir(ir, registry=registry)


class TestExecute:
    @pytest.mark.asyncio
    async def test_execute_constant_add(self, registry: TaskTypeRegistry) -> None:
        spec = CompiledWorkflow.from_ir(_ir_constant_add(), registry=registry)
        result = await WorkflowRuntime().execute(spec)
        # Final task `c` should receive a + b = 5
        assert result.outputs["c"] == 5.0

    @pytest.mark.asyncio
    async def test_execute_constant_multiply(self, registry: TaskTypeRegistry) -> None:
        ir = {
            "workflow_id": "workflow_00000000",
            "name": "k_times",
            "task_configs": [
                {
                    "task_id": "k",
                    "task_type": "core.constant",
                    "config": {"value": 7},
                    "status": "pending",
                },
                {
                    "task_id": "tripled",
                    "task_type": "core.multiply",
                    "config": {"factor": 3.0},
                    "status": "pending",
                },
            ],
            "links": [
                {"source": "k", "target": "tripled", "mapping": {}, "status": "pending"},
            ],
            "metadata": {"label": None, "description": None, "tags": [], "custom": {}},
        }
        spec = CompiledWorkflow.from_ir(ir, registry=registry)
        result = await WorkflowRuntime().execute(spec)
        assert result.outputs["tripled"] == 21.0


class TestRoundtrip:
    def test_to_dict_then_from_dict(self, registry: TaskTypeRegistry) -> None:
        original = CompiledWorkflow.from_ir(_ir_constant_add(), registry=registry)
        ir = original.to_ir()
        rebuilt = CompiledWorkflow.from_ir(ir, registry=registry)
        assert rebuilt.workflow_id == original.workflow_id
        assert rebuilt.name == original.name
        assert {t.name for t in rebuilt._tasks} == {t.name for t in original._tasks}

    def test_python_built_with_slug_serializes(self, registry: TaskTypeRegistry) -> None:
        from molexp.workflow.registry import _Add, _Constant

        wf = WorkflowCompiler(name="py_built")
        wf.add(_Constant(value=4), name="four", task_type="core.constant", config={"value": 4})
        wf.add(_Constant(value=6), name="six", task_type="core.constant", config={"value": 6})
        wf.add(
            _Add(),
            name="sum",
            depends_on=["four", "six"],
            task_type="core.add",
            config={},
        )
        spec = wf.compile()
        ir = spec.to_ir()
        # IR reflects the topology faithfully
        assert ir["name"] == "py_built"
        assert {t["task_id"] for t in ir["task_configs"]} == {"four", "six", "sum"}
        sources_for_sum = sorted(link["source"] for link in ir["links"] if link["target"] == "sum")
        assert sources_for_sum == ["four", "six"]

    def test_to_dict_rejects_unslugged_tasks(self) -> None:
        from molexp.workflow.registry import _Constant

        wf = WorkflowCompiler(name="unslugged")
        wf.add(_Constant(value=1), name="lonely")  # no task_type passed
        spec = wf.compile()
        with pytest.raises(ValueError, match="task_type slug"):
            spec.to_ir()


def _ir_excluding_id(spec: CompiledWorkflow) -> dict:
    """Serialize ``spec`` to wire IR, dropping the topology-hash ``workflow_id``.

    ``workflow_id`` is a content hash recomputed from the (post-lowering)
    task list, so it legitimately differs across a serialize/reload cycle
    even when the topology is identical — the golden test normalizes it for
    the same reason. Everything else must round-trip exactly.
    """
    ir = dict(default_codec.spec_to_ir(spec))
    ir.pop("workflow_id", None)
    return ir


class TestTypedEdgeRoundtrip:
    """Typed-edge IR: data / control / branch / loop / parallel round-trip.

    Together these cover all five ``kind`` values. ``spec.to_ir()`` no longer
    rejects control flow (the former ``codec.py`` rejection is gone); the
    structured ``entries`` / ``loops`` / ``parallels`` arrays make the
    reload lossless. pg-lowering's reachability rules make a single
    all-five-in-one graph impractical, so each kind rides a real, compilable
    workflow.
    """

    def _slug(
        self,
        wf: WorkflowCompiler,
        name: str,
        value: int,
        deps: list[str] | None = None,
        **kw: object,
    ) -> None:
        from molexp.workflow.registry import _Constant

        wf.add(
            _Constant(value=value),
            name=name,
            depends_on=deps,
            task_type="core.constant",
            config={"value": value},
            **kw,
        )

    def test_branch_and_entry_round_trip(self) -> None:
        """A spec with wf.entry + wf.branch — previously rejected — now round-trips."""
        wf = WorkflowCompiler(name="branchy", entry="fetch")
        self._slug(wf, "fetch", 1)
        self._slug(wf, "validate", 2, deps=["fetch"], routes={"ok": "publish", "fail": "rollback"})
        self._slug(wf, "publish", 3, deps=["validate"])
        self._slug(wf, "rollback", 4, deps=["validate"])
        spec = wf.compile()

        ir = default_codec.spec_to_ir(spec)  # does not raise
        assert ir["entries"] == ["fetch"]
        assert {link["kind"] for link in ir["links"]} == {"data", "branch"}
        branch = {
            (link["source"], link["condition"], link["target"])
            for link in ir["links"]
            if link["kind"] == "branch"
        }
        assert branch == {("validate", "ok", "publish"), ("validate", "fail", "rollback")}

        rebuilt = default_codec.ir_to_spec(ir)
        assert tuple(sorted(rebuilt._branch_edges)) == tuple(sorted(spec._branch_edges))
        assert rebuilt._entries == spec._entries
        assert _ir_excluding_id(rebuilt) == _ir_excluding_id(spec)

    def test_control_edge_round_trips(self) -> None:
        wf = WorkflowCompiler(name="cf", entry="a")
        self._slug(wf, "a", 1)
        self._slug(wf, "b", 2)
        wf.control("a", "b")
        spec = wf.compile()

        ir = default_codec.spec_to_ir(spec)
        assert {link["kind"] for link in ir["links"]} == {"control"}
        rebuilt = default_codec.ir_to_spec(ir)
        assert tuple(rebuilt._control_edges) == tuple(spec._control_edges) == (("a", "b"),)
        assert _ir_excluding_id(rebuilt) == _ir_excluding_id(spec)

    def test_loop_and_parallel_round_trip(self) -> None:
        wf = WorkflowCompiler(name="lp")
        self._slug(wf, "seed", 0)
        self._slug(wf, "compute", 1, deps=["seed"])
        self._slug(wf, "check_done", 2, deps=["compute"])
        self._slug(wf, "items", 3)
        self._slug(wf, "process", 4)
        self._slug(wf, "gather", 5, deps=["items"])
        wf.loop(body=["compute"], until="check_done", max_iters=10)
        wf.parallel(map_over="items", body="process", join="gather", max_concurrency=4)
        spec = wf.compile()

        ir = default_codec.spec_to_ir(spec)
        assert ir["loops"] == [
            {"body": ["compute"], "until": "check_done", "max_iters": 10, "on_exit": "_end"}
        ]
        assert ir["parallels"] == [
            {"map_over": "items", "body": "process", "join": "gather", "max_concurrency": 4}
        ]
        rebuilt = default_codec.ir_to_spec(ir)
        assert [(ld.body, ld.until, ld.max_iters, ld.on_exit) for ld in rebuilt._loops] == [
            (ld.body, ld.until, ld.max_iters, ld.on_exit) for ld in spec._loops
        ]
        assert [(p.map_over, p.body, p.join, p.max_concurrency) for p in rebuilt._parallels] == [
            (p.map_over, p.body, p.join, p.max_concurrency) for p in spec._parallels
        ]
        assert _ir_excluding_id(rebuilt) == _ir_excluding_id(spec)


class TestNodePosition:
    """Node ``position`` is carried through the IR but never hashed."""

    def _ir_with_positions(self, positions: dict[str, dict]) -> dict:
        ir = _ir_constant_add()
        for tc in ir["task_configs"]:
            tc["position"] = positions[tc["task_id"]]
        for link in ir["links"]:
            link["kind"] = "data"
        return ir

    def test_position_round_trips_in_serialized_ir(self, registry: TaskTypeRegistry) -> None:
        ir = self._ir_with_positions(
            {"a": {"x": 1.0, "y": 2.0}, "b": {"x": 3.0, "y": 4.0}, "c": {"x": 5.0, "y": 6.0}}
        )
        spec = CompiledWorkflow.from_ir(ir, registry=registry)
        out = spec.to_ir()
        by_id = {tc["task_id"]: tc for tc in out["task_configs"]}
        assert by_id["a"]["position"] == {"x": 1.0, "y": 2.0}
        assert by_id["c"]["position"] == {"x": 5.0, "y": 6.0}

    def test_position_excluded_from_snapshot_hash(self, registry: TaskTypeRegistry) -> None:
        spec_left = CompiledWorkflow.from_ir(
            self._ir_with_positions(
                {"a": {"x": 0.0, "y": 0.0}, "b": {"x": 0.0, "y": 0.0}, "c": {"x": 0.0, "y": 0.0}}
            ),
            registry=registry,
        )
        spec_right = CompiledWorkflow.from_ir(
            self._ir_with_positions(
                {"a": {"x": 99.0, "y": 88.0}, "b": {"x": 7.0, "y": 7.0}, "c": {"x": -3.0, "y": 5.0}}
            ),
            registry=registry,
        )
        for name in ("a", "b", "c"):
            assert spec_left.snapshots[name].key == spec_right.snapshots[name].key
