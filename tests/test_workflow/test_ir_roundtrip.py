"""IR (JSON) ↔ WorkflowSpec round-trip tests.

These verify that:
- ``WorkflowSpec.from_dict(d)`` produces a runnable spec.
- ``WorkflowSpec.to_dict()`` is the inverse for IR-built specs.
- Python-built specs marked with ``task_type=`` slugs round-trip too.
- The recomputed ``workflow_id`` (topology hash) is preserved.
"""

from __future__ import annotations

import pytest

from molexp.workflow import (
    TaskTypeRegistry,
    Workflow,
    WorkflowSpec,
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
        spec = WorkflowSpec.from_dict(_ir_constant_add(), registry=registry)
        names = {t.name for t in spec._tasks}
        assert names == {"a", "b", "c"}

        c = next(t for t in spec._tasks if t.name == "c")
        assert sorted(c.depends_on) == ["a", "b"]

    def test_unknown_slug_raises(self, registry: TaskTypeRegistry) -> None:
        ir = _ir_constant_add()
        ir["task_configs"][0]["task_type"] = "nonexistent.thing"
        with pytest.raises(KeyError, match="nonexistent.thing"):
            WorkflowSpec.from_dict(ir, registry=registry)

    def test_dangling_link_raises(self, registry: TaskTypeRegistry) -> None:
        ir = _ir_constant_add()
        ir["links"].append({"source": "ghost", "target": "c", "mapping": {}, "status": "pending"})
        with pytest.raises(ValueError, match="ghost"):
            WorkflowSpec.from_dict(ir, registry=registry)


class TestExecute:
    @pytest.mark.asyncio
    async def test_execute_constant_add(self, registry: TaskTypeRegistry) -> None:
        spec = WorkflowSpec.from_dict(_ir_constant_add(), registry=registry)
        result = await spec.execute()
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
        spec = WorkflowSpec.from_dict(ir, registry=registry)
        result = await spec.execute()
        assert result.outputs["tripled"] == 21.0


class TestRoundtrip:
    def test_to_dict_then_from_dict(self, registry: TaskTypeRegistry) -> None:
        original = WorkflowSpec.from_dict(_ir_constant_add(), registry=registry)
        ir = original.to_dict()
        rebuilt = WorkflowSpec.from_dict(ir, registry=registry)
        assert rebuilt.workflow_id == original.workflow_id
        assert rebuilt.name == original.name
        assert {t.name for t in rebuilt._tasks} == {t.name for t in original._tasks}

    def test_python_built_with_slug_serializes(self, registry: TaskTypeRegistry) -> None:
        from molexp.workflow.registry import _Add, _Constant

        wf = Workflow(name="py_built")
        wf.add(_Constant(value=4), name="four", task_type="core.constant", config={"value": 4})
        wf.add(_Constant(value=6), name="six", task_type="core.constant", config={"value": 6})
        wf.add(
            _Add(),
            name="sum",
            depends_on=["four", "six"],
            task_type="core.add",
            config={},
        )
        spec = wf.build()
        ir = spec.to_dict()
        # IR reflects the topology faithfully
        assert ir["name"] == "py_built"
        assert {t["task_id"] for t in ir["task_configs"]} == {"four", "six", "sum"}
        sources_for_sum = sorted(link["source"] for link in ir["links"] if link["target"] == "sum")
        assert sources_for_sum == ["four", "six"]

    def test_to_dict_rejects_unslugged_tasks(self) -> None:
        from molexp.workflow.registry import _Constant

        wf = Workflow(name="unslugged")
        wf.add(_Constant(value=1), name="lonely")  # no task_type passed
        spec = wf.build()
        with pytest.raises(ValueError, match="task_type slug"):
            spec.to_dict()

    def test_cyclic_spec_to_dict_rejected(self) -> None:
        """IR (`to_dict`) does not yet model control edges — cyclic specs must
        reject serialization rather than silently dropping the loop topology.

        Spec: .claude/specs/03-molexp-workflow-cycles.md "Out of scope" — IR
        serialization of control edges is deferred. Until then, attempting to
        round-trip a workflow that uses `wf.control` / `wf.branch` / `wf.entry`
        must raise — better than corrupting the persisted IR.
        """
        from molexp.workflow.registry import _Constant

        wf = Workflow(name="cyclic-ir")
        wf.add(_Constant(value=1), name="head", task_type="core.constant", config={"value": 1})
        wf.add(_Constant(value=2), name="tail", task_type="core.constant", config={"value": 2})
        wf.entry("head")
        wf.control("head", "tail")
        wf.branch("tail", "back", "head")  # cyclic — control edge loop
        spec = wf.build()
        with pytest.raises(ValueError, match="control edge"):
            spec.to_dict()
