"""IR serialization + cache identity for ``promote_callable`` bodies (regression).

``_EntryTask`` used to capture the live callable as its task config, so
``CompiledWorkflow.to_graph_ir()`` failed with a raw pydantic ValidationError
(``GraphTaskIR.config`` values must be JSON). The fix:

- an **importable** promoted callable serializes as a ``"module:qualname"``
  entrypoint ref in the graph IR (re-imported via importlib at execution time);
- a **non-importable** callable (lambda, closure, ``__main__``/REPL function)
  raises a clear, actionable error at IR time instead of the pydantic one;
- in-memory execution (``WorkflowRuntime().execute``) keeps working for
  non-importable callables exactly as before;
- the content-addressed cache snapshot keys on the *resolved* callable's
  source, never on the ref string, so editing the body still invalidates.
"""

from __future__ import annotations

import asyncio

import pytest

from molexp.workflow import WorkflowRuntime, promote_callable


# Module-level promoted body — importable via ``module:qualname``.
def _double(inputs, config):
    params = inputs["params"] if isinstance(inputs, dict) else {}
    return {"doubled": params.get("x", 0) * 2}


class TestGraphIRSerialization:
    def test_importable_body_serializes_as_entrypoint_ref(self) -> None:
        compiled = promote_callable(_double, "promoted")
        ir = compiled.to_graph_ir()
        (task_ir,) = ir.tasks
        assert task_ir.config["fn"] == f"{_double.__module__}:{_double.__qualname__}"

    def test_non_importable_body_raises_clear_error_at_ir_time(self) -> None:
        bias = 1

        def _local(inputs, config):
            return bias

        compiled = promote_callable(_local, "wf")
        with pytest.raises(ValueError, match="not importable"):
            compiled.to_graph_ir()


class TestSnapshotIdentity:
    def test_entrypoint_ref_and_callable_yield_same_snapshot_key(self) -> None:
        # A task reconstructed from the ref string must share the cache identity
        # of the one built from the live callable — the identity comes from the
        # resolved fn's source, not the ref string.
        from molexp.workflow.promote import _EntryTask
        from molexp.workflow.snapshot import TaskSnapshot

        ref = f"{_double.__module__}:{_double.__qualname__}"
        from_callable = TaskSnapshot.from_task_body("t", _EntryTask(_double))
        from_ref = TaskSnapshot.from_task_body("t", _EntryTask(ref))
        assert from_ref.key == from_callable.key


class TestInMemoryExecution:
    def test_non_importable_body_executes_in_memory(self) -> None:
        marker = {"value": 7}

        def _local(inputs, config):
            return {"seen": marker["value"]}

        compiled = promote_callable(_local, "wf")
        result = asyncio.run(WorkflowRuntime().execute(compiled))
        assert result.status == "succeeded"
        assert result.outputs["_local"] == {"seen": 7}
