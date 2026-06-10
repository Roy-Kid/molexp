"""Tracked execution for ``promote_callable`` workflows (regression).

``_EntryTask`` used to capture the live callable as its task config, so
``CompiledWorkflow.to_graph_ir()`` — called by ``Experiment.run`` when it
records ``workflow_source`` on the experiment — failed with a raw pydantic
ValidationError (``GraphTaskIR.config`` values must be JSON). The fix:

- an **importable** promoted callable serializes as a ``"module:qualname"``
  entrypoint ref (round-tripped via importlib at execution time), so the
  tracked path (``ws.project(p).experiment(e).run(compiled, params=...)``)
  works end to end;
- a **non-importable** callable (lambda, closure, ``__main__``/REPL function)
  raises a clear, actionable error at IR time instead of the pydantic one;
- in-memory execution (``WorkflowRuntime().execute``) keeps working for
  non-importable callables exactly as before;
- the content-addressed cache snapshot keys on the *resolved* callable's
  source, never on the ref string, so editing the body still invalidates.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from molexp.workflow import WorkflowRuntime, default_binding_registry, promote_callable
from molexp.workspace import Workspace


@pytest.fixture(autouse=True)
def _clean_bindings():
    default_binding_registry.clear()
    yield
    default_binding_registry.clear()


# Module-level promoted bodies — importable via ``module:qualname``.


def _double(inputs, config):
    params = inputs["params"] if isinstance(inputs, dict) else {}
    return {"doubled": params.get("x", 0) * 2}


def _triple(inputs, config):
    params = inputs["params"] if isinstance(inputs, dict) else {}
    return {"tripled": params.get("x", 0) * 3}


class TestImportableTrackedPath:
    def test_experiment_run_records_entrypoint_ref(self, tmp_path):
        compiled = promote_callable(_double, "promoted")
        exp = (
            Workspace(tmp_path / "ws", name="ws")
            .project("p")
            .experiment("e")
            .run(compiled, params={"x": [2]})
        )
        ir = json.loads(exp.metadata.workflow_source)
        (task_ir,) = ir["tasks"]
        assert task_ir["config"]["fn"] == f"{_double.__module__}:{_double.__qualname__}"

    def test_tracked_execution_produces_results(self, tmp_path):
        compiled = promote_callable(_double, "promoted")
        exp = (
            Workspace(tmp_path / "ws", name="ws")
            .project("p")
            .experiment("e")
            .run(compiled, params={"x": [2]})
        )
        # Drive execution the way the CLI's in-process handler does: resolve
        # the bound spec and execute it under the run's RunContext.
        spec = default_binding_registry.for_experiment(exp)
        assert spec is not None
        (run,) = exp.list_runs()
        with run.start() as ctx:
            result = asyncio.run(WorkflowRuntime().execute(spec, run_context=ctx))
        assert result.status == "completed"
        assert result.outputs["_double"] == {"doubled": 4}
        assert run.status == "succeeded"

    def test_entrypoint_ref_round_trips_to_same_snapshot(self):
        # A task reconstructed from the ref string must share the cache
        # identity of the one built from the live callable — and that identity
        # must come from the resolved fn's source, not the ref string.
        from molexp.workflow.promote import _EntryTask
        from molexp.workflow.snapshot import TaskSnapshot

        ref = f"{_double.__module__}:{_double.__qualname__}"
        from_callable = TaskSnapshot.from_task_body("t", _EntryTask(_double))
        from_ref = TaskSnapshot.from_task_body("t", _EntryTask(ref))
        assert from_ref.key == from_callable.key

    def test_snapshot_discriminates_between_promoted_bodies(self):
        # Same _EntryTask wrapper, different wrapped fns → different keys
        # (the config hash carries the resolved body's normalized source).
        from molexp.workflow.promote import _EntryTask
        from molexp.workflow.snapshot import TaskSnapshot

        a = TaskSnapshot.from_task_body("t", _EntryTask(_double))
        b = TaskSnapshot.from_task_body("t", _EntryTask(_triple))
        assert a.key != b.key


class TestNonImportableRaisesClearError:
    def test_closure_raises_clear_error_at_ir_time(self):
        bias = 1

        def _local(inputs, config):
            return bias

        compiled = promote_callable(_local, "wf")
        with pytest.raises(ValueError, match="not importable"):
            compiled.to_graph_ir()

    def test_lambda_error_is_actionable_not_pydantic(self):
        compiled = promote_callable(lambda inputs, config: None, "wf")  # noqa: ARG005
        with pytest.raises(ValueError) as excinfo:
            compiled.to_graph_ir()
        msg = str(excinfo.value)
        assert "module scope" in msg
        assert "WorkflowRuntime" in msg
        assert "ValidationError" not in type(excinfo.value).__name__

    def test_experiment_run_surfaces_the_same_clear_error(self, tmp_path):
        def _local(inputs, config):
            return None

        compiled = promote_callable(_local, "wf")
        ws = Workspace(tmp_path / "ws", name="ws")
        with pytest.raises(ValueError, match="not importable"):
            ws.project("p").experiment("e").run(compiled)


class TestInMemoryPathUnchanged:
    def test_closure_executes_in_memory(self):
        marker = {"value": 7}

        def _local(inputs, config):
            return {"seen": marker["value"]}

        compiled = promote_callable(_local, "wf")
        result = asyncio.run(WorkflowRuntime().execute(compiled))
        assert result.status == "completed"
        assert result.outputs["_local"] == {"seen": 7}

    def test_lambda_executes_in_memory(self):
        compiled = promote_callable(lambda inputs, config: 42, "wf")  # noqa: ARG005
        result = asyncio.run(WorkflowRuntime().execute(compiled))
        assert result.status == "completed"
        assert result.outputs["<lambda>"] == 42
