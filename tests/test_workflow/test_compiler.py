"""Tests for :meth:`molexp.workflow.compiler.WorkflowCompiler.compile`.

``compile()`` lowers the registrations exactly once and emits a single frozen
:class:`CompiledWorkflow` carrying the executable graph, per-task snapshots, the
workflow version, and (when an experiment is supplied) an experiment binding.
"""

from __future__ import annotations

import pytest

from molexp.workflow import CompiledWorkflow, WorkflowCompiler
from molexp.workflow.version import WorkflowVersion


class _Exp:
    """Minimal experiment stand-in (duck-typed `.id`)."""

    def __init__(self, exp_id: str) -> None:
        self.id = exp_id


class TestWorkflowCompilerCompile:
    @pytest.mark.unit
    def test_emits_compiled_workflow_with_snapshots_version_and_graph(self):
        wf = WorkflowCompiler(name="pipeline")

        @wf.task
        async def fetch(ctx):
            return {"a": 1}

        @wf.task(depends_on=["fetch"])
        async def train(ctx):
            return {"b": 2}

        compiled = wf.compile()
        assert isinstance(compiled, CompiledWorkflow)
        # exactly one TaskSnapshot per registered task
        assert set(compiled.snapshots) == {"fetch", "train"}
        assert all(s.code_hash for s in compiled.snapshots.values())
        # a populated WorkflowVersion
        assert isinstance(compiled.version, WorkflowVersion)
        assert {t.name for t in compiled.version.topology} == {"fetch", "train"}
        # the version reuses the per-task snapshot code-hash (single hasher)
        for entry in compiled.version.topology:
            assert entry.code_hash == compiled.snapshots[entry.name].code_hash
        # a non-None executable graph — the engine's structural ExecutionPlan
        # (one node per task; values-on-edges execution, no pg lowering).
        from molexp.workflow._engine.plan import ExecutionPlan

        assert isinstance(compiled.graph, ExecutionPlan)
        assert set(compiled.graph.task_names) == {"fetch", "train"}
        # no binding without an experiment
        assert compiled.binding is None

    @pytest.mark.unit
    def test_binds_to_experiment_when_given(self):
        wf = WorkflowCompiler(name="b")

        @wf.task
        async def t(ctx):
            return 1

        from molexp.workflow import WorkflowBindingRegistry

        reg = WorkflowBindingRegistry()
        exp = _Exp("exp-001")
        compiled = wf.compile(experiment=exp, registry=reg)
        assert reg.for_experiment(exp) is compiled
        assert compiled.binding is not None
        assert compiled.binding.experiment_id == "exp-001"
        assert compiled.binding.workflow_id == compiled.workflow_id
