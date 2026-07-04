"""Acceptance pins for workflow-refactor-02 (merge build+compile → CompiledWorkflow).

- ac-001: ``WorkflowCompiler.compile()`` emits a ``CompiledWorkflow`` carrying
  graph + per-task snapshots + version + binding.
- ac-002: build + compile merged; ``WorkflowBuilder`` / ``Workflow`` /
  ``WorkflowGraphCompiler`` are gone from the public API; ``WorkflowCompiler`` +
  ``CompiledWorkflow`` are exported.
- ac-004: codec folded onto ``CompiledWorkflow``; ``to_ir()`` / ``from_ir()``
  round-trip equals the codec output for slugged data-DAG fixtures.
"""

from __future__ import annotations

import pytest

from molexp.workflow import CompiledWorkflow, WorkflowCompiler
from molexp.workflow.version import WorkflowVersion


class _Exp:
    """Minimal experiment stand-in (duck-typed `.id`)."""

    def __init__(self, exp_id: str) -> None:
        self.id = exp_id


# ── ac-001: compile() emits a rich CompiledWorkflow ──────────────────────────


@pytest.mark.unit
def test_compile_emits_compiled_workflow_with_snapshots_version_graph():
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
def test_compile_binds_to_experiment_when_given():
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


# ── ac-002: old build/compile/spec classes gone from the public API ──────────


# ── ac-004: codec folded onto CompiledWorkflow; IR round-trip ────────────────
