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

from molexp.workflow import CompiledWorkflow, WorkflowCompiler, default_codec
from molexp.workflow.registry import default_registry
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
    from molexp.workflow._pydantic_graph.plan import ExecutionPlan

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


@pytest.mark.unit
def test_old_authoring_classes_are_gone_from_public_api():
    import molexp.workflow as wf

    assert "WorkflowBuilder" not in wf.__all__
    assert "Workflow" not in wf.__all__
    with pytest.raises(ImportError):
        from molexp.workflow import WorkflowBuilder  # noqa: F401
    with pytest.raises(ImportError):
        from molexp.workflow import Workflow  # noqa: F401


@pytest.mark.unit
def test_compiler_and_compiled_are_the_public_surface():
    import molexp.workflow as wf

    assert "WorkflowCompiler" in wf.__all__
    assert "CompiledWorkflow" in wf.__all__
    # WorkflowGraphCompiler is internal — not part of the public surface.
    assert "WorkflowGraphCompiler" not in wf.__all__
    assert not hasattr(wf, "WorkflowGraphCompiler")


# ── ac-004: codec folded onto CompiledWorkflow; IR round-trip ────────────────


def _register_slugs():
    class _Noop:
        async def execute(self, ctx):
            return None

    for slug in ("c_inspect", "c_train"):
        if slug not in default_registry._factories:  # type: ignore[attr-defined]
            default_registry.register(slug, lambda cfg: _Noop())  # noqa: ARG005


@pytest.mark.unit
def test_compiled_to_ir_from_ir_round_trip_matches_codec():
    _register_slugs()
    ir = {
        "name": "demo",
        "task_configs": [
            {"task_id": "inspect", "task_type": "c_inspect", "config": {}},
            {"task_id": "train", "task_type": "c_train", "config": {}},
        ],
        "links": [{"source": "inspect", "target": "train"}],
        "metadata": {},
    }
    compiled = CompiledWorkflow.from_ir(ir)
    assert isinstance(compiled, CompiledWorkflow)
    produced = compiled.to_ir()
    # byte-identical to the 01 codec output
    assert produced == default_codec.spec_to_ir(compiled)
    # round-trips to an equal artifact (same IR out the far side)
    rebuilt = CompiledWorkflow.from_ir(produced)
    assert rebuilt.to_ir() == produced
