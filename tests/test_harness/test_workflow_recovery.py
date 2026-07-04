"""``compiled_workflow_for_run`` — deterministic workflow reconstruction.

vision-loop-07 §2: the ONE shared recovery path for executing a persisted run
by id — used by the CLI (``molexp run`` on a plan-generated run) and the
harness lifecycle capabilities. Shape dispatch, not a fallback chain:

1. a plan-generated run carrying a ``workflow_source`` artifact → the
   generated ``build_workflow()`` program is compiled;
2. a script-run carrying ``metadata.workflow_snapshot.entrypoint`` →
   ``molexp.entry.load_workflow_from_entrypoint``;
3. neither shape → ``WorkflowRecoveryError`` naming BOTH shapes and what the
   run actually carries — never a silent ``None``.

The plan-shape fixture mirrors the canned ``build_workflow()`` program used by
``tests/test_cli/test_plan_cmd.py`` (a source that genuinely compiles through
the public ``molexp.workflow`` API).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.workflow_recovery import (
    WorkflowRecoveryError,
    compiled_workflow_for_run,
)
from molexp.workflow import CompiledWorkflow
from molexp.workspace import Workspace
from molexp.workspace.run import Run

# A generated program that compiles to a real workflow through the public API
# (same canned shape as tests/test_cli/test_plan_cmd.py).
_VALID_SOURCE = """\
from molexp.workflow import TaskContext, WorkflowCompiler


def build_workflow() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="plan-demo")

    @wf.task
    async def build_system(ctx: TaskContext) -> dict:
        return {"structure": "system.pdb"}

    @wf.task(depends_on=["build_system"])
    async def simulate(ctx: TaskContext) -> dict:
        return {"trajectory": "traj.dcd"}

    return wf
"""

# A malformed generated program: valid Python, but no build_workflow().
_SOURCE_WITHOUT_BUILDER = "ANSWER = 42\n"

# A script-run workflow module. ``load_workflow_from_entrypoint`` requires the
# qualname to resolve to a compiled ``molexp.workflow`` Workflow instance (the
# shape ``Workflow.bind_to(experiment)`` records), so the module compiles at
# module level.
_SCRIPT_MODULE = """\
from molexp.workflow import TaskContext, WorkflowCompiler

wf = WorkflowCompiler(name="script-demo")


@wf.task
async def only_task(ctx: TaskContext) -> dict:
    return {"ok": True}


workflow = wf.compile()
"""

# ── fixtures ─────────────────────────────────────────────────────────────────


def _bare_run(tmp_path: Path, **run_kwargs: object) -> Run:
    ws = Workspace(root=tmp_path, name="recover-lab")
    exp = ws.add_project("p").add_experiment("e", workflow_source="s.py", params={})
    run = exp.add_run(params={"seed": 1}, **run_kwargs)  # type: ignore[arg-type]
    run.materialize()
    return run


def _persist_workflow_source(run: Run, source: str) -> None:
    """Persist a ``workflow_source`` artifact under the run (plan-run shape)."""
    store = FileArtifactStore(root=Path(str(run.run_dir)) / "artifacts")
    store.put_json(
        kind="workflow_source",
        obj={
            "source": source,
            "module_name": "plan_demo",
            "bound_workflow_id": "bw-demo",
            "symbols": ["WorkflowCompiler", "TaskContext"],
        },
        created_by="test_workflow_recovery",
        parent_ids=[],
    )


def _script_entrypoint(tmp_path: Path) -> str:
    wf_file = tmp_path / "user_workflow.py"
    wf_file.write_text(textwrap.dedent(_SCRIPT_MODULE), encoding="utf-8")
    return f"{wf_file}:workflow"


# ── shape (a): plan-generated run with a workflow_source artifact ────────────


class TestPlanRunShape:
    def test_compiled_workflow_carries_the_generated_graph(self, tmp_path: Path) -> None:
        run = _bare_run(tmp_path)
        _persist_workflow_source(run, _VALID_SOURCE)
        compiled = compiled_workflow_for_run(run)
        assert isinstance(compiled, CompiledWorkflow)
        assert compiled.name == "plan-demo"
        assert set(compiled.registration_by_name) == {"build_system", "simulate"}

    def test_source_without_build_workflow_raises_loudly(self, tmp_path: Path) -> None:
        run = _bare_run(tmp_path)
        _persist_workflow_source(run, _SOURCE_WITHOUT_BUILDER)
        with pytest.raises(WorkflowRecoveryError, match="build_workflow"):
            compiled_workflow_for_run(run)


# ── shape (b): script-run with workflow_snapshot.entrypoint ──────────────────


class TestScriptRunShape:
    def test_entrypoint_loads_and_compiles(self, tmp_path: Path) -> None:
        entrypoint = _script_entrypoint(tmp_path)
        run = _bare_run(tmp_path / "ws", workflow_snapshot={"entrypoint": entrypoint})
        compiled = compiled_workflow_for_run(run)
        assert isinstance(compiled, CompiledWorkflow)
        assert compiled.name == "script-demo"
        assert set(compiled.registration_by_name) == {"only_task"}

    def test_dead_entrypoint_raises_recovery_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "gone.py"
        run = _bare_run(tmp_path / "ws", workflow_snapshot={"entrypoint": f"{missing}:wf"})
        with pytest.raises(WorkflowRecoveryError):
            compiled_workflow_for_run(run)


# ── deterministic dispatch order ─────────────────────────────────────────────


class TestShapeDispatch:
    def test_workflow_source_wins_over_entrypoint(self, tmp_path: Path) -> None:
        """Shape dispatch is deterministic: the plan artifact is shape (a)."""
        entrypoint = _script_entrypoint(tmp_path)
        run = _bare_run(tmp_path / "ws", workflow_snapshot={"entrypoint": entrypoint})
        _persist_workflow_source(run, _VALID_SOURCE)
        compiled = compiled_workflow_for_run(run)
        assert compiled.name == "plan-demo"


# ── shape (c): neither — loud, named error ───────────────────────────────────


class TestNeitherShape:
    def test_error_names_both_persisted_shapes_and_the_run(self, tmp_path: Path) -> None:
        run = _bare_run(tmp_path)
        with pytest.raises(WorkflowRecoveryError) as excinfo:
            compiled_workflow_for_run(run)
        message = str(excinfo.value)
        assert "workflow_source" in message
        assert "workflow_snapshot" in message
        assert run.id in message
