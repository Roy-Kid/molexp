"""Tests for the ``MaterializeExecution`` stage.

Contract: reads the latest ``workflow_source`` + ``test_source`` +
``workflow_ir`` artifacts and writes the workflow module, the pytest module,
and a ``run_workflow.py`` driver under ``<ctx.workspace_root>/generated/``;
each written file is registered as an ``input_file`` artifact with lineage
back to its source, and the stage returns the DRIVER's ``PlanArtifactRef``.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import pytest

WORKFLOW_SOURCE_TEXT = "def build_workflow():\n    return None\n"
TEST_SOURCE_TEXT = "def test_ok():\n    assert True\n"


@pytest.fixture()
def ctx(tmp_path: Path):
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteArtifactLineageStore(path=db, artifact_store=a)
    return HarnessRunContext(
        run_id="run-mat",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
    )


def _seed_workflow_source(store):
    from molexp.harness.schemas import WorkflowSource

    ws = WorkflowSource(
        source=WORKFLOW_SOURCE_TEXT,
        module_name="generated_workflow",
        bound_workflow_id="bw-1",
        symbols=(),
    )
    return store.put_json(
        kind="workflow_source",
        obj=json.loads(ws.model_dump_json()),
        created_by="seed",
        parent_ids=[],
    )


def _seed_test_source(store):
    # Plain dict on purpose: the stage reads the persisted JSON payload.
    return store.put_json(
        kind="test_source",
        obj={
            "source": TEST_SOURCE_TEXT,
            "module_name": "test_generated_workflow",
            "test_spec_id": "ts-1",
            "bound_workflow_id": "bw-1",
            "symbols": [],
        },
        created_by="seed",
        parent_ids=[],
    )


def _seed_workflow_ir(store):
    from molexp.harness.schemas import ParameterValue, PlanWorkflowIR

    ir = PlanWorkflowIR(
        id="wf-ir-1",
        name="demo",
        objective="materialize a runnable workflow",
        inputs={"n_steps": ParameterValue(value=500, source="user_provided")},
        tasks=[],
        edges=[],
        expected_outputs=[],
    )
    return store.put_json(
        kind="workflow_ir",
        obj=json.loads(ir.model_dump_json()),
        created_by="seed",
        parent_ids=[],
    )


def _seed_all(store):
    return _seed_workflow_source(store), _seed_test_source(store), _seed_workflow_ir(store)


def _seed_input_set(store, *, fixed_params=None):
    obj = {
        "id": "is-1",
        "experiment_spec_id": "spec-1",
        "title": "sweep",
        "sweep_axes": [{"name": "n_steps", "values": [1000, 2000], "source": "user_provided"}],
        "strategy": "grid",
        "total_runs": 2,
    }
    if fixed_params is not None:
        obj["fixed_params"] = fixed_params
    return store.put_json(kind="input_set", obj=obj, created_by="seed", parent_ids=[])


def _run(ctx):
    from molexp.harness.stages import MaterializeExecution

    return asyncio.run(MaterializeExecution().run(ctx))


def _driver_text(ctx) -> str:
    return (ctx.workspace_root / "generated" / "run_workflow.py").read_text()


def _find_by_parents(refs, parent_ids: list[str]):
    matches = [r for r in refs if r.parent_ids == parent_ids]
    assert len(matches) == 1, f"expected exactly one input_file with parents {parent_ids}"
    return matches[0]


class TestMaterializeExecution:
    """Materializes the workflow module, test module, and driver under generated/."""

    def test_writes_exactly_three_files_with_sources_verbatim(self, ctx) -> None:
        """Single-file source → exactly three files; each generated .py keeps its
        body verbatim behind a ``from __future__ import annotations`` header (so
        in-function type annotations don't NameError at pytest collection)."""
        _seed_all(ctx.artifact_store)
        _run(ctx)

        generated = ctx.workspace_root / "generated"
        names = sorted(p.name for p in generated.iterdir())
        assert names == ["generated_workflow.py", "run_workflow.py", "test_generated_workflow.py"]
        header = "from __future__ import annotations\n"
        assert (generated / "generated_workflow.py").read_text() == header + WORKFLOW_SOURCE_TEXT
        assert (generated / "test_generated_workflow.py").read_text() == header + TEST_SOURCE_TEXT

    def test_multifile_writes_package_with_injected_init_tests_and_driver(self, ctx) -> None:
        """Multi-file source: per-task modules land under ``workflow/``; the
        assembly (``source``) is injected as ``workflow/__init__.py`` when
        ``files`` omit it; the test lands under ``tests/``; and the driver
        imports the package by ``module_name``. Regression: a silent __init__
        drop left generated/workflow/ without ``build_workflow``, breaking
        CompileWorkflow with AttributeError."""
        from molexp.harness.schemas import WorkflowSource

        assembly = (
            "from molexp.workflow import WorkflowCompiler\n"
            "from workflow.task_a import task_a\n\n"
            "def build_workflow() -> WorkflowCompiler:\n"
            '    wf = WorkflowCompiler(name="demo")\n'
            "    wf.task(task_a)\n"
            "    return wf\n"
        )
        ws = WorkflowSource(
            source=assembly,
            module_name="workflow",
            bound_workflow_id="bw-1",
            symbols=(),
            files=[
                {
                    "path": "workflow/task_a.py",
                    "source": "async def task_a() -> dict:\n    return {'ok': True}\n",
                }
            ],
        )
        ctx.artifact_store.put_json(
            kind="workflow_source",
            obj=json.loads(ws.model_dump_json()),
            created_by="seed",
            parent_ids=[],
        )
        test_body = "from workflow import build_workflow\n\n\ndef test_a():\n    assert build_workflow().compile() is not None\n"
        ctx.artifact_store.put_json(
            kind="test_source",
            obj={
                "source": test_body,
                "module_name": "test_a",
                "test_spec_id": "ts-1",
                "bound_workflow_id": "bw-1",
                "symbols": [],
                "files": [{"path": "tests/test_a.py", "source": test_body}],
            },
            created_by="seed",
            parent_ids=[],
        )
        _seed_workflow_ir(ctx.artifact_store)
        _run(ctx)

        g = ctx.workspace_root / "generated"
        init = g / "workflow" / "__init__.py"
        assert init.is_file(), "assembly must land as workflow/__init__.py"
        assert "def build_workflow" in init.read_text()
        assert (g / "workflow" / "task_a.py").is_file()
        assert (g / "tests" / "test_a.py").is_file()
        assert "from workflow import build_workflow" in _driver_text(ctx)
        # One input_file per distinct written file: workflow/__init__.py +
        # task_a.py + the test (whose injected __init__ dedups by content) + driver.
        assert len(ctx.artifact_store.list_by_kind("input_file")) == 4

    def test_driver_rendered_per_contract(self, ctx) -> None:
        """Driver parses; names the runtime surface; embeds sorted params JSON;
        mentions outputs.json + the canonical terminal-success status; branches
        on --compile-only; mounts scratch_root so ``ctx.workdir`` is available
        to task bodies (regression: bare execution crashed on NoneType/str)."""
        _seed_all(ctx.artifact_store)
        _run(ctx)

        driver = _driver_text(ctx)
        ast.parse(driver)
        assert "from molexp import WorkflowRuntime" in driver
        assert "from generated_workflow import build_workflow" in driver
        assert json.dumps({"n_steps": 500}, sort_keys=True) in driver
        assert "outputs.json" in driver
        assert '== "succeeded"' in driver
        assert "--compile-only" in driver
        assert 'scratch_root=".materialize"' in driver

    def test_registers_input_file_artifacts_with_lineage_and_returns_driver(self, ctx) -> None:
        """Three ``input_file`` artifacts with exact lineage parents; the stage
        returns the driver's ref (single-ref contract)."""
        ws_ref, ts_ref, ir_ref = _seed_all(ctx.artifact_store)
        ref = _run(ctx)

        refs = ctx.artifact_store.list_by_kind("input_file")
        assert len(refs) == 3
        _find_by_parents(refs, [ws_ref.id])  # workflow module
        _find_by_parents(refs, [ts_ref.id])  # test module
        driver = _find_by_parents(refs, [ws_ref.id, ir_ref.id, ts_ref.id])  # driver
        assert ref.kind == "input_file"
        assert ref.id == driver.id

    def test_input_set_overlays_fixed_params_and_sweep_first_cell(self, ctx) -> None:
        """``fixed_params`` land in the driver PARAMS verbatim (whole values —
        the channel for list-valued root inputs), and the sweep's first cell
        overlays the IR default (n_steps 500 → 1000). Regression: a real
        ``--execute`` run swept ``sigma_values`` per-scalar and crashed
        iterating a float."""
        _seed_all(ctx.artifact_store)
        _seed_input_set(ctx.artifact_store, fixed_params={"sigma_values": [0.9, 1.0]})
        _run(ctx)
        assert json.dumps(
            {"n_steps": 1000, "sigma_values": [0.9, 1.0]}, sort_keys=True
        ) in _driver_text(ctx)

    def test_rejects_generated_file_path_escaping_sandbox(self, ctx) -> None:
        """A generated file whose path escapes ``generated/`` is refused (source
        is agent-generated), and nothing is written outside the sandbox."""
        from molexp.harness.errors import StageExecutionError

        store = ctx.artifact_store
        store.put_json(
            kind="workflow_source",
            obj={
                "source": "x",
                "module_name": "workflow",
                "bound_workflow_id": "bw-1",
                "symbols": [],
                "files": [{"path": "../escape.py", "source": "x = 1\n"}],
            },
            created_by="seed",
            parent_ids=[],
        )
        _seed_test_source(store)
        _seed_workflow_ir(store)
        with pytest.raises(StageExecutionError, match="escapes"):
            _run(ctx)
        assert not (ctx.workspace_root.parent / "escape.py").exists()
