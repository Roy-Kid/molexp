"""Tests for the ``MaterializeExecution`` stage (spec ``harness-run-mode-01-substrate``, T05).

Contract under test (RED — the stage does not exist yet):
- ``name == "materialize_execution"``; reads the latest ``workflow_source``
  + ``test_source`` + ``workflow_ir`` artifacts.
- Writes EXACTLY three files under ``<ctx.workspace_root>/generated/``:
  ``{WorkflowSource.module_name}.py`` (source verbatim),
  ``{TestSource.module_name}.py`` (source verbatim), and ``run_workflow.py``
  (driver rendered from a pure string template).
- Driver text: ``ast.parse`` succeeds; imports ``WorkflowRuntime`` from
  ``molexp``; imports ``build_workflow`` from the workflow module; embeds
  ``WorkflowIR.inputs`` values as ``json.dumps({...}, sort_keys=True)``;
  mentions ``outputs.json``; decides its exit code from the ``"completed"``
  workflow status.
- Registers three ``input_file`` artifacts with exact lineage parents and
  returns the DRIVER's ``ArtifactRef``.
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
    from molexp.harness import WorkflowSource

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
    # Plain dict on purpose: the TestSource schema is itself new in this
    # spec leg; the stage reads the persisted JSON payload.
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
    from molexp.harness import ParameterValue, WorkflowIR

    ir = WorkflowIR(
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


def _run(ctx):
    from molexp.harness import MaterializeExecution

    return asyncio.run(MaterializeExecution().run(ctx))


def _driver_text(ctx) -> str:
    return (ctx.workspace_root / "generated" / "run_workflow.py").read_text()


def _find_by_parents(refs, parent_ids: list[str]):
    matches = [r for r in refs if r.parent_ids == parent_ids]
    assert len(matches) == 1, f"expected exactly one input_file with parents {parent_ids}"
    return matches[0]


# ------------------------------------------------------------------ basics


def test_name_and_subclass() -> None:
    from molexp.harness import MaterializeExecution, Stage

    assert MaterializeExecution.name == "materialize_execution"
    assert issubclass(MaterializeExecution, Stage)


def test_writes_exactly_three_files(ctx) -> None:
    _seed_all(ctx.artifact_store)
    _run(ctx)

    generated = ctx.workspace_root / "generated"
    names = sorted(p.name for p in generated.iterdir())
    assert names == ["generated_workflow.py", "run_workflow.py", "test_generated_workflow.py"]


def test_module_files_match_sources_verbatim(ctx) -> None:
    _seed_all(ctx.artifact_store)
    _run(ctx)

    generated = ctx.workspace_root / "generated"
    assert (generated / "generated_workflow.py").read_text() == WORKFLOW_SOURCE_TEXT
    assert (generated / "test_generated_workflow.py").read_text() == TEST_SOURCE_TEXT


# ------------------------------------------------------------------ driver


def test_driver_parses_and_names_runtime_surface(ctx) -> None:
    _seed_all(ctx.artifact_store)
    _run(ctx)

    driver = _driver_text(ctx)
    ast.parse(driver)
    assert "from molexp import WorkflowRuntime" in driver
    assert "from generated_workflow import build_workflow" in driver


def test_driver_embeds_sorted_params_json(ctx) -> None:
    _seed_all(ctx.artifact_store)
    _run(ctx)

    expected_params = json.dumps({"n_steps": 500}, sort_keys=True)
    assert expected_params in _driver_text(ctx)


def test_driver_mentions_outputs_file_and_completed_status(ctx) -> None:
    _seed_all(ctx.artifact_store)
    _run(ctx)

    driver = _driver_text(ctx)
    assert "outputs.json" in driver
    assert "completed" in driver


# ----------------------------------------------------------------- lineage


def test_registers_three_input_file_artifacts_with_lineage(ctx) -> None:
    ws_ref, ts_ref, ir_ref = _seed_all(ctx.artifact_store)
    _run(ctx)

    refs = ctx.artifact_store.list_by_kind("input_file")
    assert len(refs) == 3
    _find_by_parents(refs, [ws_ref.id])  # workflow module
    _find_by_parents(refs, [ts_ref.id])  # test module
    _find_by_parents(refs, [ws_ref.id, ir_ref.id, ts_ref.id])  # driver


def test_returns_the_driver_artifact_ref(ctx) -> None:
    ws_ref, ts_ref, ir_ref = _seed_all(ctx.artifact_store)
    ref = _run(ctx)

    refs = ctx.artifact_store.list_by_kind("input_file")
    driver = _find_by_parents(refs, [ws_ref.id, ir_ref.id, ts_ref.id])
    assert ref.kind == "input_file"
    assert ref.id == driver.id
