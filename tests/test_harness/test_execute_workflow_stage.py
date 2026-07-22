"""Tests for the ``ExecuteWorkflow`` stage.

Mirrors ``molexp.harness.stages.execute_workflow``. The stage runs
``python run_workflow.py`` in the run's ``generated/`` directory through an
injected ``Executor`` (with ``expected_outputs=["outputs.json"]``), lifts the
``CommandResult`` into an ``ExecutionResult`` artifact, and — on nonzero exit —
persists the failed result first, then raises ``StagePersistedFailureError``.

The drivers below are deliberately plain Python (no molexp import) so the
subprocess stays fast and hermetic.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

DRIVER_WRITES_OUTPUTS = (
    "import json\nwith open('outputs.json','w') as fh:\n    json.dump({'estimate_d': 0.25}, fh)\n"
)
DRIVER_EXITS_NONZERO = "import sys\nsys.exit(3)\n"
DRIVER_NO_OUTPUTS_FILE = "print('no file')\n"


class _RecordingExecutor:
    """Wrap a real executor, capturing every ``CommandSpec`` passed through."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.specs: list = []

    async def execute(self, spec, *, artifact_store):
        self.specs.append(spec)
        return await self.inner.execute(spec, artifact_store=artifact_store)


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
        run_id="run-xwf",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
    )


def _seed_workflow_source(store):
    from molexp.harness.schemas import WorkflowSource

    ws = WorkflowSource(
        source="def build_workflow():\n    return None\n",
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


def _write_driver(ctx, driver_text: str) -> Path:
    generated = ctx.workspace_root / "generated"
    generated.mkdir()
    (generated / "run_workflow.py").write_text(driver_text)
    return generated


class TestExecuteWorkflow:
    def test_command_spec_built_per_contract(self, ctx) -> None:
        from molexp.harness import LocalExecutor
        from molexp.harness.stages import ExecuteWorkflow

        _seed_workflow_source(ctx.artifact_store)
        generated = _write_driver(ctx, DRIVER_WRITES_OUTPUTS)
        recorder = _RecordingExecutor(LocalExecutor())

        asyncio.run(ExecuteWorkflow(recorder).run(ctx))

        assert len(recorder.specs) == 1
        spec = recorder.specs[0]
        assert spec.cmd == [sys.executable, "run_workflow.py"]
        assert spec.cwd == str(generated)
        assert spec.timeout_s == 3600
        assert spec.expected_outputs == ["outputs.json"]

    def test_green_run_persists_succeeded_result_with_parsed_outputs(self, ctx) -> None:
        from molexp.harness import LocalExecutor
        from molexp.harness.schemas import ExecutionResult
        from molexp.harness.stages import ExecuteWorkflow

        ws_ref = _seed_workflow_source(ctx.artifact_store)
        _write_driver(ctx, DRIVER_WRITES_OUTPUTS)

        ref = asyncio.run(ExecuteWorkflow(LocalExecutor()).run(ctx))

        assert ref.kind == "execution_result"
        assert ref.parent_ids == [ws_ref.id]
        result = ExecutionResult.model_validate(json.loads(ctx.artifact_store.get(ref.id)))
        assert result.status == "succeeded"
        assert result.exit_code == 0
        assert result.outputs == {"estimate_d": 0.25}
        assert result.bound_workflow_id == "bw-1"
        assert len(result.output_artifacts) >= 1
        assert any(r.uri.endswith("outputs.json") for r in result.output_artifacts)

    def test_nonzero_exit_persists_failed_result_then_raises(self, ctx) -> None:
        from molexp.harness import LocalExecutor
        from molexp.harness.errors import StagePersistedFailureError
        from molexp.harness.schemas import ExecutionResult
        from molexp.harness.stages import ExecuteWorkflow

        _seed_workflow_source(ctx.artifact_store)
        _write_driver(ctx, DRIVER_EXITS_NONZERO)

        with pytest.raises(StagePersistedFailureError) as exc_info:
            asyncio.run(ExecuteWorkflow(LocalExecutor()).run(ctx))

        assert exc_info.value.persisted_ref.kind == "execution_result"
        refs = ctx.artifact_store.list_by_kind("execution_result")
        assert len(refs) == 1
        result = ExecutionResult.model_validate(json.loads(ctx.artifact_store.get(refs[0].id)))
        assert result.status == "failed"
        assert result.exit_code == 3
        assert result.outputs == {}

    def test_missing_outputs_json_yields_empty_outputs_without_crashing(self, ctx) -> None:
        from molexp.harness import LocalExecutor
        from molexp.harness.schemas import ExecutionResult
        from molexp.harness.stages import ExecuteWorkflow

        _seed_workflow_source(ctx.artifact_store)
        _write_driver(ctx, DRIVER_NO_OUTPUTS_FILE)

        ref = asyncio.run(ExecuteWorkflow(LocalExecutor()).run(ctx))

        result = ExecutionResult.model_validate(json.loads(ctx.artifact_store.get(ref.id)))
        assert result.status == "succeeded"
        assert result.outputs == {}
