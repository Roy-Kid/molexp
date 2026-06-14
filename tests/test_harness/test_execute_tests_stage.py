"""Tests for the ``ExecuteTests`` stage (spec ``harness-run-mode-01-substrate``, T05).

Contract under test (RED — the stage does not exist yet):
- ``name == "execute_tests"``; constructor-injected ``Executor`` plus a
  keyword-only ``timeout_s`` defaulting to 600.
- Reads the latest ``test_source`` artifact and builds
  ``CommandSpec(cmd=[sys.executable, "-m", "pytest", "<module>.py", "-q"],
  cwd=<workspace_root>/generated)``.
- Persists a ``test_result`` artifact (``TestResult`` JSON, parent = the
  ``test_source`` artifact id): ``id`` starts with ``"test-result-"``,
  ``test_spec_id`` from the ``TestSource``, ``status`` "passed" iff exit 0,
  stdout/stderr taken from the ``CommandResult`` artifacts.
- On nonzero exit: persist FIRST, then raise ``StagePersistedFailureError``
  whose ``persisted_ref.kind == "test_result"``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

PASSING_TEST_MODULE = "def test_ok():\n    assert True\n"
FAILING_TEST_MODULE = "def test_no():\n    assert False\n"


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
        run_id="run-xtests",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
    )


def _seed_test_source(store):
    return store.put_json(
        kind="test_source",
        obj={
            "source": PASSING_TEST_MODULE,
            "module_name": "test_generated_workflow",
            "test_spec_id": "ts-1",
            "bound_workflow_id": "bw-1",
            "symbols": [],
        },
        created_by="seed",
        parent_ids=[],
    )


def _make_generated_dir(ctx) -> Path:
    generated = ctx.workspace_root / "generated"
    generated.mkdir()
    return generated


# ------------------------------------------------------------------ basics


def test_name_and_subclass() -> None:
    from molexp.harness import ExecuteTests, Stage

    assert ExecuteTests.name == "execute_tests"
    assert issubclass(ExecuteTests, Stage)


def test_command_spec_built_per_contract(ctx) -> None:
    from molexp.harness import DryRunExecutor, ExecuteTests

    _seed_test_source(ctx.artifact_store)
    _make_generated_dir(ctx)
    recorder = _RecordingExecutor(DryRunExecutor())

    asyncio.run(ExecuteTests(recorder).run(ctx))

    assert len(recorder.specs) == 1
    spec = recorder.specs[0]
    assert spec.cmd == [sys.executable, "-m", "pytest", "test_generated_workflow.py", "-q"]
    assert spec.cwd == str(ctx.workspace_root / "generated")
    assert spec.timeout_s == 600


def test_custom_timeout_flows_into_command_spec(ctx) -> None:
    from molexp.harness import DryRunExecutor, ExecuteTests

    _seed_test_source(ctx.artifact_store)
    _make_generated_dir(ctx)
    recorder = _RecordingExecutor(DryRunExecutor())

    asyncio.run(ExecuteTests(recorder, timeout_s=42).run(ctx))

    assert recorder.specs[0].timeout_s == 42


def test_dry_run_persists_passed_test_result_with_lineage(ctx) -> None:
    from molexp.harness import DryRunExecutor, ExecuteTests, TestResult

    ts_ref = _seed_test_source(ctx.artifact_store)
    _make_generated_dir(ctx)

    ref = asyncio.run(ExecuteTests(DryRunExecutor()).run(ctx))

    assert ref.kind == "test_result"
    assert ref.parent_ids == [ts_ref.id]
    result = TestResult.model_validate(json.loads(ctx.artifact_store.get(ref.id)))
    assert result.id.startswith("test-result-")
    assert result.test_spec_id == "ts-1"
    assert result.status == "passed"


# --------------------------------------------- integration (real subprocess)


def test_local_executor_passing_module_yields_passed_result(ctx) -> None:
    from molexp.harness import ExecuteTests, LocalExecutor, TestResult

    _seed_test_source(ctx.artifact_store)
    generated = _make_generated_dir(ctx)
    (generated / "test_generated_workflow.py").write_text(PASSING_TEST_MODULE)

    ref = asyncio.run(ExecuteTests(LocalExecutor()).run(ctx))

    result = TestResult.model_validate(json.loads(ctx.artifact_store.get(ref.id)))
    assert result.status == "passed"
    assert result.stdout is not None
    assert result.stderr is not None


def test_local_executor_failing_module_persists_then_raises(ctx) -> None:
    from molexp.harness import (
        ExecuteTests,
        LocalExecutor,
        StagePersistedFailureError,
        TestResult,
    )

    _seed_test_source(ctx.artifact_store)
    generated = _make_generated_dir(ctx)
    (generated / "test_generated_workflow.py").write_text(FAILING_TEST_MODULE)

    with pytest.raises(StagePersistedFailureError) as exc_info:
        asyncio.run(ExecuteTests(LocalExecutor()).run(ctx))

    assert exc_info.value.persisted_ref.kind == "test_result"
    refs = ctx.artifact_store.list_by_kind("test_result")
    assert len(refs) == 1
    result = TestResult.model_validate(json.loads(ctx.artifact_store.get(refs[0].id)))
    assert result.status == "failed"
