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
from datetime import UTC
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


def test_command_spec_built_per_contract(ctx) -> None:
    from molexp.harness import DryRunExecutor
    from molexp.harness.stages import ExecuteTests

    _seed_test_source(ctx.artifact_store)
    _make_generated_dir(ctx)
    recorder = _RecordingExecutor(DryRunExecutor())

    asyncio.run(ExecuteTests(recorder).run(ctx))

    assert len(recorder.specs) == 1
    spec = recorder.specs[0]
    # pytest-asyncio is a declared dev/runtime dep here, so the invocation must
    # switch it to auto mode — the generated async tests carry no pytest config.
    assert spec.cmd == [
        sys.executable,
        "-m",
        "pytest",
        "test_generated_workflow.py",
        "-q",
        "--import-mode=importlib",
        "-o",
        "asyncio_mode=auto",
    ]
    assert spec.cwd == str(ctx.workspace_root / "generated")
    assert spec.timeout_s == 600


def test_dry_run_persists_passed_test_result_with_lineage(ctx) -> None:
    from molexp.harness import DryRunExecutor
    from molexp.harness.schemas import TestResult
    from molexp.harness.stages import ExecuteTests

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


def test_local_executor_failing_module_persists_then_raises(ctx) -> None:
    from molexp.harness import LocalExecutor
    from molexp.harness.errors import StagePersistedFailureError
    from molexp.harness.schemas import TestResult
    from molexp.harness.stages import ExecuteTests

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


# --------------------------------------------- missing-test-runner guard


def test_missing_pytest_names_the_environment_gap_and_runs_nothing(
    ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing test runner is an environment precondition, not a test
    failure: the stage must say so (naming the interpreter and the fix),
    must not invoke the executor, and must not persist a ``test_result``
    blaming the generated tests. Regression: this surfaced in production
    as ``generated tests failed (pytest exit 1)``."""
    import importlib.util as _ilu

    from molexp.harness.errors import StageExecutionError
    from molexp.harness.stages import ExecuteTests

    _seed_test_source(ctx.artifact_store)
    _make_generated_dir(ctx)

    real_find_spec = _ilu.find_spec
    monkeypatch.setattr(
        _ilu,
        "find_spec",
        lambda name, *a, **k: None if name == "pytest" else real_find_spec(name, *a, **k),
    )

    class _NeverExecutor:
        async def execute(self, spec, *, artifact_store):
            raise AssertionError("executor must not run when pytest is absent")

    with pytest.raises(StageExecutionError) as exc_info:
        asyncio.run(ExecuteTests(_NeverExecutor()).run(ctx))

    message = str(exc_info.value)
    assert sys.executable in message
    assert "molexp[agent]" in message
    assert "NOT run" in message
    assert ctx.artifact_store.list_by_kind("test_result") == []


def test_failure_from_missing_import_names_the_missing_modules(ctx) -> None:
    """A collection-time ``ModuleNotFoundError`` is an environment/dependency
    gap of the *experiment*: the failure must name the unimportable modules
    and the interpreter, not just ``pytest exit 2``. Regression: a generated
    LJ-scan's ``import numpy`` surfaced as an anonymous test failure."""
    from molexp.harness import LocalExecutor
    from molexp.harness.errors import StagePersistedFailureError
    from molexp.harness.stages import ExecuteTests

    _seed_test_source(ctx.artifact_store)
    generated = _make_generated_dir(ctx)
    (generated / "test_generated_workflow.py").write_text(
        "import definitely_missing_mod_xyz\n\ndef test_ok():\n    assert True\n"
    )

    with pytest.raises(StagePersistedFailureError) as exc_info:
        asyncio.run(ExecuteTests(LocalExecutor()).run(ctx))

    message = str(exc_info.value)
    assert "definitely_missing_mod_xyz" in message
    assert sys.executable in message


def test_failure_from_unsupported_async_tests_names_pytest_asyncio(ctx) -> None:
    """molexp task bodies are async-first, so generated tests routinely carry
    ``@pytest.mark.asyncio`` — when the interpreter lacks pytest-asyncio,
    every test fails with pytest's generic 'async def functions are not
    natively supported'. The stage must name the missing plugin. Regression:
    22 generated LJ-scan tests failed anonymously in production."""
    from datetime import datetime

    from molexp.harness.errors import StagePersistedFailureError
    from molexp.harness.schemas import CommandResult
    from molexp.harness.stages import ExecuteTests

    _seed_test_source(ctx.artifact_store)
    _make_generated_dir(ctx)
    stdout_ref = ctx.artifact_store.put_text(
        kind="stdout",
        text="FAILED tests/test_x.py::test_a - Failed: async def functions "
        "are not natively supported.\n1 failed in 0.10s\n",
        created_by="stub",
        parent_ids=[],
    )
    stderr_ref = ctx.artifact_store.put_text(
        kind="stderr", text="", created_by="stub", parent_ids=[]
    )

    class _CannedExecutor:
        async def execute(self, spec, *, artifact_store):
            now = datetime.now(UTC)
            return CommandResult(
                exit_code=1,
                started_at=now,
                ended_at=now,
                stdout_artifact=stdout_ref,
                stderr_artifact=stderr_ref,
            )

    with pytest.raises(StagePersistedFailureError) as exc_info:
        asyncio.run(ExecuteTests(_CannedExecutor()).run(ctx))

    assert "pytest-asyncio" in str(exc_info.value)


def test_red_run_persists_feedback_for_the_test_code_writer(ctx) -> None:
    """A red pytest run must leave a ``test_code_feedback`` artifact carrying
    the captured output, so the NEXT ``generate_test_code`` regeneration (the
    repair loop and the post-eviction re-run both thread
    ``feedback_inputs(ctx, "test_code_feedback")``) learns from the failure
    instead of re-rolling blind. Production: three consecutive re-runs each
    invented a different wrong assertion."""
    from molexp.harness import LocalExecutor
    from molexp.harness.errors import StagePersistedFailureError
    from molexp.harness.stages import ExecuteTests

    _seed_test_source(ctx.artifact_store)
    generated = _make_generated_dir(ctx)
    (generated / "test_generated_workflow.py").write_text(FAILING_TEST_MODULE)

    with pytest.raises(StagePersistedFailureError):
        asyncio.run(ExecuteTests(LocalExecutor()).run(ctx))

    feedback = ctx.artifact_store.latest_by_kind("test_code_feedback")
    assert feedback is not None
    raw = ctx.artifact_store.get(feedback.id)
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    assert "test_no" in text  # the failing test's name reached the feedback
