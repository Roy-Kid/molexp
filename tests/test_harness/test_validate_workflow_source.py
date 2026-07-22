"""Tests for the generated-workflow-source gate (plan-mode-revival-03).

Two units under test, one TestClass each:

* ``WorkflowSourceValidator`` (``molexp.harness.validators.workflow_source``) —
  the pure, never-raising AST pre-check: flags syntax errors (ac-004), rejects
  private-subpackage imports and passes a public-surface-only program (ac-005).
* ``ValidateWorkflowSource`` (``molexp.harness.stages.validate_workflow_source``)
  — the stage that runs the pre-check, then lazily compiles a passing program
  to a real ``CompiledWorkflow`` (ac-007), always persists a
  ``PlanValidationReport`` and raises ``StagePersistedFailureError`` on failure
  (ac-008), and never reaches ``exec`` for ast-rejected source while running
  valid source under a restricted ``__builtins__`` (ac-009).

``VALID_SOURCE`` was verified against the real public ``molexp.workflow``
surface (``WorkflowCompiler`` + decorator ``@wf.task`` + ``.compile()``) — it
compiles to a ``Workflow``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

# --------------------------------------------------------------- fixtures

# ac-007 happy-path source: defines a builder via the decorator surface and
# returns it; ``.compile()`` accepts it. Verified against real molexp.workflow.
VALID_SOURCE = """\
from molexp.workflow import Task, TaskContext, WorkflowCompiler


def build_workflow() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="demo")

    @wf.task
    async def load(ctx: TaskContext) -> list[int]:
        return [1, 2, 3]

    @wf.task(depends_on=["load"])
    async def square(ctx: TaskContext) -> list[int]:
        return [x * x for x in ctx.inputs]

    return wf
"""

# (a) syntax error.
SYNTAX_ERROR_SOURCE = "def (:\n    pass\n"

# (b) imports a private subpackage of molexp.workflow.
PRIVATE_IMPORT_SOURCE = """\
from molexp.workflow._engine import something
from molexp.workflow import WorkflowCompiler


def build_workflow() -> WorkflowCompiler:
    return WorkflowCompiler(name="sneaky")
"""

# (c) parses + imports cleanly but ``.compile()`` fails — a task depends on a
# task that is never registered → UnknownTaskError at build time.
BUILD_FAILS_SOURCE = """\
from molexp.workflow import TaskContext, WorkflowCompiler


def build_workflow() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="baddep")

    @wf.task(depends_on=["does_not_exist"])
    async def square(ctx: TaskContext) -> list[int]:
        return [1]

    return wf
"""


def _workflow_source_dict(source: str = VALID_SOURCE) -> dict:
    return {
        "source": source,
        "module_name": "generated_workflow",
        "bound_workflow_id": "bw-x",
        "symbols": ["WorkflowCompiler", "Task", "TaskContext"],
    }


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
        run_id="run-vws",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
    )


def _seed_workflow_source(ctx, source: str = VALID_SOURCE):
    return ctx.artifact_store.put_json(
        kind="workflow_source",
        obj=_workflow_source_dict(source),
        created_by="seed",
        parent_ids=[],
    )


class TestWorkflowSourceValidator:
    """Pure AST pre-check — flags syntax + private imports, never raises."""

    def test_flags_syntax_error_without_raising(self) -> None:
        from molexp.harness.validators.workflow_source import WorkflowSourceValidator

        report = WorkflowSourceValidator.validate(SYNTAX_ERROR_SOURCE)
        assert report.passed is False
        assert report.target_kind == "workflow_source"
        assert any("syntax" in v.code.lower() for v in report.violations)

    def test_never_raises_on_arbitrary_garbage(self) -> None:
        from molexp.harness.validators.workflow_source import WorkflowSourceValidator

        # Total function: even on wildly malformed input no exception escapes.
        for bad in ("def (:\n", "@@@@", "import", "class :", "\x00\x01"):
            report = WorkflowSourceValidator.validate(bad)
            assert report.passed is False

    def test_rejects_private_subpackage_import(self) -> None:
        from molexp.harness.validators.workflow_source import WorkflowSourceValidator

        report = WorkflowSourceValidator.validate(PRIVATE_IMPORT_SOURCE)
        assert report.passed is False
        # A violation must name the disallowed private import target.
        assert any("_engine" in (v.message + (v.path or "")) for v in report.violations)

    def test_passes_public_surface_only(self) -> None:
        from molexp.harness.validators.workflow_source import WorkflowSourceValidator

        report = WorkflowSourceValidator.validate(VALID_SOURCE)
        assert report.passed is True
        assert report.violations == []


class TestValidateWorkflowSource:
    """Stage — compile-gate a WorkflowSource artifact, always persisting a report."""

    def test_compiles_valid_source_to_workflow(self, ctx) -> None:
        from molexp.harness.schemas.validation import PlanValidationReport
        from molexp.harness.stages.validate_workflow_source import ValidateWorkflowSource

        ws_ref = _seed_workflow_source(ctx, VALID_SOURCE)
        report_ref = asyncio.run(ValidateWorkflowSource().run(ctx))

        assert report_ref.kind == "validation_report"
        assert ws_ref.id in report_ref.parent_ids

        raw = ctx.artifact_store.get(report_ref.id)
        report = PlanValidationReport.model_validate(json.loads(raw))
        assert report.passed is True
        assert report.target_kind == "workflow_source"

    @pytest.mark.parametrize(
        "source",
        [SYNTAX_ERROR_SOURCE, BUILD_FAILS_SOURCE],
        ids=["pre_check_fail", "build_fail"],
    )
    def test_persists_failing_report_then_raises_on_invalid(self, ctx, source: str) -> None:
        """Both distinct stage paths — pre-check reject and compile/build failure —
        persist a failing report (always-persist contract) and then raise
        ``StagePersistedFailureError`` whose ``persisted_ref`` points at it."""
        from molexp.harness.errors import StageExecutionError, StagePersistedFailureError
        from molexp.harness.schemas.validation import PlanValidationReport
        from molexp.harness.stages.validate_workflow_source import ValidateWorkflowSource

        _seed_workflow_source(ctx, source)

        with pytest.raises(StageExecutionError) as exc_info:
            asyncio.run(ValidateWorkflowSource().run(ctx))
        assert isinstance(exc_info.value, StagePersistedFailureError)

        reports = ctx.artifact_store.list_by_kind("validation_report")
        assert len(reports) == 1
        raw = ctx.artifact_store.get(reports[0].id)
        report = PlanValidationReport.model_validate(json.loads(raw))
        assert report.passed is False
        assert report.target_kind == "workflow_source"
        assert exc_info.value.persisted_ref.id == reports[0].id

    def test_ast_rejected_source_never_reaches_exec(self, ctx, monkeypatch) -> None:
        """ac-009: syntax-error and private-import fixtures are rejected at the
        ast/compile pre-check — ``exec`` is never reached for them."""
        import builtins

        from molexp.harness.errors import StagePersistedFailureError
        from molexp.harness.stages.validate_workflow_source import ValidateWorkflowSource

        exec_calls: list[object] = []
        real_exec = builtins.exec

        def _tracking_exec(*args, **kwargs):  # noqa: ANN002, ANN003
            exec_calls.append(args)
            return real_exec(*args, **kwargs)

        monkeypatch.setattr(builtins, "exec", _tracking_exec)

        for source in (SYNTAX_ERROR_SOURCE, PRIVATE_IMPORT_SOURCE):
            exec_calls.clear()
            _seed_workflow_source(ctx, source)
            with pytest.raises(StagePersistedFailureError):
                asyncio.run(ValidateWorkflowSource().run(ctx))
            assert exec_calls == [], f"exec must not run for ast-rejected source ({source[:20]!r})"

    def test_valid_source_exec_uses_restricted_builtins(self, ctx, monkeypatch) -> None:
        """ac-009: the valid fixture IS executed, but the exec namespace's
        ``__builtins__`` is restricted (not the full real builtins module)."""
        import builtins

        from molexp.harness.stages.validate_workflow_source import ValidateWorkflowSource

        captured_globals: list[dict] = []
        real_exec = builtins.exec

        def _capturing_exec(code, globals_ns=None, locals_ns=None):
            captured_globals.append(globals_ns if globals_ns is not None else {})
            return real_exec(code, globals_ns, locals_ns)

        monkeypatch.setattr(builtins, "exec", _capturing_exec)

        _seed_workflow_source(ctx, VALID_SOURCE)
        asyncio.run(ValidateWorkflowSource().run(ctx))

        assert captured_globals, "exec was expected to run for the valid fixture"
        ns = captured_globals[0]
        assert "__builtins__" in ns, "exec namespace must define __builtins__"
        assert ns["__builtins__"] is not builtins
        assert ns["__builtins__"] is not builtins.__dict__
