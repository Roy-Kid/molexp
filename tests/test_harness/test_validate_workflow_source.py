"""Tests for spec plan-mode-revival-03-workflow-codegen — schema, pure
validator, and the ``ValidateWorkflowSource`` stage.

Covers acceptance criteria:
- ac-001: ``WorkflowSource`` is a frozen pydantic model with source +
  derivation metadata, re-exported from ``molexp.harness``.
- ac-002: ``"workflow_source"`` is a registered artifact kind AND a
  ``ValidationReport`` target kind.
- ac-003: ``SYSTEM_PROMPT`` names the public ``molexp.workflow`` API.
- ac-004: ``validate_workflow_source`` flags syntax errors, never raises.
- ac-005: ``validate_workflow_source`` rejects private-subpackage imports
  and passes a public-surface-only program.
- ac-007: ``ValidateWorkflowSource`` compiles the valid fixture to a
  ``Workflow`` and persists a passing report.
- ac-008: ``ValidateWorkflowSource`` persists a failing report then raises
  ``StagePersistedFailureError`` on each invalid fixture.
- ac-009: untrusted source is ast-validated before any ``exec``.

Fixtures
--------
``VALID_SOURCE`` was verified against the real public ``molexp.workflow``
surface (``WorkflowCompiler`` + decorator ``@wf.task`` + ``.compile()``) — it
compiles to a ``Workflow``. The GREEN impl's restricted-exec namespace must
accept exactly this shape: a module-level ``build_workflow()`` returning a
``WorkflowCompiler`` whose ``.compile()`` succeeds.
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
from molexp.workflow._pydantic_graph import something
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
    from molexp.harness.store.sqlite_provenance_store import SQLiteProvenanceStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteProvenanceStore(path=db, artifact_store=a)
    return HarnessRunContext(
        run_id="run-vws",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        provenance_store=p,
    )


def _seed_workflow_source(ctx, source: str = VALID_SOURCE):
    return ctx.artifact_store.put_json(
        kind="workflow_source",
        obj=_workflow_source_dict(source),
        created_by="seed",
        parent_ids=[],
    )


# ----------------------------------------------------------- ac-001 schema


def test_workflow_source_schema_fields_and_frozen() -> None:
    from molexp.harness.schemas.workflow_source import WorkflowSource

    ws = WorkflowSource(
        source=VALID_SOURCE,
        module_name="generated_workflow",
        bound_workflow_id="bw-1",
        symbols=("WorkflowCompiler", "Task"),
    )
    assert ws.source == VALID_SOURCE
    assert ws.module_name == "generated_workflow"
    assert ws.bound_workflow_id == "bw-1"
    assert ws.symbols == ("WorkflowCompiler", "Task")


def test_workflow_source_is_frozen() -> None:
    from pydantic import ValidationError

    from molexp.harness.schemas.workflow_source import WorkflowSource

    ws = WorkflowSource(
        source="x",
        module_name="m",
        bound_workflow_id="bw",
        symbols=(),
    )
    with pytest.raises(ValidationError):
        ws.source = "mutated"  # type: ignore[misc]


def test_workflow_source_reexported_from_harness() -> None:
    import molexp.harness as h
    from molexp.harness.schemas import WorkflowSource as FromSchemas
    from molexp.harness.schemas.workflow_source import WorkflowSource as Canonical

    assert h.WorkflowSource is Canonical
    assert FromSchemas is Canonical


# ------------------------------------------------- ac-002 kind + target kind


def test_workflow_source_in_well_known_artifact_kinds() -> None:
    from molexp.harness.schemas.artifact import WELL_KNOWN_ARTIFACT_KINDS

    assert "workflow_source" in WELL_KNOWN_ARTIFACT_KINDS


def test_validation_report_accepts_workflow_source_target_kind() -> None:
    from molexp.harness.schemas.validation import ValidationReport

    report = ValidationReport.from_violations(
        target_kind="workflow_source",
        target_id="ws-1",
        violations=[],
    )
    assert report.target_kind == "workflow_source"
    assert report.passed is True


# --------------------------------------------------------- ac-003 prompt


def test_system_prompt_names_public_api() -> None:
    from molexp.harness.prompts.workflow_source import SYSTEM_PROMPT

    for symbol in ("WorkflowCompiler", "Task", "Actor", "Workflow"):
        assert symbol in SYSTEM_PROMPT, f"{symbol} missing from SYSTEM_PROMPT"
    assert "molexp.workflow" in SYSTEM_PROMPT


# ----------------------------------------------- ac-004 pure validator syntax


def test_validate_workflow_source_flags_syntax_error_without_raising() -> None:
    from molexp.harness.validators.workflow_source import validate_workflow_source

    report = validate_workflow_source(SYNTAX_ERROR_SOURCE)
    assert report.passed is False
    assert report.target_kind == "workflow_source"
    assert len(report.violations) >= 1
    # A syntax violation must be reported (code mentions syntax).
    assert any("syntax" in v.code.lower() for v in report.violations)


def test_validate_workflow_source_never_raises_on_garbage() -> None:
    from molexp.harness.validators.workflow_source import validate_workflow_source

    # Total function: even on wildly malformed input no exception escapes.
    for bad in ("def (:\n", "@@@@", "import", "class :", "\x00\x01"):
        report = validate_workflow_source(bad)
        assert report.passed is False


# ----------------------------------------- ac-005 pure validator private import


def test_validate_workflow_source_rejects_private_subpackage_import() -> None:
    from molexp.harness.validators.workflow_source import validate_workflow_source

    report = validate_workflow_source(PRIVATE_IMPORT_SOURCE)
    assert report.passed is False
    # A violation must name the disallowed private import target.
    assert any("_pydantic_graph" in (v.message + (v.path or "")) for v in report.violations)


def test_validate_workflow_source_passes_public_surface_only() -> None:
    from molexp.harness.validators.workflow_source import validate_workflow_source

    report = validate_workflow_source(VALID_SOURCE)
    assert report.passed is True
    assert report.violations == []


# ------------------------------------ ac-007 stage compiles valid → Workflow


def test_validate_workflow_source_compiles_valid_to_workflow(ctx) -> None:
    from molexp.harness.schemas.validation import ValidationReport
    from molexp.harness.stages.validate_workflow_source import ValidateWorkflowSource

    ws_ref = _seed_workflow_source(ctx, VALID_SOURCE)
    stage = ValidateWorkflowSource()
    report_ref = asyncio.run(stage.run(ctx))

    assert report_ref.kind == "validation_report"
    assert ws_ref.id in report_ref.parent_ids

    raw = ctx.artifact_store.get(report_ref.id)
    report = ValidationReport.model_validate(json.loads(raw))
    assert report.passed is True
    assert report.target_kind == "workflow_source"


def test_validate_workflow_source_stage_name() -> None:
    from molexp.harness.stages.validate_workflow_source import ValidateWorkflowSource

    assert ValidateWorkflowSource.name == "validate_workflow_source"


def test_validate_workflow_source_is_stage_subclass() -> None:
    from molexp.harness.core.stage import Stage
    from molexp.harness.stages.validate_workflow_source import ValidateWorkflowSource

    assert issubclass(ValidateWorkflowSource, Stage)


# -------------------------- ac-008 invalid fixtures persist + raise


@pytest.mark.parametrize(
    "source",
    [SYNTAX_ERROR_SOURCE, PRIVATE_IMPORT_SOURCE, BUILD_FAILS_SOURCE],
    ids=["syntax_error", "private_import", "build_fails"],
)
def test_validate_workflow_source_persists_report_and_raises(ctx, source: str) -> None:
    from molexp.harness.errors import StageExecutionError, StagePersistedFailureError
    from molexp.harness.schemas.validation import ValidationReport
    from molexp.harness.stages.validate_workflow_source import ValidateWorkflowSource

    _seed_workflow_source(ctx, source)
    stage = ValidateWorkflowSource()

    with pytest.raises(StageExecutionError) as exc_info:
        asyncio.run(stage.run(ctx))
    assert isinstance(exc_info.value, StagePersistedFailureError)

    # Report persisted despite the raise (always-persist contract).
    reports = ctx.artifact_store.list_by_kind("validation_report")
    assert len(reports) == 1
    raw = ctx.artifact_store.get(reports[0].id)
    report = ValidationReport.model_validate(json.loads(raw))
    assert report.passed is False
    assert report.target_kind == "workflow_source"
    # persisted_ref points at the persisted failing report.
    assert exc_info.value.persisted_ref.id == reports[0].id


# ---------------------------- ac-009 ast-validated before exec / restricted


def test_syntax_and_private_rejected_before_exec(ctx, monkeypatch) -> None:
    """ac-009: syntax-error and private-import fixtures are rejected at the
    ast/compile pre-check stage — ``exec`` is never reached for them.

    We trip a sentinel if ``builtins.exec`` were called; the stage must
    fail validation purely from the AST pre-checks.
    """
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
        stage = ValidateWorkflowSource()
        with pytest.raises(StagePersistedFailureError):
            asyncio.run(stage.run(ctx))
        assert exec_calls == [], f"exec must not run for ast-rejected source ({source[:20]!r})"


def test_valid_source_exec_uses_restricted_builtins(ctx, monkeypatch) -> None:
    """ac-009: the valid fixture IS executed, but the exec namespace's
    ``__builtins__`` is restricted (not the full real builtins module).
    """
    import builtins

    from molexp.harness.stages.validate_workflow_source import ValidateWorkflowSource

    captured_globals: list[dict] = []
    real_exec = builtins.exec

    def _capturing_exec(code, globals_ns=None, locals_ns=None):
        captured_globals.append(globals_ns if globals_ns is not None else {})
        return real_exec(code, globals_ns, locals_ns)

    monkeypatch.setattr(builtins, "exec", _capturing_exec)

    _seed_workflow_source(ctx, VALID_SOURCE)
    stage = ValidateWorkflowSource()
    asyncio.run(stage.run(ctx))

    assert captured_globals, "exec was expected to run for the valid fixture"
    ns = captured_globals[0]
    assert "__builtins__" in ns, "exec namespace must define __builtins__"
    # The restricted builtins must NOT be the full real builtins module.
    assert ns["__builtins__"] is not builtins
    assert ns["__builtins__"] is not builtins.__dict__
