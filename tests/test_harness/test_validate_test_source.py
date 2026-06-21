"""Tests for ``validate_test_source`` (pure validator) + the
``ValidateTestSource`` stage (spec ``harness-run-mode-01-substrate``, T03).

Pure validator contract (mirrors ``validators/workflow_source.py``):
``TestSourceValidator.validate(source: str, *, target_id: str) -> ValidationReport``
with ``target_kind == "test_source"``. Checks are parse/compile-time ONLY —
``ast.parse`` syntax check, private ``molexp.workflow._*`` import scan, at
least one ``def test_*`` function, and a ``compile(...)`` byte-compile
pre-check. The validator must NEVER ``exec`` the untrusted source and never
raise: the no-exec proof fixture carries a module-level
``raise RuntimeError`` yet must produce a PASSING report.

Stage contract (mirrors ``ValidateWorkflowSource``): always-persist a
``"validation_report"`` artifact (parents = the test_source artifact id),
persist-then-raise ``StagePersistedFailureError`` on failure, and a
``raise_on_failure=False`` knob that returns the failing ref instead.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from molexp.harness import ArtifactRef
    from molexp.harness.core.run_context import HarnessRunContext

# --------------------------------------------------------------- fixtures

VALID_TEST_SOURCE = (
    "from generated_workflow import build_workflow\n"
    "\n"
    "\n"
    "def test_ok():\n"
    "    assert callable(build_workflow)\n"
)

# (a) syntax error — ast.parse fails.
SYNTAX_ERROR_SOURCE = "def (:\n    pass\n"

# (b) imports a private subpackage of molexp.workflow.
PRIVATE_IMPORT_SOURCE = """\
from molexp.workflow._pydantic_graph import engine


def test_sneaky():
    assert engine is not None
"""

# (c) syntactically fine but defines no test_* function.
NO_TEST_FUNCTION_SOURCE = """\
from generated_workflow import build_workflow


def helper():
    return build_workflow
"""

# (d) no-exec proof: a valid test function PLUS a module-level raise.
# ast.parse and compile() both succeed; only exec()ing the module would
# trip the RuntimeError — so a PASSING report proves the validator never
# executes the untrusted source.
MODULE_RAISE_SOURCE = """\
def test_ok():
    assert True


raise RuntimeError("must not execute")
"""

# (e) parses (ast.parse succeeds) but the byte-compile pre-check fails:
# 'break' outside a loop is rejected by compile(), not by ast.parse —
# verified against CPython 3.12. A failing report proves the compile()
# stage actually runs.
COMPILE_STAGE_ERROR_SOURCE = """\
def test_ok():
    assert True


break
"""


@pytest.fixture()
def ctx(tmp_path: Path) -> HarnessRunContext:
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "events.sqlite"
    a = FileArtifactStore(root=tmp_path / "artifacts")
    e = SQLiteEventLog(path=db)
    p = SQLiteArtifactLineageStore(path=db, artifact_store=a)
    return HarnessRunContext(
        run_id="run-vtsrc",
        workspace_root=tmp_path,
        artifact_store=a,
        event_log=e,
        lineage_store=p,
    )


def _test_source_dict(source: str) -> dict:
    from molexp.harness import TestSource

    ts = TestSource(
        source=source,
        module_name="test_generated_workflow",
        test_spec_id="ts-001",
        bound_workflow_id="bw-x",
    )
    return json.loads(ts.model_dump_json())


def _seed_test_source(ctx: HarnessRunContext, source: str) -> ArtifactRef:
    return ctx.artifact_store.put_json(
        kind="test_source",
        obj=_test_source_dict(source),
        created_by="seed",
        parent_ids=[],
    )


class TestValidateTestSource:
    # ------------------------------------------------- pure validator: happy

    def test_validate_test_source_passes_valid_pytest_source(self) -> None:
        from molexp.harness import TestSourceValidator

        report = TestSourceValidator.validate(VALID_TEST_SOURCE, target_id="ts-art-1")
        assert report.passed is True
        assert report.violations == []
        assert report.target_kind == "test_source"
        assert report.target_id == "ts-art-1"

    # --------------------------------------------------- pure validator: red

    def test_validate_test_source_flags_syntax_error_without_raising(self) -> None:
        from molexp.harness import TestSourceValidator

        report = TestSourceValidator.validate(SYNTAX_ERROR_SOURCE, target_id="ts-art-1")
        assert report.passed is False
        assert report.target_kind == "test_source"
        assert any("syntax" in v.code.lower() for v in report.violations)

    def test_validate_test_source_never_raises_on_garbage(self) -> None:
        from molexp.harness import TestSourceValidator

        # Total function: even on wildly malformed input no exception escapes.
        for bad in ("def (:\n", "@@@@", "import", "class :", "\x00\x01"):
            report = TestSourceValidator.validate(bad, target_id="ts-art-1")
            assert report.passed is False

    def test_validate_test_source_rejects_private_workflow_import(self) -> None:
        from molexp.harness import TestSourceValidator

        report = TestSourceValidator.validate(PRIVATE_IMPORT_SOURCE, target_id="ts-art-1")
        assert report.passed is False
        # A violation must name the disallowed private import target.
        assert any("_pydantic_graph" in (v.message + (v.path or "")) for v in report.violations)

    def test_validate_test_source_requires_a_test_function(self) -> None:
        from molexp.harness import TestSourceValidator

        report = TestSourceValidator.validate(NO_TEST_FUNCTION_SOURCE, target_id="ts-art-1")
        assert report.passed is False
        assert any("test" in v.code.lower() for v in report.violations)

    # ------------------------------------------ pure validator: no-exec proof

    def test_validate_test_source_never_executes_module_code(self) -> None:
        """Parse + compile ONLY: a module-level ``raise`` must not fire."""
        from molexp.harness import TestSourceValidator

        report = TestSourceValidator.validate(MODULE_RAISE_SOURCE, target_id="ts-art-1")
        assert report.passed is True
        assert report.violations == []

    def test_validate_test_source_runs_byte_compile_stage(self) -> None:
        """'break' outside a loop passes ast.parse but fails compile() — a
        failing report proves the byte-compile pre-check actually runs."""
        from molexp.harness import TestSourceValidator

        report = TestSourceValidator.validate(COMPILE_STAGE_ERROR_SOURCE, target_id="ts-art-1")
        assert report.passed is False
        assert len(report.violations) >= 1

    # ------------------------------------------------------------ stage shape

    def test_stage_name_and_subclass(self) -> None:
        from molexp.harness import ValidateTestSource
        from molexp.harness.core.stage import Stage

        assert ValidateTestSource.name == "validate_test_source"
        assert issubclass(ValidateTestSource, Stage)

    # ------------------------------------------------------- stage happy path

    def test_stage_persists_passing_report_for_good_source(self, ctx) -> None:
        from molexp.harness import ValidateTestSource, ValidationReport

        ts_ref = _seed_test_source(ctx, VALID_TEST_SOURCE)
        report_ref = asyncio.run(ValidateTestSource().run(ctx))

        assert report_ref.kind == "validation_report"
        assert ts_ref.id in report_ref.parent_ids

        raw = ctx.artifact_store.get(report_ref.id)
        report = ValidationReport.model_validate(json.loads(raw))
        assert report.passed is True
        assert report.target_kind == "test_source"

    # --------------------------------------------------------- stage red path

    def test_stage_persists_failing_report_then_raises(self, ctx) -> None:
        from molexp.harness import StagePersistedFailureError, ValidateTestSource, ValidationReport

        _seed_test_source(ctx, SYNTAX_ERROR_SOURCE)

        with pytest.raises(StagePersistedFailureError) as exc_info:
            asyncio.run(ValidateTestSource().run(ctx))

        # Report persisted despite the raise (always-persist contract).
        reports = ctx.artifact_store.list_by_kind("validation_report")
        assert len(reports) == 1
        raw = ctx.artifact_store.get(reports[0].id)
        report = ValidationReport.model_validate(json.loads(raw))
        assert report.passed is False
        assert report.target_kind == "test_source"
        assert exc_info.value.persisted_ref.id == reports[0].id
        assert exc_info.value.persisted_ref.kind == "validation_report"

    def test_stage_returns_failing_ref_when_raise_disabled(self, ctx) -> None:
        from molexp.harness import ValidateTestSource, ValidationReport

        ts_ref = _seed_test_source(ctx, SYNTAX_ERROR_SOURCE)
        report_ref = asyncio.run(ValidateTestSource(raise_on_failure=False).run(ctx))

        assert report_ref.kind == "validation_report"
        assert ts_ref.id in report_ref.parent_ids
        raw = ctx.artifact_store.get(report_ref.id)
        report = ValidationReport.model_validate(json.loads(raw))
        assert report.passed is False


# ------------------------------------------ per-task coverage (required_task_ids)


_TWO_TASK_SOURCE = (
    "def test_build_ok():\n    assert True\n\n\ndef test_relax_ok():\n    assert True\n"
)


class TestValidateTestSourcePerTask:
    """Per-task coverage enforcement via ``required_task_ids``."""

    def test_rejects_module_missing_a_per_task_test(self) -> None:
        """ac-005 — a module covering only some required tasks fails with a
        ``missing_task_test`` error for the uncovered one."""
        from molexp.harness import TestSourceValidator

        report = TestSourceValidator.validate(
            _TWO_TASK_SOURCE,
            target_id="ts-art-1",
            required_task_ids={"build", "relax", "analyze"},
        )
        assert report.passed is False
        missing = [v for v in report.violations if v.code == "missing_task_test"]
        assert len(missing) == 1
        assert "analyze" in missing[0].message

    def test_accepts_a_test_per_required_task(self) -> None:
        """ac-005 — every required task covered by a ``test_*`` → passes."""
        from molexp.harness import TestSourceValidator

        report = TestSourceValidator.validate(
            _TWO_TASK_SOURCE,
            target_id="ts-art-1",
            required_task_ids={"build", "relax"},
        )
        assert report.passed is True
        assert report.violations == []

    def test_normalizes_non_identifier_task_ids(self) -> None:
        """A hyphenated task id is matched by its identifier-safe token."""
        from molexp.harness import TestSourceValidator

        source = "def test_b_build_ok():\n    assert True\n"
        report = TestSourceValidator.validate(
            source, target_id="ts-art-1", required_task_ids={"b-build"}
        )
        assert report.passed is True

    def test_none_required_keeps_legacy_behaviour(self) -> None:
        """required_task_ids=None → only the legacy 'at least one test' check."""
        from molexp.harness import TestSourceValidator

        report = TestSourceValidator.validate(VALID_TEST_SOURCE, target_id="ts-art-1")
        assert report.passed is True
