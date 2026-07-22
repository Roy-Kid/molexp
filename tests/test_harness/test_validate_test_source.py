"""Tests for ``TestSourceValidator`` (pure) and the ``ValidateTestSource`` stage.

Pure validator (``validators/test_source.py``): parse/compile-time checks ONLY
— ``ast.parse`` syntax, private ``molexp.workflow._*`` import scan, at least one
``def test_*``, per-task coverage via ``required_task_ids``, and a byte-compile
pre-check. It NEVER ``exec``s the untrusted source (a module-level ``raise`` must
not fire) and NEVER raises (malformed input → failing report).

Stage (``stages/validate_test_source.py``): always-persist a
``"validation_report"`` (parents = the test_source artifact), persist-then-raise
``StagePersistedFailureError`` on failure, and a ``raise_on_failure=False`` knob.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from molexp.harness.core.run_context import HarnessRunContext
    from molexp.harness.schemas import PlanArtifactRef

# --------------------------------------------------------------- fixtures

VALID_TEST_SOURCE = (
    "from generated_workflow import build_workflow\n"
    "\n"
    "\n"
    "def test_ok():\n"
    "    assert callable(build_workflow)\n"
)

# ast.parse fails.
SYNTAX_ERROR_SOURCE = "def (:\n    pass\n"

# imports a private subpackage of molexp.workflow.
PRIVATE_IMPORT_SOURCE = """\
from molexp.workflow._engine import engine


def test_sneaky():
    assert engine is not None
"""

# syntactically fine but defines no test_* function.
NO_TEST_FUNCTION_SOURCE = """\
from generated_workflow import build_workflow


def helper():
    return build_workflow
"""

# no-exec proof: a valid test PLUS a module-level raise. ast.parse and compile()
# both succeed; only exec()ing the module would trip the RuntimeError — so a
# PASSING report proves the validator never executes the source.
MODULE_RAISE_SOURCE = """\
def test_ok():
    assert True


raise RuntimeError("must not execute")
"""

# parses but the byte-compile pre-check fails: 'break' outside a loop is rejected
# by compile(), not by ast.parse — a failing report proves compile() runs.
COMPILE_STAGE_ERROR_SOURCE = """\
def test_ok():
    assert True


break
"""

_TWO_TASK_SOURCE = (
    "def test_build_ok():\n    assert True\n\n\ndef test_relax_ok():\n    assert True\n"
)


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
    from molexp.harness.schemas import TestSource

    ts = TestSource(
        source=source,
        module_name="test_generated_workflow",
        test_spec_id="ts-001",
        bound_workflow_id="bw-x",
    )
    return json.loads(ts.model_dump_json())


def _seed_test_source(ctx: HarnessRunContext, source: str) -> PlanArtifactRef:
    return ctx.artifact_store.put_json(
        kind="test_source",
        obj=_test_source_dict(source),
        created_by="seed",
        parent_ids=[],
    )


class TestTestSourceValidator:
    """Pure structural pre-checks — one per distinct code path."""

    def test_passes_valid_pytest_source(self) -> None:
        from molexp.harness.validators import TestSourceValidator

        report = TestSourceValidator.validate(VALID_TEST_SOURCE, target_id="ts-art-1")
        assert report.passed is True
        assert report.violations == []
        assert report.target_kind == "test_source"
        assert report.target_id == "ts-art-1"

    def test_flags_syntax_error_without_raising(self) -> None:
        from molexp.harness.validators import TestSourceValidator

        report = TestSourceValidator.validate(SYNTAX_ERROR_SOURCE, target_id="ts-art-1")
        assert report.passed is False
        assert report.target_kind == "test_source"
        assert any("syntax" in v.code.lower() for v in report.violations)

    def test_never_raises_on_malformed_input(self) -> None:
        """Total function: wildly malformed input (incl. null bytes, which take
        the ValueError branch) yields a failing report, never an exception."""
        from molexp.harness.validators import TestSourceValidator

        for bad in ("@@@@", "\x00\x01"):
            report = TestSourceValidator.validate(bad, target_id="ts-art-1")
            assert report.passed is False

    def test_rejects_private_workflow_import(self) -> None:
        from molexp.harness.validators import TestSourceValidator

        report = TestSourceValidator.validate(PRIVATE_IMPORT_SOURCE, target_id="ts-art-1")
        assert report.passed is False
        assert any("_engine" in (v.message + (v.path or "")) for v in report.violations)

    def test_requires_at_least_one_test_function(self) -> None:
        from molexp.harness.validators import TestSourceValidator

        report = TestSourceValidator.validate(NO_TEST_FUNCTION_SOURCE, target_id="ts-art-1")
        assert report.passed is False
        assert any("test" in v.code.lower() for v in report.violations)

    def test_never_executes_module_code(self) -> None:
        """Parse + compile ONLY: a module-level ``raise`` must not fire."""
        from molexp.harness.validators import TestSourceValidator

        report = TestSourceValidator.validate(MODULE_RAISE_SOURCE, target_id="ts-art-1")
        assert report.passed is True
        assert report.violations == []

    def test_runs_byte_compile_stage(self) -> None:
        """'break' outside a loop passes ast.parse but fails compile() — a
        failing report proves the byte-compile pre-check runs."""
        from molexp.harness.validators import TestSourceValidator

        report = TestSourceValidator.validate(COMPILE_STAGE_ERROR_SOURCE, target_id="ts-art-1")
        assert report.passed is False
        assert len(report.violations) >= 1

    def test_rejects_module_missing_a_per_task_test(self) -> None:
        """``required_task_ids`` — a module covering only some tasks fails with a
        ``missing_task_test`` error for the uncovered one."""
        from molexp.harness.validators import TestSourceValidator

        report = TestSourceValidator.validate(
            _TWO_TASK_SOURCE,
            target_id="ts-art-1",
            required_task_ids={"build", "relax", "analyze"},
        )
        assert report.passed is False
        missing = [v for v in report.violations if v.code == "missing_task_test"]
        assert len(missing) == 1
        assert "analyze" in missing[0].message

    def test_normalizes_non_identifier_task_ids(self) -> None:
        """A hyphenated task id is matched by its identifier-safe token."""
        from molexp.harness.validators import TestSourceValidator

        source = "def test_b_build_ok():\n    assert True\n"
        report = TestSourceValidator.validate(
            source, target_id="ts-art-1", required_task_ids={"b-build"}
        )
        assert report.passed is True


class TestValidateTestSource:
    """Stage wiring: always-persist report, persist-then-raise, knob."""

    def test_persists_passing_report_for_good_source(self, ctx) -> None:
        from molexp.harness.schemas import PlanValidationReport
        from molexp.harness.stages import ValidateTestSource

        ts_ref = _seed_test_source(ctx, VALID_TEST_SOURCE)
        report_ref = asyncio.run(ValidateTestSource().run(ctx))

        assert report_ref.kind == "validation_report"
        assert ts_ref.id in report_ref.parent_ids

        report = PlanValidationReport.model_validate(
            json.loads(ctx.artifact_store.get(report_ref.id))
        )
        assert report.passed is True
        assert report.target_kind == "test_source"

    def test_persists_failing_report_then_raises(self, ctx) -> None:
        from molexp.harness.errors import StagePersistedFailureError
        from molexp.harness.schemas import PlanValidationReport
        from molexp.harness.stages import ValidateTestSource

        _seed_test_source(ctx, SYNTAX_ERROR_SOURCE)

        with pytest.raises(StagePersistedFailureError) as exc_info:
            asyncio.run(ValidateTestSource().run(ctx))

        # Report persisted despite the raise (always-persist contract).
        reports = ctx.artifact_store.list_by_kind("validation_report")
        assert len(reports) == 1
        report = PlanValidationReport.model_validate(
            json.loads(ctx.artifact_store.get(reports[0].id))
        )
        assert report.passed is False
        assert report.target_kind == "test_source"
        assert exc_info.value.persisted_ref.id == reports[0].id
        assert exc_info.value.persisted_ref.kind == "validation_report"

    def test_returns_failing_ref_when_raise_disabled(self, ctx) -> None:
        from molexp.harness.schemas import PlanValidationReport
        from molexp.harness.stages import ValidateTestSource

        ts_ref = _seed_test_source(ctx, SYNTAX_ERROR_SOURCE)
        report_ref = asyncio.run(ValidateTestSource(raise_on_failure=False).run(ctx))

        assert report_ref.kind == "validation_report"
        assert ts_ref.id in report_ref.parent_ids
        report = PlanValidationReport.model_validate(
            json.loads(ctx.artifact_store.get(report_ref.id))
        )
        assert report.passed is False
