"""``ValidateWorkflowSource`` — prove generated source is runnable molexp.workflow.

Loads a :class:`WorkflowSource` artifact, runs the pure
:func:`validate_workflow_source` pre-checks (syntax + public-surface imports),
and only if those pass **lazily imports** ``molexp.workflow`` to compile the
source into a real ``Workflow`` (calling the program's ``build_workflow()`` and
``.build()``). A :class:`ValidationReport` is **always persisted**; on failure
the stage raises :class:`StagePersistedFailureError` (mirroring
:class:`ValidateWorkflowIR`).

Security — the source is untrusted LLM output:

* The pure ast-based pre-checks reject syntax errors and private-submodule
  imports **before any code runs** — `compile`/`exec` are never reached for
  those inputs.
* Execution then happens via ``exec`` in a constrained namespace with a
  restricted ``__builtins__`` (a small allow-list, not the real ``builtins``
  module), so the generated program cannot reach arbitrary builtins. The lazy
  ``import molexp.workflow`` also keeps the harness import-guard green (the
  workflow engine is never loaded at harness import time).

This is a structural compile gate, not a full sandbox/jail: it proves the
source *builds* into a Workflow. Executing the resulting Workflow is a separate
concern (harness executors).
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StagePersistedFailureError
from molexp.harness.schemas import (
    ArtifactRef,
    ValidationReport,
    ValidationViolation,
    WorkflowSource,
)
from molexp.harness.stages._resolve import require_latest
from molexp.harness.validators.workflow_source import validate_workflow_source

__all__ = ["ValidateWorkflowSource"]

# Minimal builtins the generated program may use to define its tasks. Kept
# small on purpose — the source targets the public molexp.workflow surface, not
# arbitrary runtime behaviour.
_SAFE_BUILTINS: dict[str, Any] = {
    "__import__": __import__,  # needed for the source's own `import molexp.workflow`
    "len": len,
    "range": range,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "sum": sum,
    "min": min,
    "max": max,
}


class ValidateWorkflowSource(Stage):
    """Compile a WorkflowSource artifact through molexp.workflow; persist a report."""

    name: ClassVar[str] = "validate_workflow_source"

    def __init__(self, *, raise_on_failure: bool = True) -> None:
        self._raise_on_failure = raise_on_failure

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        target = require_latest(ctx, "workflow_source", stage=self.name).id
        raw = ctx.artifact_store.get(target)

        try:
            ws = WorkflowSource.model_validate_json(raw)
        except Exception as exc:
            return self._persist_and_maybe_raise(
                ctx,
                [
                    ValidationViolation(
                        code="workflow_source_parse_error",
                        message=f"WorkflowSource JSON failed schema validation: {exc!r}",
                        severity="error",
                    )
                ],
                f"WorkflowSource parse failed: {exc!r}",
                target=target,
            )

        # Pure pre-checks first — reject syntax errors + private imports BEFORE
        # any compile/exec of the untrusted source.
        report = validate_workflow_source(ws.source, target_id=target)
        if not report.passed:
            return self._persist_report_and_maybe_raise(ctx, report, target=target)

        # Pre-checks passed: lazily import the engine and compile the program.
        violations = self._compile_violations(ws.source)
        return self._persist_and_maybe_raise(
            ctx,
            violations,
            f"generated workflow source did not build: {[v.code for v in violations]}",
            target=target,
        )

    def _compile_violations(self, source: str) -> list[ValidationViolation]:
        """Compile + build the source via molexp.workflow; return any violations."""
        import molexp.workflow as workflow  # lazy — keeps the harness import-guard green

        namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
        try:
            compiled = compile(source, "<workflow_source>", "exec")
            exec(compiled, namespace)
        except Exception as exc:
            return [
                ValidationViolation(
                    code="exec_error",
                    message=f"generated source failed to execute: {exc!r}",
                    severity="error",
                )
            ]

        builder_factory = namespace.get("build_workflow")
        if not callable(builder_factory):
            return [
                ValidationViolation(
                    code="missing_build_workflow",
                    message="generated source defines no callable build_workflow()",
                    severity="error",
                )
            ]

        try:
            builder = builder_factory()
            result = builder.compile()
        except Exception as exc:
            return [
                ValidationViolation(
                    code="build_error",
                    message=f"build_workflow().compile() failed: {exc!r}",
                    severity="error",
                )
            ]

        if not isinstance(result, workflow.CompiledWorkflow):
            return [
                ValidationViolation(
                    code="not_a_workflow",
                    message=(
                        f"build_workflow().compile() returned {type(result).__name__}, "
                        "not a CompiledWorkflow"
                    ),
                    severity="error",
                )
            ]
        return []

    def _persist_and_maybe_raise(
        self,
        ctx: HarnessRunContext,
        violations: list[ValidationViolation],
        error_message: str,
        *,
        target: str,
    ) -> ArtifactRef:
        report = ValidationReport.from_violations(
            target_kind="workflow_source",
            target_id=target,
            violations=violations,
        )
        return self._persist_report_and_maybe_raise(ctx, report, error_message, target=target)

    def _persist_report_and_maybe_raise(
        self,
        ctx: HarnessRunContext,
        report: ValidationReport,
        error_message: str | None = None,
        *,
        target: str,
    ) -> ArtifactRef:
        report_ref = ctx.artifact_store.put_json(
            kind="validation_report",
            obj=json.loads(report.model_dump_json()),
            created_by="ValidateWorkflowSource",
            parent_ids=[target],
        )
        if not report.passed and self._raise_on_failure:
            codes = [v.code for v in report.violations if v.severity == "error"]
            raise StagePersistedFailureError(
                report_ref,
                error_message or f"workflow source validation failed: {codes}",
            )
        return report_ref
