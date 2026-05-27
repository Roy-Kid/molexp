"""Structural validator for :class:`TestSpec` (Phase 5).

Seven structural / coherence checks. Pure function — no I/O, no LLM,
never raises. Every failure surfaces as a :class:`ValidationViolation`.

Two-mode contract:

- **shallow** (default): when ``ir`` and ``bw`` are both ``None``, only
  shape / target-presence / command-safety / tolerance-coherence checks
  fire. Useful when the caller wants to validate a TestSpec in isolation
  before any WorkflowIR has been built.

- **cross-checked**: when ``ir`` is supplied, additional codes
  ``unknown_task_target`` / ``unknown_workflow_target`` resolve the
  TestSpec's target against the supplied IR. ``bw`` is currently a
  parameter for symmetry with the Phase-3 validator family — Phase-6+
  may add bound-task-aware checks here.
"""

from __future__ import annotations

from molexp.harness.schemas.bound_workflow import BoundWorkflow
from molexp.harness.schemas.test_spec import TestSpec
from molexp.harness.schemas.validation import ValidationReport, ValidationViolation
from molexp.harness.schemas.workflow_ir import WorkflowIR

__all__ = ["validate_test_spec"]


# Defense-in-depth deny list applied per-argv-element. Shared in spirit
# with ``validators.workflow_ir._SHELL_DENY`` (we duplicate the tuple
# here rather than importing a leading-underscore symbol across modules).
_SHELL_DENY = (
    "bash",
    "sh -c",
    "os.system",
    "subprocess.run",
    "subprocess.call",
    "subprocess.Popen",
    ";",
    "&&",
    "||",
    "$(",
    "`",
)


def validate_test_spec(
    spec: TestSpec,
    *,
    ir: WorkflowIR | None = None,
    bw: BoundWorkflow | None = None,  # noqa: ARG001  — accepted for symmetry with Phase-3 validators; Phase-6 will use it
) -> ValidationReport:
    violations: list[ValidationViolation] = []

    # 1. missing_target / 2. ambiguous_target
    has_task = spec.target_task_id is not None
    has_workflow = spec.target_workflow_id is not None
    if not has_task and not has_workflow:
        violations.append(
            ValidationViolation(
                code="missing_target",
                message="TestSpec must declare either target_task_id or target_workflow_id",
                path="target_task_id|target_workflow_id",
            )
        )
    elif has_task and has_workflow:
        violations.append(
            ValidationViolation(
                code="ambiguous_target",
                message="TestSpec must declare exactly one of target_task_id / target_workflow_id, not both",
                path="target_task_id|target_workflow_id",
            )
        )

    # 3. unknown_task_target — only if ir provided
    if (
        ir is not None
        and spec.target_task_id is not None
        and spec.target_task_id not in {t.id for t in ir.tasks}
    ):
        violations.append(
            ValidationViolation(
                code="unknown_task_target",
                message=(
                    f"target_task_id {spec.target_task_id!r} not found "
                    f"in WorkflowIR {ir.id!r} tasks"
                ),
                path="target_task_id",
            )
        )

    # 4. unknown_workflow_target — only if ir provided
    if ir is not None and spec.target_workflow_id is not None and spec.target_workflow_id != ir.id:
        violations.append(
            ValidationViolation(
                code="unknown_workflow_target",
                message=(
                    f"target_workflow_id {spec.target_workflow_id!r} does not match "
                    f"WorkflowIR.id {ir.id!r}"
                ),
                path="target_workflow_id",
            )
        )

    # 5. tolerance_requires_metric (warning)
    expected_metric_keys = set(spec.expected_metrics.keys())
    for tol_key in spec.tolerance:
        if tol_key not in expected_metric_keys:
            violations.append(
                ValidationViolation(
                    code="tolerance_requires_metric",
                    message=(f"tolerance key {tol_key!r} has no matching expected_metrics entry"),
                    path=f"tolerance.{tol_key}",
                    severity="warning",
                )
            )

    # 6. command_with_shell (error)
    if spec.command is not None:
        for i, element in enumerate(spec.command):
            for needle in _SHELL_DENY:
                if needle in element:
                    violations.append(
                        ValidationViolation(
                            code="command_with_shell",
                            message=(
                                f"command[{i}]={element!r} contains deny-listed substring {needle!r}"
                            ),
                            path=f"command[{i}]",
                        )
                    )
                    break  # one report per element

    # 7. numerical_test_missing_tolerance (warning)
    if spec.kind == "numerical_tolerance_test" and not spec.tolerance:
        violations.append(
            ValidationViolation(
                code="numerical_test_missing_tolerance",
                message=(f"TestSpec kind={spec.kind!r} but tolerance is empty"),
                path="tolerance",
                severity="warning",
            )
        )

    return ValidationReport.from_violations(
        target_kind="test_spec",
        target_id=spec.id,
        violations=violations,
    )
