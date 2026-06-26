"""``ValidationReport`` + ``ValidationViolation`` — the validator return contract.

Every harness validator (:func:`molexp.harness.validators.workflow_ir.validate_workflow_ir`,
:func:`molexp.harness.validators.bound_workflow.validate_bound_workflow`)
returns a :class:`ValidationReport`. Validators never raise; failures
surface as :class:`ValidationViolation` entries with machine-readable
``code`` strings.

``ValidationReport.passed`` is derived: ``True`` iff zero ``severity =
"error"`` violations exist. Warning-only reports still pass. This gives
the Phase-4 stage wrappers a clean tri-state mental model (clean /
pass-with-warnings / fail) without needing to recompute the flag.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ValidationReport", "ValidationViolation"]


_TARGET_KINDS = Literal[
    "experiment_spec",
    "workflow_ir",
    "bound_workflow",
    "workflow_source",
    "test_spec",
    "test_source",
    "input_set",
    "provenance",
]
_SEVERITIES = Literal["error", "warning"]


class ValidationViolation(BaseModel):
    """One thing the validator found wrong (or worth flagging)."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    path: str | None = None
    severity: _SEVERITIES = "error"


class ValidationReport(BaseModel):
    """Result of one validator pass."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    violations: list[ValidationViolation] = Field(default_factory=list)
    target_kind: _TARGET_KINDS
    target_id: str

    @classmethod
    def from_violations(
        cls,
        target_kind: _TARGET_KINDS,
        target_id: str,
        violations: list[ValidationViolation],
    ) -> ValidationReport:
        """Build a report with ``passed`` derived from the violations list."""
        passed = not any(v.severity == "error" for v in violations)
        return cls(
            passed=passed,
            violations=list(violations),
            target_kind=target_kind,
            target_id=target_id,
        )
