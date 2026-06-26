"""Pure structural validator for :class:`InputSet` (plan step 6).

Checks the declarative parameter-space spec against the WorkflowIR it
expands: every swept axis names a real root input, no axis is empty, and a
grid's ``total_runs`` equals the Cartesian product of its axis lengths. Pure,
sync, no I/O, never raises.
"""

from __future__ import annotations

from math import prod

from molexp.harness.schemas.input_set import InputSet
from molexp.harness.schemas.validation import ValidationReport, ValidationViolation
from molexp.harness.schemas.workflow_ir import WorkflowIR

__all__ = ["InputSetValidator"]


class InputSetValidator:
    @staticmethod
    def validate(input_set: InputSet, *, ir: WorkflowIR | None = None) -> ValidationReport:
        violations: list[ValidationViolation] = []

        ir_input_keys = set(ir.inputs.keys()) if ir is not None else None
        for i, axis in enumerate(input_set.sweep_axes):
            if not axis.values:
                violations.append(
                    ValidationViolation(
                        code="empty_axis",
                        message=f"sweep axis {axis.name!r} has no values",
                        path=f"sweep_axes[{i}].values",
                    )
                )
            if ir_input_keys is not None and axis.name not in ir_input_keys:
                violations.append(
                    ValidationViolation(
                        code="unknown_axis",
                        message=(
                            f"sweep axis {axis.name!r} is not a WorkflowIR root input "
                            f"(known inputs: {sorted(ir_input_keys)})"
                        ),
                        path=f"sweep_axes[{i}].name",
                    )
                )

        if input_set.strategy == "grid" and input_set.sweep_axes:
            expected = prod(len(axis.values) for axis in input_set.sweep_axes)
            if input_set.total_runs != expected:
                violations.append(
                    ValidationViolation(
                        code="total_runs_mismatch",
                        message=(
                            f"grid total_runs={input_set.total_runs} != product of axis "
                            f"lengths ({expected})"
                        ),
                        path="total_runs",
                    )
                )

        return ValidationReport.from_violations(
            target_kind="input_set",
            target_id=input_set.id,
            violations=violations,
        )
