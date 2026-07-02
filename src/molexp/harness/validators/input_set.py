"""Pure structural validator for :class:`InputSet` (plan step 6).

Checks the declarative parameter-space spec against the PlanWorkflowIR it
expands: every swept axis and every ``fixed_params`` key names a real root
input, the two channels are disjoint, no axis is empty, no axis targets a
list-valued root input (an axis delivers one scalar per cell — a list-valued
input the workflow scans internally belongs in ``fixed_params``, whole), and
a grid's ``total_runs`` equals the Cartesian product of its axis lengths.
Pure, sync, no I/O, never raises.
"""

from __future__ import annotations

from math import prod

from molexp.harness.schemas.input_set import InputSet
from molexp.harness.schemas.validation import PlanValidationReport, ValidationViolation
from molexp.harness.schemas.workflow_ir import PlanWorkflowIR

__all__ = ["InputSetValidator"]


class InputSetValidator:
    @staticmethod
    def validate(input_set: InputSet, *, ir: PlanWorkflowIR | None = None) -> PlanValidationReport:
        violations: list[ValidationViolation] = []

        ir_input_keys = set(ir.inputs.keys()) if ir is not None else None
        axis_names = {axis.name for axis in input_set.sweep_axes}
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
                            f"sweep axis {axis.name!r} is not a PlanWorkflowIR root input "
                            f"(known inputs: {sorted(ir_input_keys)})"
                        ),
                        path=f"sweep_axes[{i}].name",
                    )
                )
            elif ir is not None and isinstance(ir.inputs[axis.name].value, list):
                # An axis delivers one scalar per cell; sweeping a list-valued
                # root input would change the parameter's shape (the task
                # iterates it — production crash: `for sigma in 0.9`).
                violations.append(
                    ValidationViolation(
                        code="axis_targets_list_param",
                        message=(
                            f"sweep axis {axis.name!r} targets a list-valued root input "
                            f"(IR declares {ir.inputs[axis.name].value!r}) — an axis "
                            "delivers one scalar per cell; pass this input whole via "
                            "fixed_params instead"
                        ),
                        path=f"sweep_axes[{i}].name",
                    )
                )

        for name in sorted(input_set.fixed_params):
            if ir_input_keys is not None and name not in ir_input_keys:
                violations.append(
                    ValidationViolation(
                        code="unknown_fixed_param",
                        message=(
                            f"fixed param {name!r} is not a PlanWorkflowIR root input "
                            f"(known inputs: {sorted(ir_input_keys)})"
                        ),
                        path=f"fixed_params[{name!r}]",
                    )
                )
            if name in axis_names:
                violations.append(
                    ValidationViolation(
                        code="axis_fixed_overlap",
                        message=(
                            f"root input {name!r} appears both as a sweep axis and in "
                            "fixed_params — the two channels are disjoint by definition; "
                            "keep exactly one"
                        ),
                        path=f"fixed_params[{name!r}]",
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

        return PlanValidationReport.from_violations(
            target_kind="input_set",
            target_id=input_set.id,
            violations=violations,
        )
