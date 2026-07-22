"""Structural validator for an emergent-plan :class:`ExperimentPlan`.

Grades the plan's opaque ``spec`` plus its :class:`~molexp.harness.plan.TaskBoard`
(the emergent PlanMode's step2 / step8 guard), one
:class:`~molexp.harness.schemas.validation.ValidationViolation.code` per defect.
The validator is a pure function — no I/O, no LLM, and it **never** raises — so
the caller can branch deterministically on
:attr:`~molexp.harness.schemas.validation.PlanValidationReport.passed`.

Severity follows the same convention as
:class:`molexp.harness.validators.workflow_ir.WorkflowIRValidator`: only
``severity="error"`` violations flip ``passed`` to ``False`` (derived in
:meth:`PlanValidationReport.from_violations`). An unreachable task is a
non-blocking ``warning`` — the plan is graded, not vetoed.
"""

from __future__ import annotations

from molexp.harness.plan import ExperimentPlan
from molexp.harness.schemas.validation import PlanValidationReport, ValidationViolation

__all__ = ["EmergentPlanFormValidator"]


class EmergentPlanFormValidator:
    """Grade an :class:`ExperimentPlan` (spec + task board) for well-formedness."""

    @staticmethod
    def validate(plan: ExperimentPlan) -> PlanValidationReport:
        """Return a report grading ``plan``; never raises.

        Blocking (``error``) codes: ``spec_incomplete`` (empty spec, or a
        missing/empty ``title`` / ``objective`` — one per field), ``empty_board``
        (zero tasks), ``duplicate_task_id`` (one per repeat), ``blank_task_id``
        (empty/whitespace id), ``task_missing_acceptance`` (no acceptance
        criteria), ``task_missing_feasibility`` (unprobed task). Non-blocking
        (``warning``) code: ``task_unreachable`` (a probed task whose feasibility
        verdict is ``reachable=False``).

        Args:
            plan: The experiment plan (opaque spec + task board) to grade.

        Returns:
            A :class:`PlanValidationReport` with ``target_kind="experiment_plan"``,
            ``target_id`` derived from ``spec["id"]`` / ``spec["title"]`` /
            ``"experiment_plan"``, and ``passed`` derived from the violations
            (only ``error`` severity blocks).
        """
        violations: list[ValidationViolation] = []

        spec = plan.spec
        target_id = str(spec.get("id") or spec.get("title") or "experiment_plan")

        # spec_incomplete — empty mapping, or a missing/empty required field.
        if not spec:
            violations.append(
                ValidationViolation(
                    code="spec_incomplete",
                    message="experiment spec is an empty mapping",
                    path="spec",
                )
            )
        else:
            for field in ("title", "objective"):
                if not spec.get(field):
                    violations.append(
                        ValidationViolation(
                            code="spec_incomplete",
                            message=f"spec.{field} is missing or empty",
                            path=f"spec.{field}",
                        )
                    )

        tasks = plan.board.tasks

        # empty_board — a plan with no tasks cannot be built.
        if not tasks:
            violations.append(
                ValidationViolation(
                    code="empty_board",
                    message="task board has no tasks",
                    path="board.tasks",
                )
            )

        seen: set[str] = set()
        for index, task in enumerate(tasks):
            tid = task.id
            if not tid.strip():
                violations.append(
                    ValidationViolation(
                        code="blank_task_id",
                        message="task id is empty or whitespace",
                        path=f"board.tasks[{index}].id",
                    )
                )
            elif tid in seen:
                violations.append(
                    ValidationViolation(
                        code="duplicate_task_id",
                        message=f"task id {tid!r} appears more than once",
                        path=f"board.tasks[id={tid}]",
                    )
                )
            seen.add(tid)

            if not task.acceptance:
                violations.append(
                    ValidationViolation(
                        code="task_missing_acceptance",
                        message=f"task {tid!r} has no acceptance criteria",
                        path=f"board.tasks[{index}].acceptance",
                    )
                )

            if task.feasibility is None:
                violations.append(
                    ValidationViolation(
                        code="task_missing_feasibility",
                        message=f"task {tid!r} has not been feasibility-probed",
                        path=f"board.tasks[{index}].feasibility",
                    )
                )
            elif not task.feasibility.reachable:
                violations.append(
                    ValidationViolation(
                        code="task_unreachable",
                        message=f"task {tid!r} is probed but not reachable",
                        path=f"board.tasks[{index}].feasibility",
                        severity="warning",
                    )
                )

        return PlanValidationReport.from_violations(
            target_kind="experiment_plan",
            target_id=target_id,
            violations=violations,
        )
