"""Structural validator for a plan :class:`ExperimentPlan`.

Grades the plan's opaque ``spec`` plus its :class:`~molexp.harness.plan.TaskBoard`.
The validator is pure — no I/O, no LLM, never raises.

``require_feasibility`` defaults to True for the post-probe final gate; the
in-loop ``should_stop`` guard uses ``require_feasibility=False`` because the
reachability probe runs only after the planning loop finishes.
"""

from __future__ import annotations

from molexp.harness.plan import ExperimentPlan
from molexp.harness.schemas.validation import PlanValidationReport, ValidationViolation

__all__ = ["PlanFormValidator"]


class PlanFormValidator:
    """Grade an :class:`ExperimentPlan` (spec + task board) for well-formedness."""

    @staticmethod
    def validate(
        plan: ExperimentPlan,
        *,
        require_feasibility: bool = True,
    ) -> PlanValidationReport:
        """Return a report grading ``plan``; never raises.

        Blocking (``error``) codes: ``spec_incomplete``, ``empty_board``,
        ``duplicate_task_id``, ``blank_task_id``, ``task_missing_acceptance``,
        and when ``require_feasibility`` is True, ``task_missing_feasibility``.
        Non-blocking (``warning``): ``task_unreachable``.
        """
        violations: list[ValidationViolation] = []

        spec = plan.spec
        target_id = str(spec.get("id") or spec.get("title") or "experiment_plan")

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

            if require_feasibility:
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


# Temporary alias during rename (remove once all call sites migrate).
EmergentPlanFormValidator = PlanFormValidator
