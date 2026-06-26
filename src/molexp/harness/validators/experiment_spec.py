"""Pure structural validator for :class:`ExperimentSpec` (plan step 2).

Checks that the concrete spec actually concretizes the human-readable
report it derives from: every open ``user_questions`` entry is answered and,
when the report named variables, the spec pinned at least one. Pure, sync,
no I/O, never raises — returns a :class:`ValidationReport` the owning
``ValidateExperimentSpec`` stage lifts to an error.
"""

from __future__ import annotations

from molexp.harness.schemas.experiment_report import ExperimentReport
from molexp.harness.schemas.experiment_spec import ExperimentSpec
from molexp.harness.schemas.validation import ValidationReport, ValidationViolation

__all__ = ["ExperimentSpecValidator"]


class ExperimentSpecValidator:
    @staticmethod
    def validate(
        spec: ExperimentSpec, *, report: ExperimentReport | None = None
    ) -> ValidationReport:
        violations: list[ValidationViolation] = []

        if report is not None:
            answered = {q.question.strip() for q in spec.resolved_questions}
            for question in report.user_questions:
                if question.strip() not in answered:
                    violations.append(
                        ValidationViolation(
                            code="unresolved_question",
                            message=f"open question is not resolved in the spec: {question!r}",
                            path="resolved_questions",
                        )
                    )
            if report.variables and not spec.variables:
                violations.append(
                    ValidationViolation(
                        code="no_variables_concretized",
                        message="the report names variables but the spec concretized none",
                        path="variables",
                        severity="warning",
                    )
                )

        # A blank resolved answer is as good as unresolved.
        for i, answer in enumerate(spec.resolved_questions):
            if not answer.answer.strip():
                violations.append(
                    ValidationViolation(
                        code="empty_answer",
                        message=f"resolved_questions[{i}] has an empty answer",
                        path=f"resolved_questions[{i}].answer",
                    )
                )

        return ValidationReport.from_violations(
            target_kind="experiment_spec",
            target_id=spec.id,
            violations=violations,
        )
