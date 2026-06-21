"""``WorkflowSourceValidator`` — pure structural pre-checks for generated source.

Side-effect-free checks run *before* any compile/exec of LLM-generated code:

1. **Syntax** — ``ast.parse``; a ``SyntaxError`` becomes a violation.
2. **Public-surface imports only** — an AST walk rejects any import of a
   private ``molexp.workflow`` submodule (anything under
   ``molexp.workflow._...``); generated code must target the public surface.

Returns a :class:`ValidationReport` (``target_kind="workflow_source"``) and
**never raises** — malformed input yields a failing report, not an exception.
This is the gate :class:`ValidateWorkflowSource` runs before it ever compiles
or executes the source.
"""

from __future__ import annotations

import ast

from molexp.harness.schemas.validation import ValidationReport, ValidationViolation

__all__ = ["WorkflowSourceValidator"]

_PRIVATE_PREFIX = "molexp.workflow._"


def _is_private_workflow(module: str | None) -> bool:
    """True if ``module`` names a private ``molexp.workflow`` submodule."""
    if not module:
        return False
    return module == "molexp.workflow._" or module.startswith(_PRIVATE_PREFIX)


class WorkflowSourceValidator:
    @staticmethod
    def validate(source: str, *, target_id: str = "") -> ValidationReport:
        """Run syntax + public-surface-import pre-checks on generated source.

        Args:
            source: The generated ``molexp.workflow`` program text.
            target_id: The artifact id this source came from (for the report).

        Returns:
            A :class:`ValidationReport` with ``target_kind="workflow_source"``;
            ``passed`` is False if any error-severity violation is present.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return ValidationReport.from_violations(
                target_kind="workflow_source",
                target_id=target_id,
                violations=[
                    ValidationViolation(
                        code="syntax_error",
                        message=f"generated source failed to parse: {exc!r}",
                        severity="error",
                    )
                ],
            )

        violations: list[ValidationViolation] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_private_workflow(alias.name):
                        violations.append(
                            ValidationViolation(
                                code="private_import",
                                message=f"generated source imports private module {alias.name!r}",
                                severity="error",
                            )
                        )
            elif isinstance(node, ast.ImportFrom) and _is_private_workflow(node.module):
                violations.append(
                    ValidationViolation(
                        code="private_import",
                        message=f"generated source imports from private module {node.module!r}",
                        severity="error",
                    )
                )

        return ValidationReport.from_violations(
            target_kind="workflow_source",
            target_id=target_id,
            violations=violations,
        )
