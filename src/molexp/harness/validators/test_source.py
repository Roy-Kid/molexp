"""``validate_test_source`` — pure structural pre-checks for generated pytest source.

Side-effect-free checks; the source is **never executed** (no ``exec``, no
pytest run — a module-level ``raise`` in the source cannot fire here):

1. **Syntax** — ``ast.parse``; a ``SyntaxError`` becomes a violation.
2. **Public-surface imports only** — an AST walk rejects any import of a
   private ``molexp.workflow`` submodule (anything under
   ``molexp.workflow._...``).
3. **At least one test** — a module-level ``def test_*`` / ``async def
   test_*`` function must exist, or pytest would collect nothing.
4. **Byte-compile** — ``compile(source, "<test_source>", "exec")``; catches
   what parsing alone cannot (e.g. ``break`` outside a loop).

Returns a :class:`ValidationReport` (``target_kind="test_source"``) and
**never raises** — malformed input yields a failing report, not an exception.
This is the gate :class:`ValidateTestSource` runs; actually executing the
tests is :class:`ExecuteTests`'s job, through a harness executor.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable

from molexp.harness.schemas.validation import ValidationReport, ValidationViolation

__all__ = ["TestSourceValidator"]

_PRIVATE_PREFIX = "molexp.workflow._"


def _is_private_workflow(module: str | None) -> bool:
    """True if ``module`` names a private ``molexp.workflow`` submodule."""
    if not module:
        return False
    return module == "molexp.workflow._" or module.startswith(_PRIVATE_PREFIX)


def _test_function_names(tree: ast.Module) -> list[str]:
    """Names of the module's top-level ``test_*`` (async) functions."""
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]


def _normalize_task_id(task_id: str) -> str:
    """Map a task id to the identifier-safe token a test name must contain."""
    return re.sub(r"\W", "_", task_id)


class TestSourceValidator:
    """Pure structural pre-checks for generated pytest source."""

    @staticmethod
    def validate(
        source: str,
        *,
        target_id: str = "",
        required_task_ids: Iterable[str] | None = None,
    ) -> ValidationReport:
        """Run syntax + import + test-presence + byte-compile pre-checks.

        Args:
            source: The generated pytest program text.
            target_id: The artifact id this source came from (for the report).
            required_task_ids: When given, every task id must be covered by a
                ``test_*`` function whose name contains the id's identifier-safe
                token (``test_<task_id>`` or any ``test_*`` containing it); a
                missing one yields a ``missing_task_test`` error. When ``None``
                the legacy "at least one test" check stands alone.

        Returns:
            A :class:`ValidationReport` with ``target_kind="test_source"``;
            ``passed`` is False if any error-severity violation is present.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return ValidationReport.from_violations(
                target_kind="test_source",
                target_id=target_id,
                violations=[
                    ValidationViolation(
                        code="syntax_error",
                        message=f"generated test source failed to parse: {exc!r}",
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
                                message=f"generated test source imports private module {alias.name!r}",
                                severity="error",
                            )
                        )
            elif isinstance(node, ast.ImportFrom) and _is_private_workflow(node.module):
                violations.append(
                    ValidationViolation(
                        code="private_import",
                        message=f"generated test source imports from private module {node.module!r}",
                        severity="error",
                    )
                )

        test_names = _test_function_names(tree)
        if not test_names:
            violations.append(
                ValidationViolation(
                    code="no_test_functions",
                    message="generated test source defines no module-level test_* function",
                    severity="error",
                )
            )

        for task_id in required_task_ids or ():
            token = _normalize_task_id(task_id)
            if not any(token in name for name in test_names):
                violations.append(
                    ValidationViolation(
                        code="missing_task_test",
                        message=f"generated test source has no test_* function covering "
                        f"task {task_id!r} (expected a test name containing {token!r})",
                        severity="error",
                    )
                )

        try:
            compile(source, "<test_source>", "exec")
        except (SyntaxError, ValueError) as exc:
            violations.append(
                ValidationViolation(
                    code="compile_error",
                    message=f"generated test source failed to byte-compile: {exc!r}",
                    severity="error",
                )
            )

        return ValidationReport.from_violations(
            target_kind="test_source",
            target_id=target_id,
            violations=violations,
        )
