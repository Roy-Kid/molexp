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

from molexp.harness.schemas.validation import ValidationReport, ValidationViolation

__all__ = ["validate_test_source"]

_PRIVATE_PREFIX = "molexp.workflow._"


def _is_private_workflow(module: str | None) -> bool:
    """True if ``module`` names a private ``molexp.workflow`` submodule."""
    if not module:
        return False
    return module == "molexp.workflow._" or module.startswith(_PRIVATE_PREFIX)


def _has_test_function(tree: ast.Module) -> bool:
    """True if the module defines a top-level ``test_*`` (async) function."""
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
        for node in tree.body
    )


def validate_test_source(source: str, *, target_id: str = "") -> ValidationReport:
    """Run syntax + import + test-presence + byte-compile pre-checks.

    Args:
        source: The generated pytest program text.
        target_id: The artifact id this source came from (for the report).

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

    if not _has_test_function(tree):
        violations.append(
            ValidationViolation(
                code="no_test_functions",
                message="generated test source defines no module-level test_* function",
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
