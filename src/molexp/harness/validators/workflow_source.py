"""``WorkflowSourceValidator`` — pure structural pre-checks for generated source.

Side-effect-free checks run *before* any compile/exec of LLM-generated code:

1. **Syntax** — ``ast.parse``; a ``SyntaxError`` becomes a violation.
2. **Public-surface imports only** — an AST walk rejects any import of a
   private ``molexp.workflow`` submodule (anything under
   ``molexp.workflow._...``); generated code must target the public surface.
3. **Entry/driver anti-patterns** — reject ``argparse`` CLIs, ``sys.path``
   mutation, and in-script ``execute_run`` drivers. FAIR experiment entries
   declare ``Workspace → project → experiment → run(...)`` and are driven by
   ``molexp run`` (see ``contracts/molexp_codegen.v1.yaml`` ``entry_script``).

Returns a :class:`PlanValidationReport` (``target_kind="workflow_source"``) and
**never raises** — malformed input yields a failing report, not an exception.
This is the gate :class:`ValidateWorkflowSource` runs before it ever compiles
or executes the source.
"""

from __future__ import annotations

import ast

from molexp.harness.schemas.validation import PlanValidationReport, ValidationViolation

__all__ = ["WorkflowSourceValidator"]

_PRIVATE_PREFIX = "molexp.workflow._"

_SYS_PATH_MUTATORS = frozenset({"insert", "append", "extend"})


def _is_private_workflow(module: str | None) -> bool:
    """True if ``module`` names a private ``molexp.workflow`` submodule."""
    if not module:
        return False
    return module == "molexp.workflow._" or module.startswith(_PRIVATE_PREFIX)


def _is_sys_path_attr(node: ast.AST) -> bool:
    """True for the AST of ``sys.path`` (``Name`` or ``Attribute`` form)."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "path"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _entry_driver_violations(tree: ast.Module) -> list[ValidationViolation]:
    """Flag argparse / sys.path / execute_run entry-driver anti-patterns."""
    violations: list[ValidationViolation] = []
    seen: set[str] = set()

    def _add(code: str, message: str) -> None:
        if code in seen:
            return
        seen.add(code)
        violations.append(ValidationViolation(code=code, message=message, severity="error"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "argparse" or alias.name.startswith("argparse."):
                    _add(
                        "argparse_cli",
                        "generated source must not import argparse — drive the "
                        "experiment with `molexp run` and bind knobs as task "
                        "params / molcfg profiles / --override",
                    )
                if alias.name == "sys":
                    # sys alone is fine (e.g. version checks); mutations checked below
                    pass
        elif isinstance(node, ast.ImportFrom):
            if node.module == "argparse" or (
                node.module is not None and node.module.startswith("argparse.")
            ):
                _add(
                    "argparse_cli",
                    "generated source must not import argparse — drive the "
                    "experiment with `molexp run` and bind knobs as task "
                    "params / molcfg profiles / --override",
                )
            if node.module == "molexp.workflow" or node.module == "molexp":
                for alias in node.names:
                    if alias.name in ("execute_run", "aexecute_run"):
                        _add(
                            "execute_run_in_entry",
                            "generated FAIR entry must not call execute_run — "
                            "declare Workspace.project().experiment().run(...) "
                            "and let `molexp run` drive execution "
                            "(execute_run is for library/notebook drivers only)",
                        )
        elif isinstance(node, ast.Call):
            func = node.func
            # sys.path.insert / append / extend
            if (
                isinstance(func, ast.Attribute)
                and func.attr in _SYS_PATH_MUTATORS
                and _is_sys_path_attr(func.value)
            ):
                _add(
                    "sys_path_insert",
                    "generated source must not mutate sys.path — install "
                    "science packages normally; `molexp run` already puts the "
                    "script directory on sys.path for sibling modules",
                )
            # execute_run(...) / aexecute_run(...) bare name
            if isinstance(func, ast.Name) and func.id in ("execute_run", "aexecute_run"):
                _add(
                    "execute_run_in_entry",
                    "generated FAIR entry must not call execute_run — "
                    "declare Workspace.project().experiment().run(...) "
                    "and let `molexp run` drive execution "
                    "(execute_run is for library/notebook drivers only)",
                )
            # molexp.execute_run(...) / me.execute_run(...)
            if isinstance(func, ast.Attribute) and func.attr in ("execute_run", "aexecute_run"):
                _add(
                    "execute_run_in_entry",
                    "generated FAIR entry must not call execute_run — "
                    "declare Workspace.project().experiment().run(...) "
                    "and let `molexp run` drive execution "
                    "(execute_run is for library/notebook drivers only)",
                )

    return violations


class WorkflowSourceValidator:
    @staticmethod
    def validate(source: str, *, target_id: str = "") -> PlanValidationReport:
        """Run syntax + public-surface + entry-driver pre-checks on source.

        Args:
            source: The generated ``molexp.workflow`` program text.
            target_id: The artifact id this source came from (for the report).

        Returns:
            A :class:`PlanValidationReport` with ``target_kind="workflow_source"``;
            ``passed`` is False if any error-severity violation is present.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return PlanValidationReport.from_violations(
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

        violations.extend(_entry_driver_violations(tree))

        return PlanValidationReport.from_violations(
            target_kind="workflow_source",
            target_id=target_id,
            violations=violations,
        )
