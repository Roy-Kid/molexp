"""Regression net: ``auto_grant_approver`` is never a default (vision-loop-01).

The spec's wall: ``auto_grant_approver`` survives only as an **explicit**
injection (tests, ``--yes``, deliberately unattended runs). No module under
``src/molexp`` may substitute it for a ``None`` approver — the historical
defect patterns were::

    approve if approve is not None else auto_grant_approver  # IfExp orelse
    approve or auto_grant_approver  # BoolOp fallback

plus the equivalent function-parameter default (``approve=auto_grant_approver``).
An AST scan (not a text grep) catches all spellings while still allowing the
one *explicit per-request* use (``auto_grant_approver if request.approve else
…`` — the caller decided, nothing silent) and the definition site itself.
"""

from __future__ import annotations

import ast
from pathlib import Path

import molexp

_TARGET = "auto_grant_approver"


def _is_target(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == _TARGET
    if isinstance(node, ast.Attribute):
        return node.attr == _TARGET
    return False


def _violations_in(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        # `something if cond else auto_grant_approver` — the silent default.
        # (The explicit `auto_grant_approver if cond else reject` keeps the
        # target in the BODY position and is allowed.)
        if isinstance(node, ast.IfExp) and _is_target(node.orelse):
            found.append(f"{path}:{node.lineno} — IfExp default `else {_TARGET}`")
        # `approve or auto_grant_approver` — the same default, spelled with or.
        if (
            isinstance(node, ast.BoolOp)
            and isinstance(node.op, ast.Or)
            and any(_is_target(value) for value in node.values[1:])
        ):
            found.append(f"{path}:{node.lineno} — BoolOp fallback `or {_TARGET}`")
        # `def f(..., approve=auto_grant_approver)` — a signature-level default.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = list(node.args.defaults) + [
                default for default in node.args.kw_defaults if default is not None
            ]
            if any(_is_target(default) for default in defaults):
                found.append(f"{path}:{node.lineno} — parameter default `={_TARGET}`")
    return found


def test_no_module_substitutes_auto_grant_for_a_none_approver() -> None:
    src_root = Path(molexp.__file__).resolve().parent
    violations: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        if "dist" in path.relative_to(src_root).parts:
            continue  # bundled UI assets, not molexp source
        violations.extend(_violations_in(path))

    assert violations == [], (
        "auto_grant_approver must never be a default — it is an explicit "
        "injection only (spec vision-loop-01). Offending sites:\n  " + "\n  ".join(violations)
    )
