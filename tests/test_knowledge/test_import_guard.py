"""Layer firewall for the OKF ``molexp.knowledge`` bottom layer.

``molexp.knowledge`` sits at the bottom of the dependency DAG. Its source
must not import ``molexp.workspace`` (a sibling bottom layer) nor any
upstream layer (``workflow`` / ``agent`` / ``harness`` / ``server`` /
``cli`` / ``plugins`` / ``sweep``). Only stdlib, pydantic, pyyaml, and the
sanctioned cross-layer primitives (``molexp.atomicio`` / ``molexp.ids`` /
``molexp.path`` / ``mollog`` / ``molcfg``) are allowed.

This is an **AST static scan** of the package source — not a runtime
``sys.modules`` probe. A runtime probe is unsatisfiable: importing any
``molexp.X`` submodule first runs the eager ``molexp/__init__.py``, which
loads workspace. Mirrors ``tests/test_workspace/test_import_guard.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

KNOWLEDGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "molexp" / "knowledge"

FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "molexp.workspace",
    "molexp.workflow",
    "molexp.agent",
    "molexp.harness",
    "molexp.server",
    "molexp.cli",
    "molexp.plugins",
    "molexp.sweep",
)


def _imported_modules(source: str) -> list[str]:
    """Return every module name imported by *source* (import + from-import)."""
    names: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _offenders(source: str) -> list[str]:
    return [m for m in _imported_modules(source) if m.startswith(FORBIDDEN_PREFIXES)]


def test_knowledge_source_imports_no_workspace_or_upstream_layer() -> None:
    hits: list[tuple[str, str]] = []
    for py in sorted(KNOWLEDGE_ROOT.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        for mod in _offenders(py.read_text(encoding="utf-8")):
            hits.append((py.name, mod))
    assert hits == [], f"forbidden imports in molexp.knowledge: {hits}"


def test_scanner_detects_a_planted_violation() -> None:
    # Negative control: the scanner must catch a forbidden import.
    assert _offenders("from molexp.workspace.base import atomic_write_json") == [
        "molexp.workspace.base"
    ]
    assert _offenders("import molexp.workflow.compiler") == ["molexp.workflow.compiler"]
