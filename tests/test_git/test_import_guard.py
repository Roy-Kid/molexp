"""Layer-0 firewall for the ``molexp.git`` cross-layer primitive.

``molexp.git`` is a bottom cross-layer primitive (pure ``git`` subprocess);
its source must not import any molexp layer (``workspace`` / ``workflow`` /
``agent`` / ``harness`` / ``server`` / ``cli`` / ``plugins`` / ``sweep``).

This is an **AST static scan** of the package source — not a runtime
``sys.modules`` probe. A runtime probe is unsatisfiable: importing any
``molexp.X`` submodule first runs the eager ``molexp/__init__.py``, which
loads workspace. Mirrors ``tests/test_knowledge/test_import_guard.py``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

GIT_ROOT = Path(__file__).resolve().parents[2] / "src" / "molexp" / "git"

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
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _offenders(source: str) -> list[str]:
    return [m for m in _imported_modules(source) if m.startswith(FORBIDDEN_PREFIXES)]


def test_git_source_imports_no_upstream_layer() -> None:
    hits: list[tuple[str, str]] = []
    for py in sorted(GIT_ROOT.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        for mod in _offenders(py.read_text(encoding="utf-8")):
            hits.append((py.name, mod))
    assert hits == [], f"forbidden imports in molexp.git: {hits}"


def test_scanner_detects_a_planted_violation() -> None:
    # Negative control: the scanner must catch a forbidden import.
    assert _offenders("from molexp.workspace.run import Run") == ["molexp.workspace.run"]
    assert _offenders("import molexp.harness") == ["molexp.harness"]


def test_import_git_objects_stays_light() -> None:
    # Importing the object framing must not transitively pull the heavy
    # agent / workflow engines (pydantic_ai / pydantic_graph) into the
    # interpreter. Run in a clean subprocess for an honest sys.modules.
    code = (
        "import sys, molexp.git.objects\n"
        "assert 'pydantic_ai' not in sys.modules, 'pydantic_ai leaked'\n"
        "assert 'pydantic_graph' not in sys.modules, 'pydantic_graph leaked'\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
