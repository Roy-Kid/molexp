"""Workflow layering invariant guard (rectification spec — Phase 0 / P0-05).

After rectification, ``src/molexp/workflow/`` may import from
``molexp.workspace.*`` (workspace is the storage primitive workflow
sits on top of) but must NOT import from any other upstream-or-sibling
layer:

- ``molexp.agent`` (agent depends on workflow, not the other way)
- ``molexp.plugins`` (optional capabilities)
- ``molexp.server``, ``molexp.cli``, ``molexp.sweep`` (application shell)

Additionally, ``pydantic_graph`` must not be imported anywhere —
molexp dropped the dependency; the engine under ``workflow/_engine/``
is molexp-owned (see also ``test_engine_boundary.py`` for the
src/-wide scan).

History: until 2026-05-09 this guard *also* forbade
``molexp.workspace`` imports. The rectification spec inverts that
direction; workflow now uses workspace for caching + persistence.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[2] / "src" / "molexp" / "workflow"

FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "molexp.agent",
    "molexp.plugins",
    "molexp.server",
    "molexp.cli",
    "molexp.services",
    "molexp.sweep",
)


def _iter_workflow_py_files() -> list[Path]:
    return [p for p in WORKFLOW_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


def _imports_of(prefix: str, root: Path) -> list[tuple[Path, int, str]]:
    hits: list[tuple[Path, int, str]] = []
    for py in root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == prefix or alias.name.startswith(prefix + "."):
                        hits.append((py, node.lineno, alias.name))
                        break
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                if module and (module == prefix or module.startswith(prefix + ".")):
                    hits.append((py, node.lineno, module))
    return hits


def _format(hits: list[tuple[Path, int, str]]) -> list[str]:
    return [
        f"{path.relative_to(WORKFLOW_ROOT)}:{lineno}: {module}" for path, lineno, module in hits
    ]


def test_workflow_forbids_upstream_and_application_layers() -> None:
    offenders: dict[str, list[str]] = {}
    for prefix in FORBIDDEN_PREFIXES:
        hits = _imports_of(prefix, WORKFLOW_ROOT)
        if hits:
            offenders[prefix] = _format(hits)
    assert not offenders, (
        "molexp.workflow must not import upstream / sibling layers.\n"
        "Allowed downward: molexp.workspace.*, molexp._typing, molexp.profile.\n"
        "Offenders:\n  "
        + "\n  ".join(f"[{prefix}] {hit}" for prefix, lines in offenders.items() for hit in lines)
    )


def test_no_pydantic_graph_imports_in_workflow() -> None:
    """``pydantic_graph`` may not appear anywhere under ``workflow/`` —
    the dependency was removed; the engine is molexp-owned."""
    hits = _imports_of("pydantic_graph", WORKFLOW_ROOT)
    bad = [f"{path.relative_to(WORKFLOW_ROOT)}:{lineno}: {module}" for path, lineno, module in hits]
    assert not bad, "pydantic_graph imports under workflow/:\n  " + "\n  ".join(bad)


def test_compiled_graph_is_layer_private() -> None:
    """No layer above ``workflow`` may read ``CompiledWorkflow.graph``.

    ``.graph`` holds a layer-private ``LoweredGraph`` (live task callables);
    only the workflow runtime reads it. Layers above workflow must use the
    public codec / introspection surface, never ``compiled.graph``. This
    guards the architect's N1 note on the build+compile merge (spec
    workflow-refactor-02).
    """
    src_root = Path(__file__).resolve().parents[2] / "src" / "molexp"
    upper_layers = ("server", "cli", "harness", "agent", "sweep", "plugins")
    offenders: list[str] = []
    for layer in upper_layers:
        layer_root = src_root / layer
        if not layer_root.exists():
            continue
        for py in layer_root.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r"\.graph\b", line) and not line.lstrip().startswith("#"):
                    offenders.append(f"{py.relative_to(src_root)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "Layers above 'workflow' must not read the layer-private "
        "CompiledWorkflow.graph; use the public codec/introspection surface.\n  "
        + "\n  ".join(offenders)
    )


def test_workflow_imports_from_workspace_are_allowed() -> None:
    """Sanity guard: the inversion is *expected*, not banned.

    After rectification, workflow imports workspace for caching +
    persistence backing. This test does not require any specific
    workspace import to exist (workflow may legitimately have zero on
    a transient commit), but documents that none of the workspace
    prefixes appear in ``FORBIDDEN_PREFIXES`` above.
    """
    assert "molexp.workspace" not in FORBIDDEN_PREFIXES
