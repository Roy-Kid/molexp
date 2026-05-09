"""Workflow layering invariant guard (rectification spec — Phase 0 / P0-05).

After rectification, ``src/molexp/workflow/`` may import from
``molexp.workspace.*`` (workspace is the storage primitive workflow
sits on top of) but must NOT import from any other upstream-or-sibling
layer:

- ``molexp.agent`` (agent depends on workflow, not the other way)
- ``molexp.plugins`` (optional capabilities)
- ``molexp.server``, ``molexp.cli``, ``molexp.sweep`` (application shell)

Additionally, ``pydantic_graph`` must only be imported from
``workflow/_pydantic_graph/`` — the rest of the workflow layer is
pg-agnostic and reachable without the dependency.

History: until 2026-05-09 this guard *also* forbade
``molexp.workspace`` imports. The rectification spec inverts that
direction; workflow now uses workspace for caching + persistence.
"""

from __future__ import annotations

import ast
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[2] / "src" / "molexp" / "workflow"

FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "molexp.agent",
    "molexp.plugins",
    "molexp.server",
    "molexp.cli",
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
        "Allowed downward: molexp.workspace.*, molexp._typing, molexp.config.\n"
        "Offenders:\n  "
        + "\n  ".join(f"[{prefix}] {hit}" for prefix, lines in offenders.items() for hit in lines)
    )


def test_pydantic_graph_imports_confined_to_pydantic_graph_subtree() -> None:
    """``pydantic_graph`` may only appear under ``workflow/_pydantic_graph/``."""
    hits = _imports_of("pydantic_graph", WORKFLOW_ROOT)
    allowed = WORKFLOW_ROOT / "_pydantic_graph"
    bad = [
        f"{path.relative_to(WORKFLOW_ROOT)}:{lineno}: {module}"
        for path, lineno, module in hits
        if allowed not in path.parents
    ]
    assert not bad, "pydantic_graph imports outside workflow/_pydantic_graph/:\n  " + "\n  ".join(
        bad
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
