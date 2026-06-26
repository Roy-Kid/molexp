"""Workspace boundary firewall (rectification spec — Phase 0 / P0-04).

Workspace is the bottom of the dependency DAG. The only ``molexp.*``
imports allowed under ``src/molexp/workspace/`` are ``molexp._typing``,
``molexp.profile``, ``molexp.path`` (the cross-host POSIX path primitive),
and the root-level helpers (``mollog``, ``molcfg``).
Every other ``molexp.*`` subpackage — ``workflow``, ``agent``,
``plugins``, ``server``, ``cli``, ``sweep`` — is forbidden.

This is the mechanical enforcer of the rule documented in the
``§ Layer charters → molexp.workspace`` section of CLAUDE.md.

History: until 2026-05-09 this guard allowed ``molexp.workflow``
imports — that is the leak the rectification spec flushes out. The
expanded forbidden set below makes the leak in
``workspace/experiment.py:25–28`` show up as a RED test, which is the
entry ticket for Phase 1.
"""  # noqa: RUF002

from __future__ import annotations

import ast
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2] / "src" / "molexp" / "workspace"

FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "molexp.workflow",
    "molexp.agent",
    "molexp.plugins",
    "molexp.server",
    "molexp.cli",
    "molexp.sweep",
)


def _files_importing(prefix: str, root: Path) -> list[tuple[Path, int, str]]:
    """Return ``(path, lineno, module)`` triples for every match.

    Matches both ``import molexp.<prefix>`` and ``from molexp.<prefix>
    import …`` (and any subpackage). Returns the import line so failure
    messages can quote the offender directly.
    """
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
        f"{path.relative_to(WORKSPACE_ROOT)}:{lineno}: {module}" for path, lineno, module in hits
    ]


def test_workspace_forbids_all_upstream_layers() -> None:
    """No imports of any forbidden upstream prefix anywhere in workspace/."""
    offenders: dict[str, list[str]] = {}
    for prefix in FORBIDDEN_PREFIXES:
        hits = _files_importing(prefix, WORKSPACE_ROOT)
        if hits:
            offenders[prefix] = _format(hits)
    assert not offenders, (
        "molexp.workspace must not import any upstream layer. The "
        "dependency DAG flows downward: workspace ← workflow ← agent.\n"
        "Offenders:\n  "
        + "\n  ".join(f"[{prefix}] {hit}" for prefix, lines in offenders.items() for hit in lines)
    )


def test_guard_detects_violation_when_simulated(tmp_path: Path) -> None:
    """Negative test: the AST scan must catch a freshly-introduced import."""
    fake = tmp_path / "tainted.py"
    fake.write_text("from molexp.workflow.spec import WorkflowSpec\n")
    hits = _files_importing("molexp.workflow", tmp_path)
    assert any(p == fake for p, _, _ in hits), (
        "guard failed to detect a planted molexp.workflow import"
    )


def test_workspace_init_does_not_load_workflow_or_agent() -> None:
    """``import molexp.workspace`` must not pull workflow/agent into sys.modules.

    Equivalent invariant from CLAUDE.md: workspace is the leaf — touching
    it should never cascade an upstream module load.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import molexp.workspace  # noqa: F401\n"
        "assert 'molexp.workflow' not in sys.modules, "
        "    'molexp.workspace eagerly imported molexp.workflow'\n"
        "assert 'molexp.agent' not in sys.modules, "
        "    'molexp.workspace eagerly imported molexp.agent'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout


# ── Curation subpackage allow-set ────────────────────────────────────────────
#
# ``molexp.workspace.curation`` composes existing workspace primitives. As a
# member of the workspace layer it may import the cross-layer identity / path
# / profile primitives and any intra-``molexp.workspace`` module — nothing
# else. This guard is RED until the package exists (the dir-existence
# assertion documents that it is not yet implemented), then it pins the
# subpackage's import surface using the same AST walk as the deny-list above.

CURATION_ROOT = WORKSPACE_ROOT / "curation"

CURATION_ALLOWED_MOLEXP: frozenset[str] = frozenset(
    {"molexp._typing", "molexp.profile", "molexp.path", "molexp.ids"}
)


def _curation_import_allowed(module: str) -> bool:
    """Allow the four cross-layer primitives plus any intra-workspace module."""
    return module in CURATION_ALLOWED_MOLEXP or module.startswith("molexp.workspace")


def test_curation_subpackage_allowed_imports() -> None:
    """Every ``molexp.*`` import under ``workspace/curation/`` is in the allow-set.

    Reuses :func:`_files_importing` with the broad ``"molexp"`` prefix to
    enumerate every absolute ``molexp.*`` import in the subpackage, then
    rejects any that is neither a sanctioned cross-layer primitive nor an
    intra-``molexp.workspace`` import.
    """
    assert CURATION_ROOT.exists(), (
        "molexp.workspace.curation is not implemented yet "
        f"(expected package directory at {CURATION_ROOT})"
    )
    offenders = [
        f"{path.relative_to(WORKSPACE_ROOT)}:{lineno}: {module}"
        for path, lineno, module in _files_importing("molexp", CURATION_ROOT)
        if not _curation_import_allowed(module)
    ]
    assert not offenders, (
        "molexp.workspace.curation may import only the cross-layer primitives "
        f"{sorted(CURATION_ALLOWED_MOLEXP)} plus intra-workspace modules.\n"
        "Offenders:\n  " + "\n  ".join(offenders)
    )
