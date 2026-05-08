"""Workspace core-dependency boundary firewall (spec ac-005 / ac-011).

The workspace layer is the canonical anchor of the molexp dependency
DAG: it may depend on ``molexp.workflow`` and on the molexp root-level
singletons (``mollog`` / ``molcfg`` / ``molexp.config``), and on nothing
else from the project. In particular it MUST NOT import:

- any ``molexp.agent`` subpackage (agent depends on workspace, not the
  other way around);
- any ``molexp.plugins`` subpackage (plugins are optional capabilities;
  the core workspace must not gate on any of them).

This test scans every ``.py`` file under ``src/molexp/workspace/`` with
the ``ast`` module and fails if either form of forbidden import is
present. It is the only mechanical enforcer of the rule documented in
the ``§ Workspace core-dependency boundary`` section of CLAUDE.md.
"""

from __future__ import annotations

import ast
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2] / "src" / "molexp" / "workspace"

FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "molexp.agent",
    "molexp.plugins",
)


def _files_importing(prefix: str, root: Path) -> list[Path]:
    """Return the list of ``.py`` files under *root* that import ``prefix``.

    Matches both ``import molexp.agent`` and ``from molexp.agent import …``,
    plus any subpackage (``molexp.agent.sessions`` etc.). Returns an
    empty list when the directory has no matches.
    """
    hits: list[Path] = []
    for py in root.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == prefix or alias.name.startswith(prefix + "."):
                        hits.append(py)
                        break
                else:
                    continue
                break
            if isinstance(node, ast.ImportFrom):
                module = node.module
                if module and (module == prefix or module.startswith(prefix + ".")):
                    hits.append(py)
                    break
    return hits


def test_workspace_does_not_import_molexp_agent() -> None:
    hits = _files_importing("molexp.agent", WORKSPACE_ROOT)
    bad = [str(p.relative_to(WORKSPACE_ROOT)) for p in hits]
    assert not bad, (
        "molexp.workspace must not import molexp.agent (agent depends on "
        f"workspace, not vice versa). Offending files: {bad}"
    )


def test_workspace_does_not_import_molexp_plugins() -> None:
    hits = _files_importing("molexp.plugins", WORKSPACE_ROOT)
    bad = [str(p.relative_to(WORKSPACE_ROOT)) for p in hits]
    assert not bad, (
        "molexp.workspace must not import any specific molexp.plugins "
        f"subpackage (plugins are optional capabilities). Offending files: {bad}"
    )


def test_guard_detects_violation_when_simulated(tmp_path: Path) -> None:
    """Negative test: the AST scan must catch a freshly-introduced import.

    Sanity-check that the guard is doing real work and would fail if
    someone re-introduced ``from molexp.agent import …`` under
    ``src/molexp/workspace/``. We simulate this on a temp-dir clone of
    a single file so the real workspace tree stays clean.
    """
    fake = tmp_path / "tainted.py"
    fake.write_text("from molexp.agent.sessions import SessionStore\n")
    hits = _files_importing("molexp.agent", tmp_path)
    assert fake in hits, "guard failed to detect a planted molexp.agent import"
