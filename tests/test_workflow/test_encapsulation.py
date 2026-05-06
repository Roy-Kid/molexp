"""Encapsulation lint — pydantic_graph imports must stay inside the seam.

Spec 04 §1: ``from pydantic_graph`` / ``import pydantic_graph`` is allowed
only inside the engine package ``src/molexp/workflow/_pydantic_graph/``
plus the single base-class declaration in ``src/molexp/workflow/task.py``.
Every other file under ``src/molexp/`` must use molexp-named symbols
(``Task``, ``Actor``, ``Next``, ``End``, ``WorkflowSpec``, …) so users —
and intra-package callers — never see the underlying engine name.

The check is a grep walker rather than an AST parser: line-level matching
catches the failure mode that matters (a stray ``from pydantic_graph
import …`` slipping into a public-facing module) without false-positives
on docstring prose mentioning the package by name (we trim docstring
hits explicitly).
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src" / "molexp"

ALLOWED_PREFIXES: tuple[Path, ...] = (SRC_ROOT / "workflow" / "_pydantic_graph",)
ALLOWED_FILES: frozenset[Path] = frozenset(
    {
        SRC_ROOT / "workflow" / "task.py",
    }
)

IMPORT_PATTERN = re.compile(
    r"^\s*(?:from\s+pydantic_graph(?:\.[\w.]+)?\s+import\b|import\s+pydantic_graph\b)"
)


def _executable_lines(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, source) pairs with comments and string literals stripped.

    ``tokenize`` lets us drop docstring hits ("uses pydantic_graph for…")
    so the lint only flags real ``import``-statement violations.
    """
    source = path.read_text(encoding="utf-8")
    keep: dict[int, str] = {}
    try:
        tokens = list(tokenize.tokenize(io.BytesIO(source.encode("utf-8")).readline))
    except tokenize.TokenizeError:
        return [(idx, line) for idx, line in enumerate(source.splitlines(), start=1)]
    for tok in tokens:
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
            continue
        if tok.type == tokenize.ENCODING or tok.type == tokenize.ENDMARKER:
            continue
        lineno = tok.start[0]
        keep.setdefault(lineno, source.splitlines()[lineno - 1])
    return sorted(keep.items())


def _is_allowed(path: Path) -> bool:
    if path in ALLOWED_FILES:
        return True
    for prefix in ALLOWED_PREFIXES:
        try:
            path.relative_to(prefix)
            return True
        except ValueError:
            continue
    return False


def _python_files() -> list[Path]:
    return [
        p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts and "dist" not in p.parts
    ]


def test_pydantic_graph_imports_confined_to_seam() -> None:
    """ac-001 — only `_pydantic_graph/*.py` and `workflow/task.py` may import pydantic_graph."""
    violations: list[str] = []
    for path in _python_files():
        if _is_allowed(path):
            continue
        for lineno, line in _executable_lines(path):
            if IMPORT_PATTERN.match(line):
                rel = path.relative_to(PROJECT_ROOT)
                violations.append(f"{rel}:{lineno}: {line.strip()}")

    assert not violations, (
        "pydantic_graph imports leaked outside the encapsulation seam.\n"
        "Allowed locations:\n"
        "  - src/molexp/workflow/_pydantic_graph/*.py\n"
        "  - src/molexp/workflow/task.py (the BaseNode inheritance seam)\n\n"
        "Violations:\n  " + "\n  ".join(violations)
    )


# Match the bare engine name `pydantic_graph` but NOT molexp's internal
# `_pydantic_graph` package path (which IS the encapsulation seam).
_BARE_PYDANTIC_GRAPH = re.compile(r"(?<![._a-zA-Z0-9])pydantic_graph")


def test_workflow_init_does_not_reexport_pydantic_graph_names() -> None:
    """The public surface uses molexp names only; ``End`` / ``BaseNode`` etc.
    must not be re-exported from ``pydantic_graph`` through
    ``molexp.workflow.__init__``. Imports from the internal
    ``._pydantic_graph`` engine package are allowed — that is the seam.
    """
    init_path = SRC_ROOT / "workflow" / "__init__.py"
    source = init_path.read_text(encoding="utf-8")
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        assert not _BARE_PYDANTIC_GRAPH.search(line), (
            f"src/molexp/workflow/__init__.py:{lineno} mentions the bare "
            f"`pydantic_graph` engine name; the public surface must use "
            f"molexp-named symbols only.\n  {line.strip()}"
        )
