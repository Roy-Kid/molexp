"""Encapsulation lint — zero pydantic_graph imports anywhere under src/.

The dependency was removed: the engine (``workflow/_engine/``, formerly
``_pydantic_graph/``) is molexp-owned and ``End`` lives in
``molexp.workflow.types``. There is no seam any more — NO file under
``src/molexp/`` may ``import pydantic_graph``, and every module must use
molexp-named symbols (``Task``, ``Actor``, ``Next``, ``End``,
``WorkflowSpec``, …).

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


def _python_files() -> list[Path]:
    return [
        p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts and "dist" not in p.parts
    ]


def test_no_pydantic_graph_imports_anywhere_in_src() -> None:
    """Zero-dependency invariant — NO file under src/molexp/ may import
    pydantic_graph (there is no allowed seam; the engine is molexp-owned)."""
    violations: list[str] = []
    for path in _python_files():
        for lineno, line in _executable_lines(path):
            if IMPORT_PATTERN.match(line):
                rel = path.relative_to(PROJECT_ROOT)
                violations.append(f"{rel}:{lineno}: {line.strip()}")

    assert not violations, (
        "pydantic_graph imported under src/ — molexp dropped the dependency "
        "entirely; the workflow engine (workflow/_engine/) is molexp-owned "
        "and End lives in molexp.workflow.types.\n"
        "Violations:\n  " + "\n  ".join(violations)
    )


# Match the bare `pydantic_graph` name but NOT molexp's internal
# `_engine` package path (nor prose mentioning `_pydantic_graph` history).
_BARE_PYDANTIC_GRAPH = re.compile(r"(?<![._a-zA-Z0-9])pydantic_graph")


def test_workflow_init_does_not_reexport_pydantic_graph_names() -> None:
    """The public surface uses molexp names only; the retired
    ``pydantic_graph`` name must not appear in ``molexp.workflow.__init__``
    at all. Imports from the internal ``._engine`` package are fine —
    that package is molexp-owned.
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
