"""Import-boundary firewall.

Three invariants:

1. ``pydantic_ai`` may only be imported from ``src/molexp/agent/_pydanticai/``.
2. ``pydantic_graph`` must NOT be imported anywhere under ``src/molexp/agent/`` —
   it lives exclusively under ``src/molexp/workflow/_pydantic_graph/``.
3. Plain ``import molexp.agent`` does not eagerly load ``pydantic_ai`` —
   the SDK is loaded lazily when ``PydanticAIHarness`` is constructed.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[2] / "src" / "molexp" / "agent"


def _files_importing(module: str, root: Path) -> list[Path]:
    hits: list[Path] = []
    for py in root.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name == module or a.name.startswith(module + ".") for a in node.names):
                    hits.append(py)
                    break
            elif isinstance(node, ast.ImportFrom):
                if node.module == module or (node.module and node.module.startswith(module + ".")):
                    hits.append(py)
                    break
    return hits


def test_pydantic_ai_imports_confined_to_pydanticai_subtree() -> None:
    hits = _files_importing("pydantic_ai", AGENT_ROOT)
    allowed = AGENT_ROOT / "_pydanticai"
    bad = [str(p.relative_to(AGENT_ROOT)) for p in hits if allowed not in p.parents]
    assert not bad, f"pydantic_ai imported outside agent/_pydanticai/: {bad}"


def test_pydantic_graph_never_imported_in_agent() -> None:
    """``pydantic_graph`` is a workflow-layer concern; ``agent/`` never imports it."""
    bad = [str(p.relative_to(AGENT_ROOT)) for p in _files_importing("pydantic_graph", AGENT_ROOT)]
    assert not bad, f"pydantic_graph imported inside agent/: {bad}"


def test_importing_molexp_agent_does_not_load_pydantic_ai() -> None:
    """``import molexp.agent`` must not eagerly import pydantic_ai."""
    code = (
        "import sys\n"
        "import molexp.agent  # noqa: F401\n"
        "assert 'pydantic_ai' not in sys.modules, sorted(sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
