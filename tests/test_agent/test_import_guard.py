"""Agent boundary firewall (rectification spec — Phase 0 / P0-06).

Agent sits above the bottom storage layers. Its sanctioned downstream edges
are ``molexp.workspace.*`` and ``molexp.knowledge.*`` (Agent/AgentSession are
Concepts after the OKF rehome). It MUST NOT import ``molexp.workflow`` /
``molexp.harness`` (sibling/upstream) nor the application layers
(``plugins`` / ``server`` / ``cli`` / ``services`` / ``sweep``). Two SDK
invariants live here too:

1. ``pydantic_ai`` may only be imported from ``src/molexp/agent/_pydanticai/``.
2. Importing the agent surface stays SDK-lazy — ``import molexp.agent`` (and
   ``molexp.agent.loops``) loads neither ``pydantic_ai`` nor the ``mcp`` SDK
   until a router is actually built.

(``pydantic_graph`` is banned across all of ``src/`` — owned by the full-src
AST scan in ``tests/test_workflow/test_engine_boundary.py`` — so it is not
re-scanned here.)
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[2] / "src" / "molexp" / "agent"

FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "molexp.plugins",
    "molexp.server",
    "molexp.cli",
    "molexp.services",  # application-service layer sits above agent
    "molexp.sweep",
    "molexp.workflow",  # spec 03b: agent stopped being the orchestrator
    "molexp.harness",  # spec 03b: agent sits below harness in the DAG
)


def _is_type_checking_block(node: ast.AST) -> bool:
    """True if ``node`` is an ``if TYPE_CHECKING:`` (or ``typing.TYPE_CHECKING``) gate."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
    )


def _runtime_imports(tree: ast.AST) -> list[ast.Import | ast.ImportFrom]:
    """Walk ``tree`` collecting only the imports executed at runtime.

    Imports inside ``if TYPE_CHECKING:`` (or ``if typing.TYPE_CHECKING:``)
    are skipped — they exist only for type checkers and are never loaded,
    so they do not violate the runtime-firewall invariant.
    """
    collected: list[ast.Import | ast.ImportFrom] = []

    def _walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if _is_type_checking_block(child):
                continue
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                collected.append(child)
            _walk(child)

    _walk(tree)
    return collected


def _files_importing(module: str, root: Path) -> list[tuple[Path, int, str]]:
    hits: list[tuple[Path, int, str]] = []
    for py in root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in _runtime_imports(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == module or alias.name.startswith(module + "."):
                        hits.append((py, node.lineno, alias.name))
                        break
            elif isinstance(node, ast.ImportFrom):
                m = node.module
                if m and (m == module or m.startswith(module + ".")):
                    hits.append((py, node.lineno, m))
    return hits


def _format(hits: list[tuple[Path, int, str]]) -> list[str]:
    return [f"{path.relative_to(AGENT_ROOT)}:{lineno}: {module}" for path, lineno, module in hits]


def test_agent_forbids_application_layers() -> None:
    """No imports of plugins / server / cli / services / sweep / workflow / harness."""
    offenders: dict[str, list[str]] = {}
    for prefix in FORBIDDEN_PREFIXES:
        hits = _files_importing(prefix, AGENT_ROOT)
        if hits:
            offenders[prefix] = _format(hits)
    assert not offenders, (
        "molexp.agent must not import the application shell. The "
        "agent is a library that stays beneath the application layer.\n"
        "Offenders:\n  "
        + "\n  ".join(f"[{prefix}] {hit}" for prefix, lines in offenders.items() for hit in lines)
    )


def test_pydantic_ai_imports_confined_to_pydanticai_subtree() -> None:
    hits = _files_importing("pydantic_ai", AGENT_ROOT)
    allowed = AGENT_ROOT / "_pydanticai"
    bad = [
        f"{path.relative_to(AGENT_ROOT)}:{lineno}: {module}"
        for path, lineno, module in hits
        if allowed not in path.parents
    ]
    assert not bad, "pydantic_ai imports outside agent/_pydanticai/:\n  " + "\n  ".join(bad)


def test_importing_loops_does_not_load_mcp_clients() -> None:
    """Sentinel — importing the public loop surface stays MCP-client free.

    Plain ``import molexp.agent.loops`` must not pull ``pydantic_ai.mcp`` /
    the ``mcp`` SDK into ``sys.modules``; MCP wiring stays lazy until a
    router is built.
    """
    code = (
        "import sys\n"
        "import molexp.agent.loops  # noqa: F401\n"
        "for forbidden in ('pydantic_ai', 'pydantic_ai.mcp', 'mcp', 'mcp.client'):\n"
        "    assert forbidden not in sys.modules, (\n"
        "        f'{forbidden} eagerly loaded by molexp.agent.loops; '\n"
        "        'loop imports should stay SDK-free until run().'\n"
        "    )\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout


def test_pydanticai_router_public_reexport_is_lazy() -> None:
    """``from molexp.agent import PydanticAIRouter`` works and stays lazy.

    The public spelling resolves through a module-level ``__getattr__``:
    ``import molexp.agent`` alone must not load ``pydantic_ai``; touching the
    ``PydanticAIRouter`` attribute loads the SDK and returns the same class
    that lives under the ``_pydanticai/`` firewall.
    """
    code = (
        "import sys\n"
        "import molexp.agent\n"
        "assert 'pydantic_ai' not in sys.modules, (\n"
        "    'import molexp.agent must stay pydantic_ai-free even with the '\n"
        "    'PydanticAIRouter re-export declared'\n"
        ")\n"
        "from molexp.agent import PydanticAIRouter\n"
        "from molexp.agent._pydanticai.router import PydanticAIRouter as Private\n"
        "assert PydanticAIRouter is Private, 'public re-export must be the same class'\n"
        "assert 'pydantic_ai' in sys.modules, 'attribute access should have loaded the SDK'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
