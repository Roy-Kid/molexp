"""Agent boundary firewall (rectification spec — Phase 0 / P0-06).

Agent is the top of the three-layer DAG. It may import from both
``molexp.workspace.*`` and ``molexp.workflow.*`` (those are downstream).
It MUST NOT import from sibling application layers:

- ``molexp.plugins`` (the agent stays a library, never reaches the
  application's plugin shell)
- ``molexp.server``, ``molexp.cli``, ``molexp.sweep``

Two pydantic-SDK invariants also live here:

1. ``pydantic_ai`` may only be imported from
   ``src/molexp/agent/_pydanticai/``.
2. ``pydantic_graph`` must NOT be imported anywhere under
   ``src/molexp/agent/`` — pg lives exclusively under
   ``src/molexp/workflow/_pydantic_graph/``. PlanMode drives multi-step
   workflows through the public ``molexp.workflow`` API.
3. Plain ``import molexp.agent`` does not eagerly load ``pydantic_ai``
   — the SDK is loaded lazily when ``PydanticAIRouter`` is
   constructed (on first ``AgentRunner.run``).
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
    "molexp.sweep",
)


def _files_importing(module: str, root: Path) -> list[tuple[Path, int, str]]:
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
    """No imports of plugins / server / cli / sweep anywhere in agent/."""
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


def test_pydantic_graph_never_imported_in_agent() -> None:
    """``pydantic_graph`` is a workflow-layer concern; ``agent/`` never imports it.

    PlanMode drives multi-step workflows through the public
    ``molexp.workflow`` API, which is the sole sanctioned pg site.
    """
    hits = _files_importing("pydantic_graph", AGENT_ROOT)
    bad = _format(hits)
    assert not bad, (
        "pydantic_graph imported inside agent/. The workflow layer is "
        "the only sanctioned pg site:\n  " + "\n  ".join(bad)
    )


def test_importing_molexp_agent_does_not_load_pydantic_ai() -> None:
    """``import molexp.agent`` must not eagerly import pydantic_ai.

    The router is heavy and the SDK takes time to load; agent's
    runner constructs it lazily on first ``.run()``. We only assert
    pydantic_ai laziness here — pydantic_graph may legitimately load
    transitively through ``molexp.workflow`` (PlanMode wiring), and
    its confinement to ``workflow/_pydantic_graph/`` is enforced
    separately by ``tests/test_workflow/test_import_guard.py``.
    """
    code = (
        "import sys\n"
        "import molexp.agent  # noqa: F401\n"
        "assert 'pydantic_ai' not in sys.modules, (\n"
        "    f'pydantic_ai was eagerly loaded; '\n"
        "    f'check that no module under agent/ imports it at top level '\n"
        "    f'outside agent/_pydanticai/.'\n"
        ")\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout


def test_importing_molexp_agent_router_does_not_load_pydantic_ai() -> None:
    """``import molexp.agent.router`` (the protocol module) must also stay lazy.

    The protocol file deliberately defers all SDK imports to the concrete
    :class:`~molexp.agent._pydanticai.router.PydanticAIRouter` so test
    fakes can implement the protocol without paying the SDK load cost.
    """
    code = (
        "import sys\n"
        "import molexp.agent.router  # noqa: F401\n"
        "assert 'pydantic_ai' not in sys.modules, (\n"
        "    f'pydantic_ai was eagerly loaded by molexp.agent.router; '\n"
        "    f'the protocol module must not import the SDK.'\n"
        ")\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout


def test_importing_mcp_defaults_stays_lazy() -> None:
    """ac-014 — ``import molexp.agent.mcp.defaults`` stays SDK-free.

    The defaults module declares the platform's seeded MCP servers and
    the seeding helper; nothing in it should pull in ``pydantic_ai`` or
    ``pydantic_graph``. The seeding fires under
    :class:`~molexp.agent.mcp.store.McpStore` construction, but neither
    side path should require the SDKs.
    """
    code = (
        "import sys\n"
        "import molexp.agent.mcp.defaults  # noqa: F401\n"
        "assert 'pydantic_ai' not in sys.modules, (\n"
        "    f'pydantic_ai was eagerly loaded by molexp.agent.mcp.defaults'\n"
        ")\n"
        "assert 'pydantic_graph' not in sys.modules, (\n"
        "    f'pydantic_graph was eagerly loaded by molexp.agent.mcp.defaults'\n"
        ")\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout


def test_agent_workspace_and_workflow_imports_are_allowed() -> None:
    """Sanity guard: workspace + workflow are not in FORBIDDEN_PREFIXES.

    Agent is the top of the DAG; both downstream layers are reachable.
    """
    assert "molexp.workspace" not in FORBIDDEN_PREFIXES
    assert "molexp.workflow" not in FORBIDDEN_PREFIXES
