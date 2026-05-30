"""Boundary gate: ``server.schemas`` must never import ``server.agent_runtime``.

The agent runtime holds live objects (``AgentRunner`` / ``Session`` / asyncio
tasks / sinks). If a wire schema module imported the runtime — or worse, named
a runtime type in a ``response_model`` — those objects would leak into the
OpenAPI schema and break the UI codegen. The schema layer stays import-blind to
the runtime; routes translate runtime → frozen ``*Response`` explicitly.
"""

from __future__ import annotations

import ast
from pathlib import Path

SCHEMAS_ROOT = Path(__file__).resolve().parents[2] / "src" / "molexp" / "server" / "schemas"
FORBIDDEN = "molexp.server.agent_runtime"


def _imports(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
    return names


def test_schemas_never_import_agent_runtime() -> None:
    offenders: list[str] = []
    for path in SCHEMAS_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module in _imports(tree):
            if module == FORBIDDEN or module.startswith(FORBIDDEN + "."):
                offenders.append(f"{path.name}: imports {module}")
    assert not offenders, "server.schemas must not import server.agent_runtime:\n" + "\n".join(
        offenders
    )
