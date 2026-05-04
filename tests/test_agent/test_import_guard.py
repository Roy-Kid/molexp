"""Phase 0 contract: ``molexp.agent`` is stdlib-only.

The harness must not pull in ``pydantic_ai``, MCP SDKs, HTTP clients,
or any provider SDK. We enforce two ways:

1. **Static AST sweep** — every ``.py`` file under ``src/molexp/agent/``
   is parsed and every ``import`` / ``from ... import`` is checked
   against the forbidden roots. This catches accidental imports even
   when the optional dependency is installed locally.
2. **Runtime sentinel** — after a fresh ``importlib`` import of
   ``molexp.agent``, no forbidden module name appears in
   ``sys.modules`` (apart from anything an unrelated test pre-loaded
   into the interpreter, which we filter against a sentinel snapshot
   taken at process start).

If a future plugin lives under ``molexp.agent`` (it should not), this
test must be tightened to walk the import graph as well.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

import molexp.agent

FORBIDDEN_ROOTS = {
    "pydantic_ai",
    "openai",
    "anthropic",
    "google.genai",
    "mcp",  # MCP Python SDK
    "httpx",
    "aiohttp",
    "requests",
}

AGENT_PACKAGE_PATH = Path(molexp.agent.__file__).parent


def _iter_py_files() -> list[Path]:
    return sorted(AGENT_PACKAGE_PATH.rglob("*.py"))


def _import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
                roots.add(alias.name.rsplit(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
            roots.add(node.module)
    return roots


@pytest.mark.parametrize("source_file", _iter_py_files(), ids=lambda p: p.name)
def test_no_forbidden_static_imports(source_file: Path) -> None:
    """Every file under ``molexp.agent`` is free of forbidden imports."""

    tree = ast.parse(source_file.read_text())
    roots = _import_roots(tree)
    leaks = roots & FORBIDDEN_ROOTS
    assert not leaks, (
        f"{source_file.relative_to(AGENT_PACKAGE_PATH.parent.parent.parent)} "
        f"imports forbidden modules: {sorted(leaks)}"
    )


def test_runtime_import_does_not_load_forbidden_modules() -> None:
    """``molexp.agent`` does not pull in plugin SDKs *itself*.

    Runs in a subprocess so the eviction-and-reimport does not break
    ``isinstance`` correlations in other tests in this session. The
    parent ``molexp`` package may load extra modules (workspace,
    plugins, etc.) — we only care about what ``molexp.agent``
    *adds* on top.
    """

    probe = (
        "import json, sys\n"
        "import molexp  # noqa: F401\n"
        "before = set(sys.modules)\n"
        "import molexp.agent  # noqa: F401\n"
        "added = sorted(set(sys.modules) - before)\n"
        "print(json.dumps(added))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    added = set(json.loads(proc.stdout))
    leaked = {
        name
        for name in added
        if any(name == root or name.startswith(root + ".") for root in FORBIDDEN_ROOTS)
    }
    assert not leaked, f"molexp.agent pulled in forbidden modules: {sorted(leaked)}"
