"""Import-firewall guard for the ``_planning`` package (ac-011).

The shared planning-contracts package is pure frozen-pydantic data
models. Its own source must import neither pydantic SDK, importing it
must keep ``pydantic_ai`` lazy, and it must stay purely additive — it is
not exported from ``molexp.agent.modes``.

Note: ``pydantic_graph`` legitimately loads transitively through
``molexp.workflow`` whenever ``molexp.agent.modes`` is imported (the
``modes`` package eagerly imports PlanMode, which wires the workflow
layer). That is pre-existing, sanctioned behaviour — see
``tests/test_agent/test_import_guard.py``. This guard therefore asserts
the achievable invariant: the ``_planning`` package adds no SDK import
of its own.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_PLANNING_ROOT = (
    Path(__file__).resolve().parents[3] / "src" / "molexp" / "agent" / "modes" / "_planning"
)


def test_planning_package_source_imports_no_pydantic_sdk() -> None:
    """No module under ``_planning/`` imports ``pydantic_ai`` / ``pydantic_graph``."""
    offenders: list[str] = []
    for py in sorted(_PLANNING_ROOT.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "pydantic_ai" or name.startswith("pydantic_ai."):
                    offenders.append(f"{py.name}:{node.lineno}: {name}")
                if name == "pydantic_graph" or name.startswith("pydantic_graph."):
                    offenders.append(f"{py.name}:{node.lineno}: {name}")
    assert not offenders, "_planning must import no pydantic SDK:\n  " + "\n  ".join(offenders)


def test_importing_planning_keeps_pydantic_ai_lazy() -> None:
    """``import molexp.agent.modes._planning`` does not eagerly load ``pydantic_ai``."""
    code = (
        "import sys\n"
        "import molexp.agent.modes._planning  # noqa: F401\n"
        "assert 'pydantic_ai' not in sys.modules, (\n"
        "    'pydantic_ai was eagerly loaded by molexp.agent.modes._planning'\n"
        ")\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout


def test_planning_package_is_additive_to_modes_surface() -> None:
    """``_planning`` is private — it is not exported from ``molexp.agent.modes``."""
    import molexp.agent.modes as modes

    assert "_planning" not in getattr(modes, "__all__", ())


def test_importing_molexp_agent_modes_still_succeeds() -> None:
    """The new sub-package does not break the ``molexp.agent.modes`` import."""
    code = "import molexp.agent.modes  # noqa: F401\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
