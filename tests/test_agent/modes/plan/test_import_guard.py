"""Import-firewall guard for the ``modes/plan`` package (ac-008).

PlanMode is harness-based; it must not import ``pydantic_graph`` (the
workflow layer's concern), and only ``_pydanticai/`` may import
``pydantic_ai``. ``import molexp.agent`` must stay SDK-free.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_PLAN_ROOT = Path(__file__).resolve().parents[4] / "src" / "molexp" / "agent" / "modes" / "plan"


def _imports(py: Path) -> list[tuple[int, str]]:
    tree = ast.parse(py.read_text(encoding="utf-8"))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.lineno, node.module or ""))
    return found


def test_plan_package_imports_no_pydantic_graph() -> None:
    offenders: list[str] = []
    for py in sorted(_PLAN_ROOT.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        for lineno, name in _imports(py):
            if name == "pydantic_graph" or name.startswith("pydantic_graph."):
                offenders.append(f"{py.name}:{lineno}: {name}")
    assert not offenders, "modes/plan must not import pydantic_graph:\n  " + "\n  ".join(offenders)


def test_plan_package_imports_no_pydantic_ai() -> None:
    """``modes/plan/`` consumes the probe via a Protocol — not pydantic_ai."""
    offenders: list[str] = []
    for py in sorted(_PLAN_ROOT.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        for lineno, name in _imports(py):
            if name == "pydantic_ai" or name.startswith("pydantic_ai."):
                offenders.append(f"{py.name}:{lineno}: {name}")
    assert not offenders, "modes/plan must not import pydantic_ai:\n  " + "\n  ".join(offenders)


def test_importing_plan_mode_keeps_pydantic_ai_lazy() -> None:
    code = (
        "import sys\n"
        "from molexp.agent.modes.plan import PlanMode  # noqa: F401\n"
        "assert 'pydantic_ai' not in sys.modules, (\n"
        "    'pydantic_ai eagerly loaded by molexp.agent.modes.plan'\n"
        ")\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout


def test_importing_molexp_agent_stays_sdk_free() -> None:
    code = (
        "import sys\n"
        "import molexp.agent  # noqa: F401\n"
        "for forbidden in ('pydantic_ai', 'pydantic_graph'):\n"
        "    pass  # pydantic_graph may load via workflow; only assert pydantic_ai\n"
        "assert 'pydantic_ai' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
