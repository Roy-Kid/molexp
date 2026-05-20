"""Import-boundary guard for ``molexp.agent.modes.review`` (ac-009).

ReviewMode's pipeline is plain async stages on the harness — it imports
no ``pydantic_ai`` and the router is injected lazily by ``AgentRunner``.
``pydantic_graph`` may legitimately load transitively through
``molexp.workflow`` (sibling PlanMode / AuthorMode / RunMode wiring,
pulled in by ``molexp.agent.modes.__init__``); its confinement to
``workflow/_pydantic_graph/`` is enforced separately. So:

- ``import molexp.agent`` must load neither SDK (harness is lazy).
- ``import molexp.agent.modes.review`` must not load ``pydantic_ai``.
"""

from __future__ import annotations

import subprocess
import sys


def _assert_no_sdks(module: str, forbidden: tuple[str, ...]) -> None:
    """Assert importing ``module`` leaves every ``forbidden`` SDK unloaded."""
    forbidden_repr = repr(forbidden)
    code = (
        "import sys\n"
        f"import {module}  # noqa: F401\n"
        f"for forbidden in {forbidden_repr}:\n"
        "    assert forbidden not in sys.modules, (\n"
        f"        f'{{forbidden}} was eagerly loaded by {module}'\n"
        "    )\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout


def test_import_molexp_agent_stays_sdk_free() -> None:
    """Plain ``import molexp.agent`` loads neither pydantic SDK."""
    _assert_no_sdks("molexp.agent", ("pydantic_ai", "pydantic_graph"))


def test_import_review_package_loads_no_pydantic_ai() -> None:
    """``import molexp.agent.modes.review`` never eagerly loads pydantic_ai."""
    _assert_no_sdks("molexp.agent.modes.review", ("pydantic_ai",))
