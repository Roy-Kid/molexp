"""Architectural direction guard for ``molexp.harness``.

``import molexp.harness`` MUST NOT pull ``pydantic_ai`` / ``pydantic_graph``
/ ``molexp.workflow`` into ``sys.modules``.

Post spec ``harness-as-mode-substrate-03a``: ``molexp.agent`` is **allowed**
— the new ``RouterBackedAgentGateway`` legally imports ``agent.router``
(the Protocol module, which is itself SDK-free per spec 01 ac-013). The
charter pivot lets harness sit above agent in the DAG; the only remaining
forbidden-edge invariant is that pydantic-ai SDKs and the workflow layer
must not load eagerly when the harness package is imported.

We run the check in a fresh subprocess so a stale ``sys.modules`` (from
another test that already imported the workflow layer) can't poison the
assertion.
"""

from __future__ import annotations

import subprocess
import sys

_FORBIDDEN = ("pydantic_ai", "pydantic_graph", "molexp.workflow")


def test_import_molexp_harness_does_not_pull_forbidden_modules() -> None:
    probe = (
        "import sys, importlib;"
        "importlib.import_module('molexp.harness');"
        "loaded = [m for m in sys.modules if m in " + repr(list(_FORBIDDEN)) + "];"
        "print('LOADED:' + ','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    assert output.startswith("LOADED:"), output
    loaded = [m for m in output.removeprefix("LOADED:").split(",") if m]
    assert loaded == [], f"forbidden modules imported transitively: {loaded}"


def test_import_molexp_harness_mode_does_not_pull_forbidden_modules() -> None:
    """ac-008: ``import molexp.harness.mode`` must load ``molexp.workflow`` lazily.

    ``Mode.run`` imports ``molexp.workflow`` *inside the method body* (mirroring
    ``agent.AgentRunner.run()`` deferring ``pydantic_ai``). Merely importing the
    ``molexp.harness.mode`` module must therefore leave ``molexp.workflow`` —
    and the ``pydantic_ai`` / ``pydantic_graph`` SDKs it transitively loads —
    out of ``sys.modules``. Run in a fresh subprocess so a stale ``sys.modules``
    from another test cannot poison the assertion.
    """
    probe = (
        "import sys, importlib;"
        "importlib.import_module('molexp.harness.mode');"
        "loaded = [m for m in sys.modules if m in " + repr(list(_FORBIDDEN)) + "];"
        "print('LOADED:' + ','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    assert output.startswith("LOADED:"), output
    loaded = [m for m in output.removeprefix("LOADED:").split(",") if m]
    assert loaded == [], f"forbidden modules imported transitively by harness.mode: {loaded}"
