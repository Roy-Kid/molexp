"""Architectural direction guard for ``molexp.harness``.

Spec ac-011 — ``import molexp.harness`` MUST NOT pull
``pydantic_ai`` / ``pydantic_graph`` / ``molexp.agent`` / ``molexp.workflow``
into ``sys.modules``. The harness is a foundational layer that the agent
and workflow layers will eventually rebuild on top of; if it ever depends
upward, that becomes a circular import waiting to happen.

We run the check in a fresh subprocess so a stale ``sys.modules`` (from
another test that did import ``molexp.agent``) can't poison the assertion.
"""

from __future__ import annotations

import subprocess
import sys

_FORBIDDEN = ("pydantic_ai", "pydantic_graph", "molexp.agent", "molexp.workflow")


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
