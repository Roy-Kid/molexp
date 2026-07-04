"""Tests for the ``molexp.harness.prompts`` package (spec
``plan-mode-revival-02-structured-planning``).

Locks the two load-bearing contracts: ``prompts_by_agent()`` keys exactly
the canonical ``agent_name`` registry (with non-empty prompts), and
importing the package pulls no pydantic-ai / pydantic-graph SDK into
``sys.modules`` (subprocess probe, mirroring ``test_import_guard.py``).
"""

from __future__ import annotations

import subprocess
import sys

# The four canonical agent_names that key the planning stages.
_AGENT_NAMES = {
    "experiment_report_writer",
    "workflow_ir_extractor",
    "bound_workflow_binder",
    "plan_reviewer",
    "test_spec_writer",
}

# The two run-mode agent_names added by spec harness-run-mode-01-substrate.
_RUN_AGENT_NAMES = {
    "test_code_writer",
    "final_report_writer",
}

# The agent_names added by the 9-step plan-pipeline redesign.
_V2_AGENT_NAMES = {
    "experiment_spec_generator",
    "capability_selector",
    "input_set_generator",
}


# ── ac-005 / ac-008: importing prompts pulls no SDK into sys.modules ───────


def test_import_prompts_pulls_no_sdk_into_sys_modules() -> None:
    """ac-005/ac-008 — ``import molexp.harness.prompts`` is SDK-free.

    Run in a fresh subprocess so a stale ``sys.modules`` (from another
    test that already imported pydantic-ai) cannot poison the assertion.
    """
    forbidden = ["pydantic_ai", "pydantic_graph"]
    probe = (
        "import sys, importlib;"
        "importlib.import_module('molexp.harness.prompts');"
        "loaded = [m for m in sys.modules if m in " + repr(forbidden) + "];"
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
    assert loaded == [], f"prompts import pulled forbidden SDK modules: {loaded}"


# ── ac-006: prompts_by_agent() keys exactly the canonical agent_names ──────


def test_prompts_by_agent_keys_exactly_the_canonical_agent_names() -> None:
    """ac-006 + harness-run-mode-01 — ``prompts_by_agent()`` keys == the four
    planning agent_names plus the two run-mode writer agent_names.

    (``workflow_source_writer`` is a known pre-existing gap, deliberately
    left to the 02-wire leg — see the substrate spec's Out of scope.)
    """
    from molexp.harness.prompts import prompts_by_agent

    mapping = prompts_by_agent()
    assert set(mapping) == _AGENT_NAMES | _RUN_AGENT_NAMES | _V2_AGENT_NAMES


def test_prompts_by_agent_values_are_non_empty_strings() -> None:
    """ac-006 — every value in ``prompts_by_agent()`` is a non-empty str."""
    from molexp.harness.prompts import prompts_by_agent

    mapping = prompts_by_agent()
    for agent_name, prompt in mapping.items():
        assert isinstance(prompt, str)
        assert prompt.strip(), f"prompt for {agent_name!r} must be non-empty"
