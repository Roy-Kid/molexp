"""Tests for the ``molexp.harness.prompts`` package.

Locks the two load-bearing contracts: importing the package pulls no
pydantic-ai / pydantic-graph SDK into ``sys.modules`` (subprocess probe,
mirroring ``test_import_guard.py``), and ``prompts_by_agent()`` keys exactly
the canonical planning ``agent_name`` registry.
"""

from __future__ import annotations

import subprocess
import sys

# Every canonical agent_name the planning stages set, across all pipeline specs.
_AGENT_NAMES = {
    "experiment_report_writer",
    "workflow_ir_extractor",
    "bound_workflow_binder",
    "plan_reviewer",
    "test_spec_writer",
    "test_code_writer",
    "final_report_writer",
    "experiment_spec_generator",
    "capability_selector",
    "input_set_generator",
}


class TestHarnessPrompts:
    def test_import_prompts_pulls_no_sdk_into_sys_modules(self) -> None:
        """``import molexp.harness.prompts`` is SDK-free.

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

    def test_prompts_by_agent_keys_exactly_the_canonical_agent_names(self) -> None:
        """``prompts_by_agent()`` keys == the canonical planning agent_names."""
        from molexp.harness.prompts import prompts_by_agent

        assert set(prompts_by_agent()) == _AGENT_NAMES
