"""Tests for the ``molexp.harness.prompts`` package (spec
``plan-mode-revival-02-structured-planning``).

RED before implementation: the package does not exist yet, so the
imports fail. After GREEN, these assert that each of the four planning
agents has a non-empty, SDK-free ``SYSTEM_PROMPT`` string, that
``prompts_by_agent()`` keys the four canonical ``agent_name``s, that the
package docstring records the DeepSeek production wiring for spec 04, and
that importing the package pulls no pydantic-ai / pydantic-graph SDK into
``sys.modules`` (subprocess probe, mirroring ``test_import_guard.py``).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# The four canonical agent_names that key the planning stages.
_AGENT_NAMES = {
    "experiment_report_writer",
    "workflow_ir_extractor",
    "bound_workflow_binder",
    "test_spec_writer",
}


# ── ac-005: each SYSTEM_PROMPT module is a non-empty str ───────────────────


@pytest.mark.parametrize(
    "module_name",
    [
        "experiment_report",
        "workflow_ir",
        "bound_workflow",
        "test_spec",
    ],
)
def test_each_module_defines_non_empty_system_prompt(module_name: str) -> None:
    """ac-005 — each prompt module exposes a non-empty ``SYSTEM_PROMPT: str``."""
    import importlib

    module = importlib.import_module(f"molexp.harness.prompts.{module_name}")
    prompt = module.SYSTEM_PROMPT
    assert isinstance(prompt, str)
    assert prompt.strip(), f"{module_name}.SYSTEM_PROMPT must be non-empty"


def test_package_reexports_four_system_prompts() -> None:
    """ac-005 — ``molexp.harness.prompts`` re-exports the four SYSTEM_PROMPTs."""
    from molexp.harness import prompts

    for name in (
        "EXPERIMENT_REPORT_SYSTEM_PROMPT",
        "WORKFLOW_IR_SYSTEM_PROMPT",
        "BOUND_WORKFLOW_SYSTEM_PROMPT",
        "TEST_SPEC_SYSTEM_PROMPT",
    ):
        value = getattr(prompts, name)
        assert isinstance(value, str)
        assert value.strip(), f"{name} must be a non-empty str"


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


# ── ac-006: prompts_by_agent() keys exactly the four agent_names ───────────


def test_prompts_by_agent_keys_exactly_the_four_agent_names() -> None:
    """ac-006 — ``prompts_by_agent()`` keys == the four stage agent_names."""
    from molexp.harness.prompts import prompts_by_agent

    mapping = prompts_by_agent()
    assert set(mapping) == _AGENT_NAMES


def test_prompts_by_agent_values_are_non_empty_strings() -> None:
    """ac-006 — every value in ``prompts_by_agent()`` is a non-empty str."""
    from molexp.harness.prompts import prompts_by_agent

    mapping = prompts_by_agent()
    for agent_name, prompt in mapping.items():
        assert isinstance(prompt, str)
        assert prompt.strip(), f"prompt for {agent_name!r} must be non-empty"


def test_prompts_by_agent_returns_a_dict() -> None:
    """ac-006 — ``prompts_by_agent()`` returns a ``dict[str, str]``."""
    from molexp.harness.prompts import prompts_by_agent

    assert isinstance(prompts_by_agent(), dict)


# ── ac-010: docstring records the DeepSeek production wiring ───────────────


def test_package_docstring_documents_deepseek_wiring() -> None:
    """ac-010 — the package docstring cites the DeepSeek production wiring.

    Spec 04 performs the live call; this spec documents the one-liner so
    04 has a single cited reference.
    """
    from molexp.harness import prompts

    doc = prompts.__doc__ or ""
    assert "deepseek:deepseek-v4-flash" in doc
    assert "dict.fromkeys" in doc
