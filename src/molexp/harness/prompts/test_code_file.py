"""System prompt for the ``test_code_file_writer`` per-task pytest agent.

User payload is YAML from
:func:`molexp.harness.prompts.codegen_prompt.assemble_task_test_prompt`.
"""

from __future__ import annotations

__all__ = ["SYSTEM_PROMPT"]

SYSTEM_PROMPT = (
    "You generate ONE pytest module from a YAML user document.\n"
    "Read sections: contract (MUST), task, module_under_test, wiring, "
    "repair (if present), domain_context.\n"
    "Output: TestSource JSON with exactly one file `tests/test_<slug>.py`. "
    "Prefer sync `def test_*` + `asyncio.run` for async tasks. "
    "Import every name used; mock only symbols present in module_under_test.source. "
    "No prose, no markdown fences."
)
