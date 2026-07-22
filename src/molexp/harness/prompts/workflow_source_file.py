"""System prompt for the ``workflow_source_file_writer`` per-task codegen agent.

User payload is a **YAML document** built by
:func:`molexp.harness.prompts.codegen_prompt.assemble_task_codegen_prompt`
(contract + task + wiring + optional repair/domain context). Keep this
system string short — rules live in the YAML ``contract`` section.
"""

from __future__ import annotations

__all__ = ["SYSTEM_PROMPT"]

SYSTEM_PROMPT = (
    "You generate ONE molexp.workflow task module from a YAML user document.\n"
    "Read sections: contract (MUST), task, wiring, repair (if present), "
    "domain_context (molmcp pages for domain APIs only).\n"
    "Output: WorkflowSource JSON with exactly one file "
    "`workflow/<task.slug>.py` containing only `async def <slug>(...)` and "
    "in-function imports. No WorkflowCompiler, no build_workflow, no prose, "
    "no markdown fences.\n"
    "Dataflow: return a dict whose keys match task.outputs; upstream values "
    "arrive as parameters named like task.inputs / wiring edges. "
    "Domain APIs (molpy/molpack) via contract.domain + domain_context / molmcp; "
    "never invent symbols; never import molrs."
)
