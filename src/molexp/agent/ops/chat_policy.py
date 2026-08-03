"""Chat Mode policy — deny authoritative workspace mutation tools.

Used as a :class:`~molexp.agent.loops.hooks.BeforeToolHook` so MCP tools
that create project/experiment/run cannot bypass the chat tool surface.
"""

from __future__ import annotations

from typing import Any

from molexp.agent.loops.hooks import HookOutcome

__all__ = [
    "CHAT_MUTATOR_NEEDLES",
    "chat_before_tool",
    "is_chat_workspace_mutator",
]

#: Substrings matched case-insensitively against the tool name.
CHAT_MUTATOR_NEEDLES: tuple[str, ...] = (
    "add_project",
    "add_experiment",
    "create_run",
    "workspace_ensure",
    "run_land",
    "remove_project",
    "remove_experiment",
    "delete_project",
    "delete_experiment",
    "delete_run",
    "molexp_materialize",
)


def is_chat_workspace_mutator(tool_name: str) -> bool:
    """Return True if *tool_name* must not run in Chat Mode."""
    low = tool_name.lower().replace("-", "_")
    return any(n in low for n in CHAT_MUTATOR_NEEDLES)


async def chat_before_tool(tool_name: str, args: Any = None) -> HookOutcome:  # noqa: ANN401
    """Deny structure-mutating tools; allow everything else."""
    del args
    if is_chat_workspace_mutator(tool_name):
        return HookOutcome.deny(
            "Chat Mode refuses workspace mutators "
            f"({tool_name!r}). Do not create project/experiment/run here. "
            "Keep work under agent/.scratch/ and ask the user whether to 落盘; "
            "use Plan Mode for formal experiments."
        )
    return HookOutcome.proceed()
