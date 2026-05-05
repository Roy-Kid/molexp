"""Native chat tools — currently just :func:`ask_user`.

Plan mode lives in the runner state machine, not in this surface;
reject feedback flows back as a synthetic user message via
:func:`molexp.agent.orchestration.render_reject_feedback`.
"""

from __future__ import annotations

from typing import Any

from molexp.agent.tools.native._helpers import err, ok
from molexp.agent.tools.registry import native_tool
from molexp.agent.tools.spec import ToolContext, ToolResult, ToolSpec


@native_tool(
    ToolSpec(
        name="native:ask_user",
        description="Pause the run and prompt the user for free-form input.",
        input_schema={
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
        category="chat",
        mutates=False,
    )
)
async def ask_user(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    if ctx.chat is None:
        return err("ask_user requires an interactive session with a chat gateway")
    reply = await ctx.chat.ask(args["prompt"])
    return ok({"content": reply})
