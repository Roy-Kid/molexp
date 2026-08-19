"""Agent with MCP toolsets — ``MCPServerStdio`` pattern + ReAct tool events.

Matches ``docs/concept/agent.md``.

Demonstrates:

1. Offline-first ``ScriptedRouter`` simulating MCP tool-call responses.
2. ``MCPServerStdio`` construction pattern (commented — for real LLM runs).
3. Live mode: configure ``mcp.json`` (or user ``~/.molexp/mcp.json``);
   a ReAct turn opens entries via ``McpCatalog`` and passes
   them as ``stream_agentic(toolsets=...)``.
4. ReAct emitting ``ToolCallStartedEvent`` / ``ToolCallCompletedEvent``.

The offline mode proves the loop's tool-call contract works without a network;
paste a key into ``API_KEY`` for live LLM mode. MCP servers come from McpStore,
not from ``AgentRunner(toolsets=...)``. The scripted stream yields one tool
round then a streamed answer — the demo asserts the events landed.

Run directly::

    python examples/agent/mcp_integration.py
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import molexp
from molexp.agent import AgentRunner
from molexp.agent.events import ToolCallCompletedEvent, ToolCallStartedEvent
from molexp.agent.router import (
    AgenticChunk,
    FinalChunk,
    ModelTier,
    TextDeltaChunk,
    ThinkingDeltaChunk,
    ToolCallChunk,
    ToolResultChunk,
)
from molexp.agent.types import UsageBreakdown

MODEL = "deepseek:deepseek-v4-flash"
API_KEY = ""  # ← paste your key here for live mode

_RESULT = "Dataset summary: 1,000 rows, 12 columns, no outliers detected."


class ScriptedRouter:
    """In-file ``Router`` Protocol — scripts one MCP-style tool round."""

    async def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        toolsets: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
    ) -> AsyncIterator[AgenticChunk]:
        del prompt, system, tools, toolsets, tier, message_history  # scripted
        yield ThinkingDeltaChunk(text="The user wants dataset analysis — query the DB first.")
        yield ToolCallChunk(tool_name="query_db", args_summary="SELECT COUNT(*), ...")
        yield ToolResultChunk(tool_name="query_db", result_summary=_RESULT, ok=True)
        yield TextDeltaChunk(text=_RESULT)
        yield FinalChunk(text=_RESULT)

    async def complete_text(self, **kwargs: Any) -> Any:
        raise NotImplementedError("this demo uses stream_agentic")

    async def complete_structured(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def _build_runner(workspace: Path) -> AgentRunner:
    # Live MCP: write workspace/mcp.json (or ~/.molexp/mcp.json); the ReAct turn
    # opens valid entries automatically via McpCatalog + stream_agentic.
    kwargs = {
        "workspace": workspace,
        "mode": "agentic",
        "system_prompt": "you are a data-analysis assistant with database tool access",
    }
    if API_KEY:
        molexp.config["deepseek_api_key"] = API_KEY
        return AgentRunner(model=MODEL, **kwargs)
    return AgentRunner(router=ScriptedRouter(), **kwargs)


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "lab"
        workspace.mkdir()

        runner = _build_runner(workspace)
        session = runner.session("mcp-demo")
        mode = "live" if API_KEY else "offline (ScriptedRouter simulates MCP tools)"
        print(f"mode:       {mode}")
        print(f"session_id: {session.session_id}")

        result = await runner.run(session, "analyze the dataset for outliers")
        print(f"answer:        {result.text[:120]}")
        print(f"events emitted: {len(result.events)}")

        started = [e for e in result.events if isinstance(e, ToolCallStartedEvent)]
        completed = [e for e in result.events if isinstance(e, ToolCallCompletedEvent)]
        assert started, "expected at least one ToolCallStartedEvent"
        assert completed, "expected at least one ToolCallCompletedEvent"
        print(f"tool rounds:   {len(started)} started / {len(completed)} completed")


if __name__ == "__main__":
    asyncio.run(main())
