"""``AgentRunner`` + ``InteractiveLoop`` — offline-first emergent tool loop.

Where ``ChatLoop`` is one user input → one round-trip, ``InteractiveLoop``
drives an *emergent* loop through ``Router.stream_agentic``: the model may
call tools across several rounds before answering, and every increment
surfaces as a typed chunk that the loop translates into ``AgentEvent``s
(``ToolCallChunk → ToolCallStartedEvent``, ``ToolResultChunk →
ToolCallCompletedEvent``, …). This is the loop behind the ``molexp agent``
CLI REPL.

OFFLINE BY DEFAULT: the in-file :class:`ScriptedRouter` implements the
SDK-free Router Protocol with a scripted ``stream_agentic`` — one thinking
delta, one full tool round, then the streamed answer — injected via
``AgentRunner(router=...)``. The demo self-asserts that the run's event
stream contains a complete tool round, proving the chunk→event translation
end-to-end with zero network. Paste a DeepSeek key into ``API_KEY`` for the
live version of the same loop.

Run directly::

    python examples/agent/interactive_loop.py
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
from molexp.agent.loops import InteractiveLoop, InteractiveLoopConfig
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
API_KEY = ""  # ← paste your DeepSeek key here for live mode (in-code key law)

_NOTE = "ionic mobility ~ 1e-8 m^2/Vs"
_ANSWER = f"notes.txt says: {_NOTE}"


class ScriptedRouter:
    """In-file ``Router`` Protocol impl scripting one emergent tool round."""

    async def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
    ) -> AsyncIterator[AgenticChunk]:
        del prompt, system, tools, tier, message_history  # scripted
        yield ThinkingDeltaChunk(text="The user asks about notes.txt — read it first.")
        yield ToolCallChunk(tool_name="read_file", args_summary="notes.txt")
        yield ToolResultChunk(tool_name="read_file", result_summary=_NOTE, ok=True)
        yield TextDeltaChunk(text="notes.txt says: ")
        yield TextDeltaChunk(text=_NOTE)
        yield FinalChunk(text=_ANSWER)

    async def complete_text(self, **kwargs: Any) -> Any:
        raise NotImplementedError("see examples/agent/chat_loop.py")

    async def complete_structured(self, **kwargs: Any) -> Any:
        raise NotImplementedError("ScriptedRouter scripts the agentic stream only")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def _build_runner(workspace: Path) -> AgentRunner:
    """Offline by default (router injection); live when API_KEY is set."""
    loop = InteractiveLoop(
        config=InteractiveLoopConfig(
            system_prompt="you are a concise, read-only coding assistant",
            workspace_root=workspace,
        )
    )
    if API_KEY:
        molexp.config["deepseek_api_key"] = API_KEY
        return AgentRunner(loop=loop, model=MODEL, workspace=workspace)
    return AgentRunner(loop=loop, router=ScriptedRouter(), workspace=workspace)


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "lab"
        workspace.mkdir()
        (workspace / "notes.txt").write_text(_NOTE + "\n")

        runner = _build_runner(workspace)
        session = runner.session("interactive-demo")
        print(f"mode      : {'live' if API_KEY else 'offline (ScriptedRouter)'}")
        print(f"session_id: {session.session_id}")

        result = await runner.run(session, "What's in notes.txt?")
        print(f"answer        : {result.text[:120]}")
        print(f"events emitted: {len(result.events)}")

        # The emergent-loop proof: the chunk stream surfaced as AgentEvents
        # with at least one complete tool round before the final answer.
        started = [e for e in result.events if isinstance(e, ToolCallStartedEvent)]
        completed = [e for e in result.events if isinstance(e, ToolCallCompletedEvent)]
        assert started, "expected at least one ToolCallStartedEvent"
        assert completed, "expected at least one ToolCallCompletedEvent"
        print(f"tool rounds   : {len(started)} started / {len(completed)} completed")


if __name__ == "__main__":
    asyncio.run(main())
