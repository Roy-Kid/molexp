"""``AgentRunner`` + ``ChatLoop`` — offline-first multi-turn conversation.

``ChatLoop`` is the minimal :class:`~molexp.agent.loop.AgentLoop`: each turn
is one user input → one LLM round-trip → one ``AgentRunResult``. The LLM
seam is the SDK-free :class:`molexp.agent.router.Router` Protocol — and this
example exercises it OFFLINE: the in-file :class:`ScriptedRouter` below
implements the Protocol with canned replies and is injected through
``AgentRunner(router=...)`` (the runner's one-of-three config:
``model=`` | ``models=`` | ``router=``). No network, no API key, exit 0 —
which is what lets the examples smoke test gate this file against API drift.

What the demo proves: with ``workspace=`` set, the named session persists to
``entries.jsonl`` — after two turns on the same ``runner.session(...)``, the
second result carries all four messages (2 user + 2 assistant) rebuilt from
disk.

LIVE MODE — paste a DeepSeek key into ``API_KEY`` (molexp reads LLM keys
from ``molexp.config``, registered in code, never from the environment) and
the same loop runs against the real model via ``model=MODEL``.

Run directly::

    python examples/agent/chat_loop.py
"""

from __future__ import annotations

import asyncio
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

import molexp
from molexp.agent import AgentRunner
from molexp.agent.loops import ChatLoop, ChatLoopConfig
from molexp.agent.router import ModelTier, RouterTextResult
from molexp.agent.types import UsageBreakdown

MODEL = "deepseek:deepseek-v4-flash"
API_KEY = ""  # ← paste your DeepSeek key here for live mode (in-code key law)


class ScriptedRouter:
    """In-file ``Router`` Protocol impl: pops canned replies, no network.

    Only ``complete_text`` does real work — that is the one method
    ``ChatLoop`` drives. The other Protocol methods exist so the class
    satisfies the runtime-checkable structural type; ``snapshot_usage``
    must return a real :class:`UsageBreakdown` because ``ChatLoop`` folds
    it into the pydantic-validated ``AgentRunResult``.
    """

    def __init__(self, replies: list[str]) -> None:
        self._replies = deque(replies)

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        del prompt, system, message_history, tier  # scripted: input-independent
        return RouterTextResult(text=self._replies.popleft())

    async def complete_structured(self, **kwargs: Any) -> Any:
        raise NotImplementedError("ScriptedRouter scripts text turns only")

    def stream_agentic(self, **kwargs: Any) -> Any:
        raise NotImplementedError("see examples/agent/interactive_loop.py")

    def clear_usage(self) -> None:  # scripted calls cost nothing
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def _build_runner(workspace: Path) -> AgentRunner:
    """Offline by default (router injection); live when API_KEY is set."""
    loop = ChatLoop(config=ChatLoopConfig(system_prompt="you are concise"))
    if API_KEY:
        molexp.config["deepseek_api_key"] = API_KEY
        return AgentRunner(loop=loop, model=MODEL, workspace=workspace)
    scripted = ScriptedRouter(["2 + 2 = 4.", "Times two, that makes 8."])
    return AgentRunner(loop=loop, router=scripted, workspace=workspace)


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "lab"
        workspace.mkdir()

        runner = _build_runner(workspace)
        session = runner.session("chat-demo")
        print(f"mode      : {'live' if API_KEY else 'offline (ScriptedRouter)'}")
        print(f"session_id: {session.session_id}")

        result_a = await runner.run(session, "what is 2 + 2?")
        print(f"[turn 1] {result_a.text[:80]}")

        # Same named session → the loop rebuilds prior context from the
        # persisted entry-tree (entries.jsonl under the workspace).
        result_b = await runner.run(session, "and what about times two?")
        print(f"[turn 2] {result_b.text[:80]}")

        # The persistence proof: turn 2 sees all four messages from disk.
        assert len(result_b.messages) == 4, (
            f"expected 4 persisted messages (2 user + 2 assistant), got {len(result_b.messages)}"
        )
        print(f"messages persisted across turns: {len(result_b.messages)}")


if __name__ == "__main__":
    asyncio.run(main())
