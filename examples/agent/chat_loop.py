"""``AgentRunner`` + ``ChatLoop`` — multi-turn conversation against real DeepSeek.

``ChatLoop`` is the minimal :class:`~molexp.agent.loop.AgentLoop`: each turn is
one user input → one LLM round-trip → one ``AgentRunResult``. ``AgentRunner``
builds the :class:`~molexp.agent.runtime.AgentRuntime` (session + router +
execution-env + hooks) and injects it; ``run`` drains the loop's ``AgentEvent``
stream into the result. With ``workspace=`` set, the named session persists to
``entries.jsonl`` and resumes across runner instances.

Key registration — molexp reads the LLM key from ``molexp.config``
(registered in code), **never from the environment**. Paste your DeepSeek key
into ``API_KEY`` below.

Run directly::

    python examples/agent/chat_loop.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import molexp
from molexp.agent import AgentRunner
from molexp.agent.loops import ChatLoop, ChatLoopConfig

MODEL = "deepseek:deepseek-v4-flash"
API_KEY = ""  # ← paste your DeepSeek key here (registered via molexp.config, not env)


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "lab"
        workspace.mkdir()

        runner = AgentRunner(
            loop=ChatLoop(config=ChatLoopConfig(system_prompt="you are concise")),
            model=MODEL,
            workspace=workspace,
        )
        session = runner.session("chat-with-roy")
        print(f"session_id: {session.session_id}")

        result_a = await runner.run(session, "what is 2 + 2?")
        print(f"[turn 1] {result_a.text[:80]}")

        # The model sees turn 1 because we pass the same session — the loop
        # rebuilds prior context from the persisted entry-tree.
        result_b = await runner.run(session, "and what about times two?")
        print(f"[turn 2] {result_b.text[:80]}")


if __name__ == "__main__":
    if not API_KEY:
        print("Set API_KEY at the top of this file and re-run.")
    else:
        molexp.config["deepseek_api_key"] = API_KEY
        asyncio.run(main())
