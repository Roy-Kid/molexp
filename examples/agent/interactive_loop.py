"""``AgentRunner`` + ``InteractiveLoop`` — emergent read-only tool loop, real DeepSeek.

The second shipped agent loop (alongside ``ChatLoop``). Where ``ChatLoop`` is
one user input → one LLM round-trip, ``InteractiveLoop`` drives an *emergent*
loop: the model may call read-only workspace/code tools across several turns
(via ``Router.stream_agentic``) before answering. This is the loop behind the
``molexp agent`` CLI REPL.

Key registration — molexp reads the LLM key from ``molexp.config``
(registered in code), **never from the environment**. Paste your DeepSeek key
into ``API_KEY`` below.

Run directly::

    python examples/agent/interactive_loop.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import molexp
from molexp.agent import AgentRunner
from molexp.agent.loops import InteractiveLoop, InteractiveLoopConfig

MODEL = "deepseek:deepseek-v4-flash"
API_KEY = ""  # ← paste your DeepSeek key here (registered via molexp.config, not env)


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "lab"
        workspace.mkdir()
        (workspace / "notes.txt").write_text("ionic mobility ~ 1e-8 m^2/Vs\n")

        runner = AgentRunner(
            loop=InteractiveLoop(
                config=InteractiveLoopConfig(
                    system_prompt="you are a concise, read-only coding assistant",
                    workspace_root=workspace,
                )
            ),
            model=MODEL,
            workspace=workspace,
        )
        session = runner.session("interactive-demo")
        print(f"session_id: {session.session_id}")

        # One emergent turn: the model may invoke read-only tools (search /
        # read) before answering — all surfaced as AgentEvents.
        result = await runner.run(session, "What's in notes.txt?")
        print(f"answer:         {result.text[:120]}")
        print(f"events emitted: {len(result.events)}")


if __name__ == "__main__":
    if not API_KEY:
        print("Set API_KEY at the top of this file and re-run.")
    else:
        molexp.config["deepseek_api_key"] = API_KEY
        asyncio.run(main())
