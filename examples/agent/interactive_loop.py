"""``AgentRunner`` + ``InteractiveLoop`` — the emergent read-only tool loop.

The second shipped agent loop (alongside ``ChatLoop``). Where ``ChatLoop`` is
one user input → one LLM round-trip, ``InteractiveLoop`` drives an *emergent*
loop: the model may call read-only workspace/code tools across several
turns (via ``Router.stream_agentic``) before producing its answer. This is
the loop behind the ``molexp agent`` CLI REPL.

Like ``chat_loop.py`` this uses ``pydantic_ai.models.test.TestModel`` so it
runs offline without an API key — TestModel exercises the loop deterministically.
To talk to a real provider, pass ``model="deepseek:deepseek-v4-flash"`` (and set
``DEEPSEEK_API_KEY``) instead of the ``TestModel`` object.

Run directly::

    python examples/agent/interactive_loop.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from pydantic_ai.models.test import TestModel

from molexp.agent import AgentRunner
from molexp.agent.loops import InteractiveLoop, InteractiveLoopConfig


def _make_runner(workspace: Path) -> AgentRunner:
    """Build an InteractiveLoop runner backed by an on-disk workspace.

    ``InteractiveLoopConfig.workspace_root`` scopes the read-only tools (code
    search, file read) the emergent loop may call.
    """
    return AgentRunner(
        loop=InteractiveLoop(
            config=InteractiveLoopConfig(
                system_prompt="you are a concise, read-only coding assistant",
                workspace_root=workspace,
            )
        ),
        # ``call_tools=[]`` keeps the offline demo deterministic: TestModel
        # synthesizes a text answer instead of invoking the read-only tools
        # with placeholder args (which would miss the workspace files). A
        # real provider would actually call ``read_file`` / ``search_code``.
        model=TestModel(call_tools=[]),  # type: ignore[arg-type] — model object in lieu of an id
        workspace=workspace,
    )


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "lab"
        workspace.mkdir()
        (workspace / "notes.txt").write_text("ionic mobility ~ 1e-8 m^2/Vs\n")

        runner = _make_runner(workspace)
        session = runner.session("interactive-demo")
        print(f"session_id:     {session.session_id}")

        # One emergent turn: the model may invoke read-only tools (search /
        # read) before answering — all surfaced as AgentEvents.
        result = await runner.run(session, "What's in notes.txt?")
        print(f"answer:         {result.text[:80]}")
        print(f"events emitted: {len(result.events)}")
        print(f"event types:    {sorted({type(e).__name__ for e in result.events})}")


if __name__ == "__main__":
    asyncio.run(main())
