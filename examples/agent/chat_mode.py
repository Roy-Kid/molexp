"""``AgentRunner`` + ``ChatMode`` — multi-turn conversation with named sessions.

Demonstrates the public agent surface after the harness restructure:
``AgentRunner``, ``ChatMode``, the runtime ``AgentSession`` (the harness
``Session`` entry-tree), and the ``AgentRunResult`` ``runner.run``
returns.

``ChatMode`` is the minimal :class:`~molexp.agent.mode.AgentMode`: each
turn is one user input -> one LLM round-trip -> one
:class:`~molexp.agent.mode.AgentRunResult`. Every mode runs *on* an
:class:`~molexp.agent.runtime.AgentHarness` that ``AgentRunner``
builds and injects; ``run`` drains the mode's ``AgentEvent`` stream and
folds it into the terminal result.

Uses ``pydantic_ai.models.test.TestModel`` so the example is reproducible
without an API key. To talk to a real provider, replace ``model=`` with
a string like ``"openai:gpt-4o-mini"`` and set ``OPENAI_API_KEY``.

Two turns are issued on the same named session. With ``workspace=`` set,
the runner backs the session with a ``JsonlSessionStorage`` writing
``entries.jsonl`` under an ``AgentSession`` folder, so a later
``runner.session(same_id)`` — even in a fresh process — resumes the
prior conversation entry-tree.

Run directly::

    python examples/agent/chat_mode.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from pydantic_ai.models.test import TestModel

from molexp.agent import AgentRunner
from molexp.agent.modes import ChatMode, ChatModeConfig


def _make_runner(workspace: Path) -> AgentRunner:
    """Build a ChatMode runner backed by an on-disk workspace.

    ``TestModel`` is a pydantic-ai model *object* — accepted in lieu of a
    model-id string. The runner normalizes the single value across every
    tier (``cheap`` / ``default`` / ``heavy``).
    """
    return AgentRunner(
        mode=ChatMode(config=ChatModeConfig(system_prompt="you are concise")),
        model=TestModel(),  # type: ignore[arg-type]
        workspace=workspace,
    )


async def main() -> None:
    # A temporary workspace so the example does not pollute ``cwd``. In
    # real code, point this at the project workspace path so chat
    # sessions survive across restarts.
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "lab"
        workspace.mkdir()

        runner = _make_runner(workspace)

        # Named session: looked up or created. With ``workspace=`` set,
        # the entry-tree is persisted to ``entries.jsonl`` after every
        # turn and restored on the next ``runner.session(same_id)``.
        session = runner.session("chat-with-roy")
        print(f"session_id:        {session.session_id}")
        print(f"prior entries:     {len(session.path_to_root())}\n")

        result_a = await runner.run(session, "what is 2 + 2?")
        print(f"[turn 1]           {result_a.text[:60]}")
        print(f"events emitted:    {len(result_a.events)}")
        print(f"context after t1:  {len(session.build_context())} messages\n")

        # The model sees turn 1 because we pass the same session — the
        # mode rebuilds prior context from the entry-tree.
        result_b = await runner.run(session, "and what about times two?")
        print(f"[turn 2]           {result_b.text[:60]}")
        print(f"context after t2:  {len(session.build_context())} messages\n")

        # A fresh runner / process resumes the conversation by id.
        runner_resumed = _make_runner(workspace)
        resumed = runner_resumed.session("chat-with-roy")
        print(f"resumed entries:   {len(resumed.path_to_root())} (loaded from disk)")


if __name__ == "__main__":
    asyncio.run(main())
