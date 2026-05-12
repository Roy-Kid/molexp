"""``AgentRunner`` + ``ChatMode`` — multi-turn conversation with named sessions.

Demonstrates the entire public agent surface: ``AgentRunner``,
``ChatMode``, ``AgentSession`` (named, persisted across runs), and the
``AgentRunResult`` it returns.

Uses ``pydantic_ai.models.test.TestModel`` so the example is reproducible
without an API key. To talk to a real provider, replace ``model=`` with
a string like ``"openai:gpt-4o-mini"`` and set ``OPENAI_API_KEY``.

Two turns are issued on the same named session. Between them the
runner persists the pydantic-ai ``ModelMessage`` history under
``<workspace>/.subsystems/agent.sessions/<session_id>/model_messages.json``,
so the second call's request carries the full prior context — even
across runner instances or process restarts.

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


async def main() -> None:
    # Use a temporary workspace so the example does not pollute ``cwd``.
    # In real code, point this at the project workspace path so chat
    # sessions survive across restarts.
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "lab"
        workspace.mkdir()

        runner = AgentRunner(
            mode=ChatMode(config=ChatModeConfig(system_prompt="you are concise")),
            # ``TestModel`` is a pydantic-ai model object — accepted in lieu of
            # a model-id string. The runner normalizes this single value across
            # every tier so PlanMode-style routing also works in tests.
            model=TestModel(),  # type: ignore[arg-type]
            workspace=workspace,
        )

        # Named session: look up or create. With ``workspace=`` set, the
        # runner persists ``session.model_messages`` after each turn and
        # restores it on the next ``runner.session(same_id)`` call.
        session = runner.session("chat-with-roy")
        print(f"session_id:        {session.session_id}")
        print(f"prior messages:    {len(session.model_messages)}\n")

        result_a = await runner.run(session, "what is 2 + 2?")
        print(f"[turn 1]           {result_a.text[:60]}")
        print(f"history after t1:  {len(session.model_messages)} messages\n")

        # The model now sees turn 1 because we pass the same session.
        # The router forwards ``session.model_messages`` as ``message_history``.
        result_b = await runner.run(session, "and what about times two?")
        print(f"[turn 2]           {result_b.text[:60]}")
        print(f"history after t2:  {len(session.model_messages)} messages\n")

        # The next process / runner instance can resume by id.
        runner_resumed = AgentRunner(
            mode=ChatMode(config=ChatModeConfig(system_prompt="you are concise")),
            model=TestModel(),  # type: ignore[arg-type]
            workspace=workspace,
        )
        resumed = runner_resumed.session("chat-with-roy")
        print(f"resumed messages:  {len(resumed.model_messages)} (loaded from disk)")


if __name__ == "__main__":
    asyncio.run(main())
