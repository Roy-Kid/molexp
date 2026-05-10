"""``AgentRunner`` + ``ChatMode`` — the minimum viable agent loop.

Demonstrates the entire public agent surface: ``AgentRunner``,
``ChatMode``, ``AgentSession``, and the ``AgentRunResult`` it returns.

Uses ``pydantic_ai.models.test.TestModel`` so the example is reproducible
without an API key. To talk to a real provider, replace ``model=`` with
a string like ``"openai:gpt-4o-mini"`` and set ``OPENAI_API_KEY``.

Run directly::

    python examples/agent/chat_mode.py
"""

from __future__ import annotations

import asyncio

from pydantic_ai.models.test import TestModel

from molexp.agent import AgentRunner, AgentSession
from molexp.agent.modes import ChatMode, ChatModeConfig


async def main() -> None:
    runner = AgentRunner(
        mode=ChatMode(config=ChatModeConfig(system_prompt="you are concise")),
        # ``TestModel`` is a pydantic-ai model object — accepted in lieu of
        # a model-id string. The runner normalizes this single value across
        # every tier so PlanMode-style routing also works in tests.
        model=TestModel(),  # type: ignore[arg-type]
    )

    session = AgentSession()
    result = await runner.run(session, "ping")

    print(f"text:           {result.text}")
    print(f"history turns:  {len(result.messages)}")
    for msg in result.messages:
        print(f"  [{msg.role:<9}] {msg.content[:60]}")


if __name__ == "__main__":
    asyncio.run(main())
