"""``AgentRunner`` + ``ChatMode`` — the minimum viable agent loop.

Demonstrates the entire public agent surface: ``AgentRunner``, ``ChatMode``,
``AgentSession``, and the ``AgentRunResult`` it returns.

Uses ``pydantic_ai.models.test.TestModel`` so the example is reproducible
without an API key. To talk to a real provider, replace the harness setup
with a model string like ``"openai:gpt-4o-mini"`` and set ``OPENAI_API_KEY``.

Run directly::

    python examples/agent/chat_mode.py
"""

from __future__ import annotations

import asyncio

from pydantic_ai.models.test import TestModel

from molexp.agent import AgentSession
from molexp.agent._pydanticai.harness import PydanticAIHarness
from molexp.agent.modes import ChatMode, ChatModeConfig


async def main() -> None:
    # The harness is normally constructed by ``AgentRunner`` from the
    # model string passed to its constructor.  Here we drive it directly
    # so the example doesn't depend on a real provider.
    mode = ChatMode(config=ChatModeConfig(system_prompt="you are concise"))
    harness = PydanticAIHarness(model=TestModel())  # offline, deterministic.

    session = AgentSession()
    result = await mode.run(harness=harness, session=session, user_input="ping")

    print(f"text:           {result.text}")
    print(f"history turns:  {len(result.messages)}")
    for msg in result.messages:
        print(f"  [{msg.role:<9}] {msg.content[:60]}")


if __name__ == "__main__":
    asyncio.run(main())
