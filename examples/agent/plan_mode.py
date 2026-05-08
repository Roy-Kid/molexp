"""``AgentRunner`` + ``PlanMode`` — multi-step planning over a private graph.

Demonstrates the public agent surface for planning: ``AgentRunner`` drives
``PlanMode``, which internally walks a three-step pydantic-graph workflow
(intake -> design -> output) and surfaces the structured plan via
``AgentRunResult.mode_state["plan"]``.

Uses pydantic-ai's built-in ``"test"`` model identifier so the example runs
offline without an API key. To talk to a real provider, swap the model
string for one like ``"openai:gpt-4o-mini"`` and set ``OPENAI_API_KEY``.

Run directly::

    python examples/agent/plan_mode.py
"""

from __future__ import annotations

import asyncio

from molexp.agent import AgentRunner, AgentSession
from molexp.agent.modes import PlanMode


async def main() -> None:
    runner = AgentRunner(mode=PlanMode(max_iterations=4), model="test")

    session = AgentSession()
    result = await runner.run(
        session,
        "design a workflow that screens solvent conditions for a Suzuki coupling",
    )

    print(f"summary:        {result.text}")
    print(f"history turns:  {len(result.messages)}")
    for msg in result.messages:
        print(f"  [{msg.role:<9}] {msg.content[:60]}")

    plan = (result.mode_state or {}).get("plan", {})
    print("\nstructured plan sections:")
    for section, body in plan.items():
        print(f"  - {section}: {str(body)[:80]}")


if __name__ == "__main__":
    asyncio.run(main())
