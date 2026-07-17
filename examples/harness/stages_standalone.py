"""Harness building blocks — ToolCapability, capability registration, and the AgentGateway pattern.

Matches ``docs/architecture/plan-mode.md``.

Demonstrates:

1. ``ToolCapability`` — the schema for one harness-invokable capability.
2. In-memory capability registry — built from a list of grounded capabilities.
3. The ``AgentGateway`` pattern — the Protocol that bridges harness → agent
   (offline ``CannedGateway`` for LLM-free testing, same pattern used by
   ``examples/harness/experiment_pipeline.py``).

Run directly::

    python examples/harness/stages_standalone.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import molexp as me


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="molexp-harness-"))
    me.Workspace(root, name="harness-demo")

    # ── 1. ToolCapability — the schema for one harness-invokable capability ──
    from molexp.harness.schemas.capability import ToolCapability

    caps = [
        ToolCapability(
            id="packmol",
            package="molpack",
            name="Packmol",
            description="Pack molecules into a simulation box using Packmol",
            input_schema={},
            output_schema={},
            source="molmcp",
        ),
        ToolCapability(
            id="lj_cut",
            package="lammps",
            name="LJ/Cut Pair Style",
            description="Lennard-Jones potential with a cutoff",
            input_schema={},
            output_schema={},
            source="lammps",
        ),
    ]

    # ── 2. In-memory capability registry ───────────────────────────────────
    from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry

    registry = InMemoryCapabilityRegistry(capabilities=caps)
    print(f"\nRegistry entries: {len(registry.list_capabilities())}")

    # Lookup by id
    found = registry.get("packmol")
    print(f"get('packmol'): {found.id} — {found.description}")

    # Search
    results = registry.search("potential")
    print(f"search('potential'): {len(results)} match(es)")

    # ── 3. The CannedGateway pattern — offline harness testing ────────────
    # In production, RouterBackedAgentGateway drives agent.router.Router.
    # Here we show the mock gateway Protocol shape used throughout tests
    # and offline examples (the same pattern as experiment_pipeline.py).
    class CannedGateway:
        """Mock ``AgentGateway`` for offline harness-stage testing."""

        async def plan(self, *, system_prompt: str, user_prompt: str) -> str:
            return f"Canned plan response for: {user_prompt[:40]}..."

        async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            return "def build_workflow(): ..."

        async def review(self, *, system_prompt: str, user_prompt: str) -> dict:
            return {"approved": True, "issues": []}

    gateway = CannedGateway()
    plan = await gateway.plan(
        system_prompt="you are a research planner",
        user_prompt="design an NVT equilibration experiment for a polymer melt",
    )
    print(f"\nCannedGateway.plan() → {plan[:70]}...")
    print("Done — harness building blocks demonstrated.")


if __name__ == "__main__":
    asyncio.run(main())
