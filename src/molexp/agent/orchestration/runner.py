"""AgentRunner — single-turn pipeline (spec §6.3).

Phase 1a only needs the surface so other layers can reference it.
Behavioral implementation arrives in Phase 1b/1c; this stub raises
``NotImplementedError`` from the entry points so accidental early use
is loud.
"""

from __future__ import annotations

from dataclasses import dataclass

from molexp.agent.context.manager import ContextManager, DefaultContextManager
from molexp.agent.model import ModelClient
from molexp.agent.tools.dispatcher import ToolDispatcher
from molexp.agent.tools.registry import ToolRegistry


@dataclass
class AgentRunner:
    """Run one turn end-to-end: context → model → tools → state.

    Phase 1a placeholder. Phase 1b implements the no-tool path,
    Phase 1c lights up the tool dispatch loop.
    """

    model: ModelClient
    registry: ToolRegistry
    dispatcher: ToolDispatcher
    context_manager: ContextManager

    @classmethod
    def with_defaults(
        cls,
        model: ModelClient,
        registry: ToolRegistry,
        dispatcher: ToolDispatcher | None = None,
    ) -> "AgentRunner":
        return cls(
            model=model,
            registry=registry,
            dispatcher=dispatcher or ToolDispatcher(registry),
            context_manager=DefaultContextManager(),
        )

    async def run_turn(self, *args, **kwargs):
        raise NotImplementedError(
            "AgentRunner.run_turn is implemented in Phase 1b; the Phase 0 "
            "skeleton only provides the surface."
        )
