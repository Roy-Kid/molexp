"""Public agent surface — pydantic-ai facade + LLM-only loops.

Post spec ``harness-as-mode-substrate-03b`` the agent layer is a
**pydantic-ai facade**: ``Session`` / ``Router`` / ``ExecutionEnv``
primitives + two LLM-only loops (:class:`ChatLoop` for single
round-trip, :class:`InteractiveLoop` for the emergent tool loop).
Pipeline-style orchestration (Plan / Author / Run / Review) moved to
:mod:`molexp.harness`.

The user-visible surface is five names — :class:`AgentRunner`,
:class:`AgentLoop`, :class:`AgentRunResult`, :class:`AgentRuntime`,
:class:`AgentSession` — plus one **lazy** re-export:
``PydanticAIRouter``, the concrete :class:`~molexp.agent.router.Router`
impl living under the ``_pydanticai/`` firewall. It resolves through a
module-level ``__getattr__`` so plain ``import molexp.agent`` still does
not pull ``pydantic_ai`` into ``sys.modules``; the SDK loads only when
the attribute is actually touched (or on the first
:meth:`AgentRunner.run`). Spell it
``from molexp.agent import PydanticAIRouter`` — never import from
``_pydanticai`` directly.

``AgentSession`` is the runtime conversation value — the
:class:`~molexp.agent.session.Session` entry-tree class re-exported
under the historical name. ``AgentRuntime`` is the frozen dataclass
bundle a loop reaches for at run time (session + router +
execution_env); ``AgentRunner`` constructs it once per :meth:`run`
and passes it through to the loop.

Layer position: **agent uses workspace only**. The agent imports the
public surface of workspace (for ``Workspace`` / ``Folder`` / session
storage on disk). It **MUST NOT** import :mod:`molexp.workflow`,
:mod:`molexp.harness`, or any sibling application layer
(``plugins`` / ``server`` / ``cli`` / ``sweep``); the agent stays a
library *below* harness in the DAG. The harness imports agent via the
sanctioned ``agent.router`` Protocol edge — never the other way.

Two SDKs sit behind import-boundary firewalls:

- ``pydantic_ai`` is a private implementation detail confined to
  :mod:`molexp.agent._pydanticai`. ``import molexp.agent`` does not
  eagerly load it; the router is constructed lazily on first
  :meth:`AgentRunner.run`.
- ``pydantic_graph`` is **not** imported anywhere under ``agent/``.

Tool injection
==============

To register tools on :class:`ChatLoop` (or any single-shot loop that
takes the runner's text path), build a :class:`pydantic_ai.tools.Tool`
or pass a bare async callable — pydantic-ai accepts both shapes
natively — then forward them through :class:`AgentRunner`::

    from pydantic_ai.tools import Tool
    from molexp.agent import AgentRunner
    from molexp.agent.loops import ChatLoop


    async def echo(message: str) -> str:
        return message


    runner = AgentRunner(
        loop=ChatLoop(),
        model="openai:gpt-5.2",
        tools=(Tool(echo), echo),  # Tool instance OR bare callable
    )

MCP servers do not go through ``tools=``; see
:mod:`molexp.agent._pydanticai` for that wiring.

See ``§ Architecture`` in CLAUDE.md and the import-guard tests under
``tests/test_agent/`` for the binding rules.
"""

from typing import TYPE_CHECKING

from molexp.agent.loop import AgentLoop, AgentRunResult
from molexp.agent.runner import AgentRunner
from molexp.agent.runtime import AgentRuntime
from molexp.agent.session import Session as AgentSession

if TYPE_CHECKING:
    # Redundant alias = the sanctioned re-export idiom: type checkers see
    # the name without the SDK loading at runtime (served by __getattr__).
    from molexp.agent._pydanticai.router import PydanticAIRouter as PydanticAIRouter

__all__ = [
    "AgentLoop",
    "AgentRunResult",
    "AgentRunner",
    "AgentRuntime",
    "AgentSession",
]


def __getattr__(name: str) -> object:
    """Lazy re-export — keeps ``import molexp.agent`` pydantic-ai-free.

    ``PydanticAIRouter`` is the only attribute served here; everything
    else surfaces through the eager imports above. It is deliberately
    *not* in ``__all__`` (the five-name loop-orchestration contract is
    pinned by ``tests/test_agent/test_public_surface.py``), but it is
    the official public spelling for the concrete router. The SDK
    behind the ``_pydanticai/`` firewall loads on first attribute
    access, matching the laziness contract pinned by
    ``tests/test_agent/test_import_guard.py``.
    """
    if name == "PydanticAIRouter":
        from molexp.agent._pydanticai.router import PydanticAIRouter

        return PydanticAIRouter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
