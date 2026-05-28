"""Public agent surface — pydantic-ai facade + LLM-only modes.

Post spec ``harness-as-mode-substrate-03b`` the agent layer is a
**pydantic-ai facade**: ``Session`` / ``Router`` / ``ExecutionEnv`` /
``HookRegistry`` primitives + two LLM-only modes (:class:`ChatMode` for
single round-trip, :class:`InteractiveMode` for the emergent tool loop).
Pipeline-style orchestration (Plan / Author / Run / Review) moved to
:mod:`molexp.harness`.

The user-visible surface is six names — :class:`AgentRunner`,
:class:`AgentMode`, :class:`AgentRunResult`, :class:`AgentRuntime`,
:class:`AgentSession` — plus three workflow-orthogonal approval
primitives (:class:`ReviewDecision`, the :data:`ReviewPolicy` callable
alias, and the bundled :func:`cli_ask` policy).

``AgentSession`` is the runtime conversation value — the
:class:`~molexp.agent.session.Session` entry-tree class re-exported
under the historical name. ``AgentRuntime`` is the frozen dataclass
bundle a mode reaches for at run time (session + router +
execution_env + hooks); ``AgentRunner`` constructs it once per
:meth:`run` and passes it through to the mode.

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

To register tools on :class:`ChatMode` (or any single-shot mode that
takes the runner's text path), build a :class:`pydantic_ai.tools.Tool`
or pass a bare async callable — pydantic-ai accepts both shapes
natively — then forward them through :class:`AgentRunner`::

    from pydantic_ai.tools import Tool
    from molexp.agent import AgentRunner
    from molexp.agent.modes import ChatMode


    async def echo(message: str) -> str:
        return message


    runner = AgentRunner(
        mode=ChatMode(),
        model="openai:gpt-5.2",
        tools=(Tool(echo), echo),  # Tool instance OR bare callable
    )

MCP servers do not go through ``tools=``; see
:mod:`molexp.agent._pydanticai` for that wiring.

See ``§ Architecture`` in CLAUDE.md and the import-guard tests under
``tests/test_agent/`` for the binding rules.
"""

from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.review import ReviewDecision, ReviewPolicy, cli_ask
from molexp.agent.runner import AgentRunner
from molexp.agent.runtime import AgentRuntime
from molexp.agent.session import Session as AgentSession

__all__ = [
    "AgentMode",
    "AgentRunResult",
    "AgentRunner",
    "AgentRuntime",
    "AgentSession",
    "ReviewDecision",
    "ReviewPolicy",
    "cli_ask",
]
