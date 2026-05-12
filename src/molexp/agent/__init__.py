"""Public agent surface.

The entire ``molexp.agent`` layer is rebuilt around four user-visible
mode-orchestration names — :class:`AgentRunner`, :class:`AgentMode`,
:class:`AgentRunResult`, :class:`AgentSession` — plus a small set of
workflow-orthogonal review primitives (:class:`ReviewPolicy`,
:class:`ReviewDecision`, :class:`ReviewView`, :class:`StepView`,
:class:`BypassPolicy`, :class:`AutoPolicy`, :class:`HumanPolicy`,
:func:`cli_ask`) that any mode with a multi-step workflow consumes.
Concrete modes (``PlanMode``, ``ChatMode``, ``ReviewMode``) live under
:mod:`molexp.agent.modes`.

The review module sits parallel to ``mode.py`` because the policies are
NOT mode-specific concepts — putting them under a single mode's
subpackage would force duplication or upward imports as soon as a
second workflow-bearing mode lands.  :class:`HumanPolicy` is UI-agnostic
by construction: the rendering surface is the ``ask`` callable
(:func:`cli_ask` is the default; web / Slack / mobile push are drop-in
replacements).

Layer position: **agent uses workflow + workspace**. The agent imports
the public surface of both downstream layers — ``Workspace`` /
``Run`` / ``AssetCatalog`` from workspace; ``Workflow`` /
``WorkflowSpec`` / ``Task`` / ``TaskContext`` from workflow. It does
not import any sibling application layer (``plugins`` / ``server`` /
``cli`` / ``sweep``); the agent stays a library.

Two SDKs sit behind import-boundary firewalls:

- ``pydantic_ai`` is a private implementation detail confined to
  :mod:`molexp.agent._pydanticai`. ``import molexp.agent`` does not
  eagerly load it; the harness is constructed lazily on first
  :meth:`AgentRunner.run`.
- ``pydantic_graph`` is **not** imported anywhere under ``agent/``.
  Multi-step modes (``PlanMode``) drive their workflows through the
  public ``molexp.workflow`` API, the sole sanctioned pg site in the
  project.

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

MCP servers do not go through ``tools=``. PlanMode's discovery agent
mounts them internally via ``Agent(toolsets=[MCPServerStdio(...)])`` —
see :mod:`molexp.agent._pydanticai` for that wiring.

See ``§ Layer charters`` in CLAUDE.md and the import-guard tests
under ``tests/test_agent/`` for the binding rules.
"""

from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.review import (
    AutoPolicy,
    BypassPolicy,
    HumanPolicy,
    ReviewDecision,
    ReviewPolicy,
    ReviewView,
    StepView,
    cli_ask,
)
from molexp.agent.runner import AgentRunner
from molexp.agent.session import AgentSession

__all__ = [
    "AgentMode",
    "AgentRunResult",
    "AgentRunner",
    "AgentSession",
    "AutoPolicy",
    "BypassPolicy",
    "HumanPolicy",
    "ReviewDecision",
    "ReviewPolicy",
    "ReviewView",
    "StepView",
    "cli_ask",
]
