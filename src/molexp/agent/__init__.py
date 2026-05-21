"""Public agent surface.

The ``molexp.agent`` layer is built around four user-visible
mode-orchestration names — :class:`AgentRunner`, :class:`AgentMode`,
:class:`AgentRunResult`, :class:`AgentSession` — plus three
workflow-orthogonal approval primitives (:class:`ReviewDecision`, the
:data:`ReviewPolicy` callable alias, and the bundled :func:`cli_ask`
policy) that wire into the harness's ``before_approval`` hook.

``AgentSession`` is the runtime conversation value — it is the
harness's :class:`~molexp.agent.harness.session.Session` entry-tree
class (re-exported under the historical name). The shared agent-runtime
layer it sits on lives under :mod:`molexp.agent.harness`
(:class:`~molexp.agent.harness.harness.AgentHarness`, the
:data:`~molexp.agent.harness.events.AgentEvent` stream, the
:class:`SessionStorage` repository, context compaction, and
:class:`~molexp.agent.harness.execution_env.ExecutionEnv`). The
reference mode :class:`~molexp.agent.modes.ChatMode` lives under
:mod:`molexp.agent.modes`; the four pipeline modes (Plan / Author /
Run / Review) are rebuilt in later specs.

The review module sits parallel to ``mode.py`` because approval is NOT
a mode-specific concept — every mode that reaches an
:class:`~molexp.agent.modes._planning.ApprovalGate` consults the same
:data:`ReviewPolicy`. A policy is just an async
``(gate, summary) -> ReviewDecision`` callable; :func:`cli_ask` is the
bundled CLI implementation, and web / Slack / mobile push are drop-in
replacements with the same signature.

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
mounts them internally via ``Agent(toolsets=[MCPToolset(...)])`` —
see :mod:`molexp.agent._pydanticai` for that wiring.

See ``§ Layer charters`` in CLAUDE.md and the import-guard tests
under ``tests/test_agent/`` for the binding rules.
"""

from molexp.agent.harness.session import Session as AgentSession
from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.review import ReviewDecision, ReviewPolicy, cli_ask
from molexp.agent.runner import AgentRunner

__all__ = [
    "AgentMode",
    "AgentRunResult",
    "AgentRunner",
    "AgentSession",
    "ReviewDecision",
    "ReviewPolicy",
    "cli_ask",
]
