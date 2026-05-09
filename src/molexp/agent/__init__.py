"""Public agent surface.

The entire ``molexp.agent`` layer is rebuilt around four user-visible
names: :class:`AgentRunner`, :class:`AgentMode`, :class:`AgentRunResult`,
:class:`AgentSession`. Concrete modes (``PlanMode``, ``ChatMode``,
``ReviewMode``) live under :mod:`molexp.agent.modes`.

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

See ``§ Layer charters`` in CLAUDE.md and the import-guard tests
under ``tests/test_agent/`` for the binding rules.
"""

from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.runner import AgentRunner
from molexp.agent.session import AgentSession

__all__ = [
    "AgentMode",
    "AgentRunResult",
    "AgentRunner",
    "AgentSession",
]
