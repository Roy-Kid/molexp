"""``AgentRuntime`` — frozen bundle of live handles a mode reaches for.

Replaces the prior ``AgentHarness`` god-object. After spec
``harness-as-mode-substrate-03b``, ``molexp.agent`` only ships two
modes (:class:`~molexp.agent.loops.ChatLoop` + the emergent
:class:`~molexp.agent.loops.InteractiveLoop`), neither of which needs
stage brackets, unified-approval gates, or compaction wiring — those
concerns moved to ``molexp.harness``. What a mode does need is a typed
bundle of the three runtime services the agent layer still owns: the
:class:`~molexp.agent.session.Session` entry tree, the LLM dispatch
:class:`~molexp.agent.router.Router`, and the
:class:`~molexp.agent.execution_env.ExecutionEnv` subprocess sandbox.
``AgentRunner`` builds the bundle once per :meth:`run` and passes it
through to the mode; the mode forwards events to its
:class:`~molexp.agent.events.AsyncIteratorEventSink` directly, with no
intermediate runtime methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.agent.execution_env import ExecutionEnv
    from molexp.agent.router import Router
    from molexp.agent.session import Session

__all__ = ["AgentRuntime"]


@dataclass(frozen=True)
class AgentRuntime:
    """Thin, immutable bundle of services a mode reaches for at run time.

    Plain frozen dataclass — no methods. Modes interact with the three
    handles directly (``runtime.session.append_message(...)``,
    ``runtime.router.complete_text(...)``, etc.) rather than through a
    runtime facade. ``arbitrary_types_allowed=True`` is forbidden under
    ``src/molexp/agent/`` per the layer charter, so the bundle is a
    dataclass rather than a ``pydantic.BaseModel``.
    """

    session: Session
    router: Router
    execution_env: ExecutionEnv
