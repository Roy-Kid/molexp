"""``AgentRuntime`` — frozen bundle one turn reaches for.

Session (conversation history) + router + execution sandbox.
``AgentRunner`` builds it once per :meth:`run`. A REPL is many turns on
the same :class:`~molexp.agent.session.Session`, not one call.
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
    """Thin, immutable bundle of services a turn reaches for at run time."""

    session: Session
    router: Router
    execution_env: ExecutionEnv
