"""Per-call extras that cannot live on the frozen :class:`AgentCallSpec`.

Tools, hooks, and event observers are live objects. The spec stays a
pure data envelope; this bag rides the same ``AgentGateway.call`` method
so there is still one model-call entry.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.agent.loops.hooks import LoopHooks
    from molexp.agent.session import Session

__all__ = ["AgentCallRuntime"]


@dataclass(frozen=True, slots=True)
class AgentCallRuntime:
    """Runtime co-arguments for one :meth:`AgentGateway.call`.

    On ``call_mode="agentic"`` these tools/hooks are forwarded to
    ``Router.stream_agentic``. Chat (``call_mode="structured"``) ignores them.
    """

    tools: tuple[object, ...] = ()
    hooks: LoopHooks | None = None
    on_event: Callable[[object], Awaitable[None]] | None = None
    system_prompt: str = ""
    context_block: str = ""
    workspace_root: Path | None = None
    operation_mode: str = "chat"
    session: Session | None = None
    extra: dict[str, str] = field(default_factory=dict)
