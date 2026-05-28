"""Approval primitives for the agent layer.

The harness fires a ``before_approval`` hook at every approval gate (the
``ApprovalGate`` stage under ``molexp.harness.stages``); a handler returns a
:class:`ReviewDecision`. :class:`~molexp.agent.runner.AgentRunner`'s
``approval=`` argument wires a :data:`ReviewPolicy` callable into that
hook — :func:`cli_ask` is the bundled interactive implementation.

A policy is just an async callable ``(gate, summary) -> ReviewDecision``;
there are no policy classes. ``approval=None`` registers no hook and the
harness auto-approves; ``approval=cli_ask`` prompts the operator; any
custom ``async def (gate, summary)`` drives the decision programmatically.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict

__all__ = ["ReviewDecision", "ReviewPolicy", "cli_ask"]


class ReviewDecision(BaseModel):
    """Outcome of one approval-gate consultation.

    Attributes:
        approved: ``True`` to clear the gate; ``False`` to reject it.
        reason: One-sentence justification, surfaced to logs and the
            ``approval_decided`` event.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    approved: bool
    reason: str = ""


# A policy decides one gate: given the gate name + a one-line summary it
# returns a ReviewDecision. AgentRunner adapts it into a before_approval
# hook handler; the harness collects the decision (first denial wins).
ReviewPolicy = Callable[[str, str], Awaitable[ReviewDecision]]


async def cli_ask(gate: str, summary: str) -> ReviewDecision:
    """Bundled CLI approval prompt — the default ``approval=`` policy.

    Prints the gate name and its one-line summary, then reads a ``y/n``
    verdict from stdin. Anything other than ``y`` / ``yes`` rejects.

    The blocking ``print``/``input`` work is offloaded via
    :func:`asyncio.to_thread` so the operator's deliberation never stalls
    the event loop driving the run.

    Args:
        gate: The approval-gate name (e.g. ``"approve_direction"``).
        summary: One-line description of what is being approved.

    Returns:
        The operator's :class:`ReviewDecision`.
    """
    return await asyncio.to_thread(_prompt, gate, summary)


def _prompt(gate: str, summary: str) -> ReviewDecision:
    """Synchronous ``print``/``input`` body for :func:`cli_ask`."""
    print()
    print("=" * 72)
    print(f"APPROVAL REQUIRED — {gate}")
    print("=" * 72)
    if summary:
        print(f"  {summary}")
    raw = input("  approve? [y/N]: ").strip().lower()
    approved = raw in ("y", "yes")
    return ReviewDecision(
        approved=approved,
        reason="approved at CLI" if approved else "rejected at CLI",
    )
