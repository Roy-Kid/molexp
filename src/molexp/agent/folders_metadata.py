"""Agent-layer ``meta.yaml`` models — ``ConceptMeta`` subtypes.

Rehomed onto ``molexp.knowledge`` (the OKF rewrite): ``Agent`` / ``AgentSession``
are knowledge Concepts, so their structured metadata is a ``ConceptMeta``
subtype (the okf-07 ``ReferenceMeta`` pattern), persisted to each Concept's
``meta.yaml``. The pydantic-ai ``ModelMessage`` history stays in a sibling
``messages.jsonl`` (binary, via the lazy codec) — never inlined here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from molexp._typing import JSONValue
from molexp.knowledge import ConceptMeta

SessionStatusStr = Literal["pending", "running", "paused", "succeeded", "failed", "cancelled"]


class AgentMeta(ConceptMeta):
    """``meta.yaml`` payload for an :class:`~molexp.agent.folders.Agent` Concept."""

    type: str = "agent.agent"
    system_prompt: str = ""
    model: str = ""
    tier: str = ""
    description: str = ""


class AgentSessionMeta(ConceptMeta):
    """``meta.yaml`` payload for an :class:`~molexp.agent.folders.AgentSession`."""

    type: str = "agent.session"
    goal_summary: str = ""
    status: SessionStatusStr = "pending"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    extras: dict[str, JSONValue] = Field(default_factory=dict)


__all__ = ["AgentMeta", "AgentSessionMeta", "SessionStatusStr"]
