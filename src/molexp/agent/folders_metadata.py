"""Agent-layer ``meta.yaml`` models — agent-owned, OKF-shaped.

After the OKF rehome onto ``molexp.workspace.Folder`` (wsokf-06), ``Agent`` /
``AgentSession`` are workspace Concepts whose settled, human-meaningful identity
*is* their ``meta.yaml`` payload. These models are agent-owned frozen pydantic
models in the OKF Concept-meta shape — a required ``type`` discriminator (the
string the shared concept-type registry resolves on) plus ``extra="allow"`` so
forward fields survive a round-trip. The pydantic-ai ``ModelMessage`` history
stays in a sibling ``messages.jsonl`` (binary, via the lazy codec) — never
inlined here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from molexp._typing import JSONValue

SessionStatusStr = Literal["pending", "running", "paused", "succeeded", "failed", "cancelled"]


class AgentMeta(BaseModel):
    """``meta.yaml`` payload for an :class:`~molexp.agent.folders.Agent` Concept.

    Attributes:
        type: The concept-type discriminator (``"agent.agent"``) the shared
            registry resolves back to :class:`~molexp.agent.folders.Agent`.
        id: The agent's slug (its directory name).
        system_prompt: The agent persona's system prompt.
        model: The model id this agent runs on (empty when unset).
        tier: The model tier label (empty when unset).
        description: A human-readable description.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    type: str = "agent.agent"
    id: str = ""
    system_prompt: str = ""
    model: str = ""
    tier: str = ""
    description: str = ""


class AgentSessionMeta(BaseModel):
    """``meta.yaml`` payload for an :class:`~molexp.agent.folders.AgentSession`.

    Attributes:
        type: The concept-type discriminator (``"agent.session"``) the shared
            registry resolves back to
            :class:`~molexp.agent.folders.AgentSession`.
        id: The session's slug (its directory name).
        goal_summary: A one-line summary of the conversation's goal.
        status: The session lifecycle state.
        started_at: When the session first ran (``None`` until it does).
        finished_at: When the session reached a terminal state.
        extras: Free-form forward-compatible metadata.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    type: str = "agent.session"
    id: str = ""
    goal_summary: str = ""
    status: SessionStatusStr = "pending"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    extras: dict[str, JSONValue] = Field(default_factory=dict)


__all__ = ["AgentMeta", "AgentSessionMeta", "SessionStatusStr"]
