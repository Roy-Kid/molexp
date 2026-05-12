"""Frozen pydantic metadata models for agent-layer ``Folder`` subclasses.

The agent layer defines its own on-disk entities (``Agent`` /
``AgentSession`` in :mod:`molexp.agent.folders`) as subclasses of
:class:`molexp.workspace.Folder`. Each carries an
*entity-shaped* metadata model that extends :class:`FolderMetadata`
with the agent-specific fields workspace does **not** know about
(workspace stores their JSON as opaque dicts).

Sub-spec ``unify-folder-abstraction-03`` § Design § 4 introduces these
models so the agent layer can persist Agent personas + conversation
state without re-implementing the storage primitive.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from molexp._typing import JSONValue
from molexp.workspace import FolderMetadata


def _utc_now() -> datetime:
    return datetime.now()


SessionStatusStr = Literal["pending", "running", "paused", "succeeded", "failed", "cancelled"]


class AgentMetadata(FolderMetadata):
    """Persisted ``agent.json`` payload for an :class:`Agent` folder.

    Extends :class:`FolderMetadata` with the agent persona fields the
    agent layer owns (workspace stores the JSON as opaque dicts).
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    system_prompt: str = ""
    model: str = ""
    tier: str = ""
    description: str = ""


class AgentSessionMetadata(FolderMetadata):
    """Persisted ``agent_session.json`` payload for an :class:`AgentSession`.

    Extends :class:`FolderMetadata` with conversation-shaped fields.
    Pydantic-ai ``ModelMessage`` history is kept in a sibling
    ``messages.jsonl`` (encoded via the lazy
    :mod:`molexp.agent._pydanticai.messages_codec`), NOT inlined here —
    metadata stays small.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    goal_summary: str = ""
    status: SessionStatusStr = "pending"
    started_at: datetime = Field(default_factory=_utc_now)
    finished_at: datetime | None = None
    # Opaque agent-layer custom state — additional projection columns the
    # session catalog may carry without forking the schema.
    extras: dict[str, JSONValue] = Field(default_factory=dict)


__all__ = [
    "AgentMetadata",
    "AgentSessionMetadata",
    "SessionStatusStr",
]
