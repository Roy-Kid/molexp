"""The ``SessionEntry`` discriminated union.

A :data:`SessionEntry` is one append-only record in the session tree.
Every member carries ``id``, ``parent_id`` (``None`` only for the
tree root), ``timestamp``, and an ``entry_kind`` discriminator. The
tree is reconstructed by following ``parent_id`` pointers.

Six member kinds:

- :class:`MessageEntry` — one conversation turn (wraps a
  :class:`~molexp.agent.types.Message`).
- :class:`CompactionEntry` — a context-compaction cut: ``summary``
  replaces every entry from the root up to (excluding)
  ``first_kept_entry_id``.
- :class:`ModelChangeEntry` — records a model swap mid-session.
- :class:`StageEntry` — a molexp-specific orchestration stage marker.
- :class:`ArtifactEntry` — a molexp-specific artefact record.
- :class:`ApprovalEntry` — a molexp-specific approval-gate verdict.

Pure frozen-pydantic data; no LLM, no I/O.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from molexp.agent.types import Message, utc_now

__all__ = [
    "ApprovalEntry",
    "ArtifactEntry",
    "CompactionEntry",
    "MessageEntry",
    "ModelChangeEntry",
    "SessionEntry",
    "StageEntry",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class _BaseEntry(BaseModel):
    """Common shape every :data:`SessionEntry` member shares."""

    model_config = _FROZEN

    id: str
    parent_id: str | None
    timestamp: datetime = Field(default_factory=utc_now)


class MessageEntry(_BaseEntry):
    """One conversation turn in the session tree."""

    entry_kind: Literal["message"] = "message"
    message: Message


class CompactionEntry(_BaseEntry):
    """A context-compaction cut point.

    Attributes:
        summary: The LLM-produced summary that stands in for every
            entry preceding the cut.
        first_kept_entry_id: ``id`` of the first entry *retained* after
            the cut. Entries from the root up to (but excluding) this
            one are represented by ``summary``.
        tokens_before: Estimated token count of the summarized span.
    """

    entry_kind: Literal["compaction"] = "compaction"
    summary: str
    first_kept_entry_id: str
    tokens_before: int


class ModelChangeEntry(_BaseEntry):
    """Records a model swap mid-session."""

    entry_kind: Literal["model_change"] = "model_change"
    from_model: str
    to_model: str


class StageEntry(_BaseEntry):
    """A molexp-specific orchestration-stage marker."""

    entry_kind: Literal["stage"] = "stage"
    stage_name: str
    completed: bool = False


class ArtifactEntry(_BaseEntry):
    """A molexp-specific artefact record."""

    entry_kind: Literal["artifact"] = "artifact"
    path: str
    description: str = ""


class ApprovalEntry(_BaseEntry):
    """A molexp-specific approval-gate verdict record."""

    entry_kind: Literal["approval"] = "approval"
    gate: str
    approved: bool
    reason: str = ""


SessionEntry = Annotated[
    MessageEntry | CompactionEntry | ModelChangeEntry | StageEntry | ArtifactEntry | ApprovalEntry,
    Field(discriminator="entry_kind"),
]
"""Discriminated union of every session-tree record kind."""
