"""``NoteMeta`` -- structured ``meta.json`` payload for an OKF Note Concept.

A :class:`molexp.workspace.concepts.Note` stores lightweight document metadata
in ``meta.json`` via this :class:`ConceptMeta` subtype, while its narrative body
lives in ``index.md`` and its knowledge-graph edges are markdown links. ``tags``
is inherited from :class:`ConceptMeta` (default empty = untagged); ``status`` is
added here (default ``"active"``).

The ``type`` defaults to the dotted concept type ``"note.note"`` -- the same
value :class:`~molexp.workspace.concepts.Note` stamps on write, so
:func:`molexp.workspace.folder.concept_from_dir` rebuilds the right subclass. A
legacy bare marker (only ``{type, id}`` on disk) reads back with the additive
defaults (``tags == []`` / ``status == "active"``), so no migration is needed.
"""

from __future__ import annotations

from .concept_meta import ConceptMeta

NOTE_TYPE = "note.note"


class NoteMeta(ConceptMeta):
    """Document ``meta.json`` payload of a Note Concept.

    Inherits :class:`ConceptMeta` (frozen, ``extra="allow"``,
    ``from_json`` / ``to_json``, inherited ``tags``); ``type`` defaults
    to ``"note.note"``.

    Attributes:
        type: Concept discriminator (defaults to ``"note.note"``).
        status: Document lifecycle label (defaults to ``"active"``); e.g.
            ``"active"`` / ``"archived"`` / ``"draft"`` -- an open string, not a
            fixed enum.
    """

    type: str = NOTE_TYPE
    status: str = "active"


__all__ = ["NOTE_TYPE", "NoteMeta"]
