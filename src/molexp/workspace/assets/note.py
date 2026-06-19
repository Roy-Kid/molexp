"""NoteAsset — a markdown note registered in the catalog.

A note is a user- (or agent-) authored markdown document living under a
scope's ``library/notes/<slug>.md``.  Unlike :class:`DataAsset`, a note has
no ``assets/<id>/payload/`` directory — the file is registered *in place*
beneath ``library/``, which keeps a human-meaningful filename and lets the
markdown preview / agent ``read_file`` reach it by its natural path.

The owning :class:`~molexp.workspace.library.library.Library` writes the
``.md`` file, then records the ``NoteAsset`` in the scope's authoritative
``assets.json`` manifest (and the derived catalog).
"""

from __future__ import annotations

from typing import Literal

from .base import Asset


class NoteAsset(Asset):
    """A markdown note tied to a workspace/project/experiment/run scope.

    ``path`` is scope-relative (``library/notes/<slug>.md``).  ``summary``
    is a one-line gloss surfaced in the generated library index so the
    agent can triage notes without reading each body; ``refs`` lists the
    citation keys (into the scope's ``references.json``) the note cites.
    """

    kind: Literal["note"] = "note"
    title: str
    summary: str = ""
    refs: tuple[str, ...] = ()
