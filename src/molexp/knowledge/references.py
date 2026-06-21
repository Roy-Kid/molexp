"""``ReferenceMeta`` — structured bib fields for a Reference Concept (okf-07-01).

A ``Reference`` Concept stores its bibliographic record in ``meta.yaml`` via
this :class:`ConceptMeta` subtype; the narrative citation text lives in its
``index.md``. PDFs are *pointed at* (``pdf_path`` / ``pdf_asset_id``), never
copied — mirroring ``molexp.workspace.library.Reference``.
"""

from __future__ import annotations

from .models import ConceptMeta


class ReferenceMeta(ConceptMeta):
    """Bibliographic ``meta.yaml`` payload of a Reference Concept.

    Inherits ``ConceptMeta`` (frozen, ``extra="allow"``, ``from_yaml`` /
    ``to_yaml``); ``type`` defaults to ``"reference"``.
    """

    type: str = "reference"
    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    venue: str | None = None
    url: str | None = None
    pdf_path: str | None = None
    pdf_asset_id: str | None = None
    source: str = "manual"
    source_key: str | None = None


__all__ = ["ReferenceMeta"]
