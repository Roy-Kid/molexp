"""``ReferenceMeta`` — structured bib fields for an OKF Reference Concept (wsokf-05).

A :class:`molexp.workspace.concepts.ReferenceConcept` stores its bibliographic
record in ``meta.yaml`` via this :class:`ConceptMeta` subtype; the narrative
citation text lives in its ``index.md``. PDFs are *pointed at* (``pdf_path`` /
``pdf_asset_id``), never copied — mirroring the legacy bib-record
``molexp.workspace.library.Reference``.

This is a port of ``molexp.knowledge.references.ReferenceMeta`` onto the
workspace surface, subclassing the workspace-local
:class:`molexp.workspace.concept_meta.ConceptMeta` (no ``molexp.knowledge``
storage dependency).
"""

from __future__ import annotations

from .concept_meta import ConceptMeta


class ReferenceMeta(ConceptMeta):
    """Bibliographic ``meta.yaml`` payload of a Reference Concept.

    Inherits :class:`ConceptMeta` (frozen, ``extra="allow"``, ``from_yaml`` /
    ``to_yaml``); ``type`` defaults to ``"reference"``.

    Attributes:
        type: Concept discriminator (defaults to ``"reference"``).
        title: Work title.
        authors: Ordered author names.
        year: Publication year.
        doi: Digital Object Identifier.
        venue: Journal / conference / publisher.
        url: Canonical URL.
        pdf_path: Filesystem pointer to a PDF (never copied).
        pdf_asset_id: Asset-catalog id of an imported PDF, if any.
        source: Provenance label (``"manual"`` / ``"zotero"`` / …).
        source_key: Stable key in the external source (e.g. a Zotero key).
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
