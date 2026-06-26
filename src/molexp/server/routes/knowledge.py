"""Knowledge routes — read-only browse of the workspace's OKF Concepts.

Notes (``Note``) and literature references (``ReferenceConcept``) are OKF
Concept ``Folder``s mounted anywhere under the workspace, reached through the
:class:`~molexp.workspace.bundle.Bundle` façade. These routes expose them over
HTTP for the UI's Knowledge tab; the legacy per-scope ``/api/library`` surface
was removed in wsokf-11, so this is the greenfield read API for OKF knowledge.

Read-only by design: authoring notes/imports happens through the workspace /
CLI, not the browser.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from molexp.server.dependencies import get_workspace

if TYPE_CHECKING:
    from molexp.workspace import Bundle, Workspace

__all__ = ["router"]

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

_EXCERPT_CHARS = 320


class NoteSummary(BaseModel):
    name: str
    relPath: str
    excerpt: str


class ReferenceSummary(BaseModel):
    name: str
    relPath: str
    title: str | None = None
    authors: list[str] = []
    year: int | None = None
    doi: str | None = None
    venue: str | None = None
    url: str | None = None
    source: str = "manual"


class KnowledgeListResponse(BaseModel):
    notes: list[NoteSummary]
    references: list[ReferenceSummary]
    total: int


class NoteDetailResponse(BaseModel):
    name: str
    relPath: str
    body: str
    links: list[str]


def _bundle(workspace: Workspace) -> Bundle:
    from molexp.workspace import Bundle

    return Bundle(workspace.root)


@router.get("", response_model=KnowledgeListResponse)
def list_knowledge(workspace: Workspace = Depends(get_workspace)) -> KnowledgeListResponse:
    """List every Note + ReferenceConcept in the active workspace's bundle."""
    bundle = _bundle(workspace)

    notes: list[NoteSummary] = []
    for note in bundle.notes():
        body = note.body() or ""
        notes.append(
            NoteSummary(
                name=note.name,
                relPath=bundle.rel_path(note),
                excerpt=body[:_EXCERPT_CHARS],
            )
        )

    references: list[ReferenceSummary] = []
    for ref in bundle.references():
        meta = ref.read_ref_meta()
        references.append(
            ReferenceSummary(
                name=ref.name,
                relPath=bundle.rel_path(ref),
                title=meta.title,
                authors=list(meta.authors),
                year=meta.year,
                doi=meta.doi,
                venue=meta.venue,
                url=meta.url,
                source=meta.source,
            )
        )

    notes.sort(key=lambda n: n.name)
    references.sort(key=lambda r: (r.year or 0, r.name), reverse=True)
    return KnowledgeListResponse(
        notes=notes, references=references, total=len(notes) + len(references)
    )


@router.get("/note", response_model=NoteDetailResponse)
def get_note(
    path: str = Query(..., description="The note Concept's bundle-relative path (its identity)."),
    workspace: Workspace = Depends(get_workspace),
) -> NoteDetailResponse:
    """Return one note's full body (its ``index.md``) + its outgoing links."""
    from molexp.workspace.concepts import Note
    from molexp.workspace.errors import ConceptNotFoundError

    bundle = _bundle(workspace)
    try:
        concept = bundle.get(path)
    except ConceptNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"note {path!r} not found") from exc
    if not isinstance(concept, Note):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"concept {path!r} is not a note")
    return NoteDetailResponse(
        name=concept.name,
        relPath=path,
        body=concept.body(),
        links=list(concept.out_edges()),
    )
