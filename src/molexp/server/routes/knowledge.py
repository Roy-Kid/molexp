"""Knowledge routes — browse + author the workspace's OKF Concepts.

Notes (``Note``) and literature references (``ReferenceConcept``) are OKF
Concept ``Folder``s mounted anywhere under the workspace, reached through the
:class:`~molexp.workspace.bundle.Bundle` façade. These routes expose them over
HTTP for the UI's Knowledge tab; the legacy per-scope ``/api/library`` surface
was removed in wsokf-11, so this is the greenfield read API for OKF knowledge.

The mutating document endpoints (create / edit-body / rename-move / delete /
backlinks / export) are **thin delegators** to the workspace-owned ``Bundle``
verbs (``create_note`` / ``rename_note`` / ``move_note`` / ``delete_note`` /
``backlinks`` / ``export_markdown``) — CLI and server call the same verbs, so
the CRUD logic lives in one place (the Python==UI invariant), never re-built at
the HTTP boundary. Each mutating handler is gated by :func:`_require_writable`
(405 against a remote/read-only served workspace); every path-addressed handler
maps :class:`ConceptNotFoundError` (and a non-``Note`` concept) to a 404.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from molexp.server.dependencies import get_workspace

from ..deps.served import active_served_key, assert_workspace_writable
from ..schemas import MessageResponse

if TYPE_CHECKING:
    from molexp.workspace import Bundle, Workspace
    from molexp.workspace.concepts import Note

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


class DocCreateRequest(BaseModel):
    name: str
    body: str = ""
    parentPath: str | None = None


class DocBodyUpdate(BaseModel):
    body: str


class DocMoveRequest(BaseModel):
    name: str | None = None
    parentPath: str | None = None


class BacklinksResponse(BaseModel):
    backlinks: list[NoteSummary]


def _bundle(workspace: Workspace) -> Bundle:
    from molexp.workspace import Bundle

    return Bundle(workspace.root)


def _require_writable(request: Request) -> None:
    """Reject a mutating request against a remote (read-only) served workspace.

    The flat ``/knowledge`` router is not under the ``/workspaces/{ws}`` scoped
    router, so it carries its own write-gate: it 405s a mutating verb against a
    remote served workspace (via :func:`assert_workspace_writable`); a local /
    unmanaged workspace stays writable. Safe methods pass through untouched.
    """
    assert_workspace_writable(active_served_key() or "", request.method)


def _note_summary(bundle: Bundle, note: Note) -> NoteSummary:
    """Build a :class:`NoteSummary` for *note* (name + identity path + excerpt)."""
    body = note.body() or ""
    return NoteSummary(
        name=note.name,
        relPath=bundle.rel_path(note),
        excerpt=body[:_EXCERPT_CHARS],
    )


def _note_detail(note: Note, path: str) -> NoteDetailResponse:
    """Build a :class:`NoteDetailResponse` for *note* at its identity *path*."""
    return NoteDetailResponse(
        name=note.name,
        relPath=path,
        body=note.body(),
        links=list(note.out_edges()),
    )


def _resolve_note(bundle: Bundle, path: str) -> Note:
    """Resolve *path* to a :class:`Note`, mapping miss / non-note to a 404."""
    from molexp.workspace.concepts import Note
    from molexp.workspace.errors import ConceptNotFoundError

    try:
        concept = bundle.get(path)
    except ConceptNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"note {path!r} not found") from exc
    if not isinstance(concept, Note):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"concept {path!r} is not a note")
    return concept


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
    bundle = _bundle(workspace)
    concept = _resolve_note(bundle, path)
    return _note_detail(concept, path)


# ── Document authoring — thin delegators to workspace ``Bundle`` verbs ────────


@router.post(
    "/doc",
    response_model=NoteSummary,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(_require_writable)],
)
def create_doc(
    body: DocCreateRequest,
    workspace: Workspace = Depends(get_workspace),
) -> NoteSummary:
    """Create a :class:`Note` document — delegates to ``Bundle.create_note``."""
    bundle = _bundle(workspace)
    parent = _resolve_note(bundle, body.parentPath) if body.parentPath else None
    note = bundle.create_note(body.name, parent=parent, body=body.body)
    return _note_summary(bundle, note)


@router.put(
    "/doc",
    response_model=NoteDetailResponse,
    dependencies=[Depends(_require_writable)],
)
def edit_doc(
    payload: DocBodyUpdate,
    path: str = Query(..., description="The note Concept's bundle-relative path (its identity)."),
    workspace: Workspace = Depends(get_workspace),
) -> NoteDetailResponse:
    """Rewrite a note's body (its ``index.md``) — delegates to ``Note.set_body``."""
    bundle = _bundle(workspace)
    note = _resolve_note(bundle, path)
    note.set_body(payload.body)
    return _note_detail(note, path)


@router.patch(
    "/doc",
    response_model=NoteSummary,
    dependencies=[Depends(_require_writable)],
)
def move_doc(
    payload: DocMoveRequest,
    path: str = Query(..., description="The note Concept's bundle-relative path (its identity)."),
    workspace: Workspace = Depends(get_workspace),
) -> NoteSummary:
    """Rename and/or reparent a note — delegates to ``Bundle.rename_note`` / ``move_note``."""
    bundle = _bundle(workspace)
    note = _resolve_note(bundle, path)
    if payload.name is not None:
        bundle.rename_note(note, payload.name)
    if payload.parentPath is not None:
        bundle.move_note(note, _resolve_note(bundle, payload.parentPath))
    return _note_summary(bundle, note)


@router.delete(
    "/doc",
    response_model=MessageResponse,
    dependencies=[Depends(_require_writable)],
)
def delete_doc(
    path: str = Query(..., description="The note Concept's bundle-relative path (its identity)."),
    workspace: Workspace = Depends(get_workspace),
) -> MessageResponse:
    """Delete a note (its directory subtree) — delegates to ``Bundle.delete_note``."""
    bundle = _bundle(workspace)
    note = _resolve_note(bundle, path)
    bundle.delete_note(note)
    return MessageResponse(message=f"note {path!r} deleted")


@router.get("/backlinks", response_model=BacklinksResponse)
def get_backlinks(
    path: str = Query(..., description="The target Concept's bundle-relative path (its identity)."),
    workspace: Workspace = Depends(get_workspace),
) -> BacklinksResponse:
    """Return every Concept linking at *path* — delegates to ``Bundle.backlinks``."""
    from molexp.workspace.errors import ConceptNotFoundError

    bundle = _bundle(workspace)
    try:
        concept = bundle.get(path)
    except ConceptNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"concept {path!r} not found") from exc

    backlinks: list[NoteSummary] = []
    for link in bundle.backlinks(concept):
        source = link.source
        body = source.read_index() or ""
        backlinks.append(
            NoteSummary(
                name=source.name,
                relPath=bundle.rel_path(source),
                excerpt=body[:_EXCERPT_CHARS],
            )
        )
    return BacklinksResponse(backlinks=backlinks)


@router.get("/doc/export")
def export_doc(
    path: str = Query(..., description="The note Concept's bundle-relative path (its identity)."),
    workspace: Workspace = Depends(get_workspace),
) -> PlainTextResponse:
    """Export a note as portable Markdown — delegates to ``Bundle.export_markdown``."""
    bundle = _bundle(workspace)
    note = _resolve_note(bundle, path)
    markdown = bundle.export_markdown(note)
    filename = f"{note.name}.md"
    return PlainTextResponse(
        content=markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
