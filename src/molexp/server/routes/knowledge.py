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

from pathlib import Path as _StdPath
from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from molexp.server.dependencies import get_workspace
from molexp.workspace.edges import EdgeRole

from ..deps.served import active_served_key, assert_workspace_writable
from ..schemas import MessageResponse

if TYPE_CHECKING:
    from molexp.workspace import Bundle, Workspace
    from molexp.workspace.assets.base import Asset
    from molexp.workspace.concepts import Note
    from molexp.workspace.experiment import Experiment
    from molexp.workspace.folder import Folder
    from molexp.workspace.run import Run

__all__ = ["router"]

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

_EXCERPT_CHARS = 320


class NoteSummary(BaseModel):
    name: str
    relPath: str
    excerpt: str
    tags: list[str] = []
    status: str | None = None


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


class EntityCard(BaseModel):
    """A clickable summary card for an entity a document embeds (05 resolver)."""

    kind: str
    id: str
    title: str
    relPath: str | None = None
    status: str | None = None


class NoteDetailResponse(BaseModel):
    name: str
    relPath: str
    body: str
    links: list[str]
    cards: list[EntityCard] = []


class EmbedRequest(BaseModel):
    """Embed a live workspace entity into a document as one typed provenance edge."""

    target_kind: Literal["run", "asset", "experiment", "reference"]
    target: str
    # ``None`` = defer to ``Bundle.embed``'s per-kind default (run/experiment ->
    # records, reference -> cites, asset/other -> references), so an HTTP embed
    # with no explicit role writes the SAME edge a CLI ``Bundle.embed`` call
    # would — the Python==UI invariant. An explicit role overrides.
    role: EdgeRole | None = None
    text: str | None = None


class EmbedResponse(BaseModel):
    """Echo of the written embed edge."""

    srcPath: str
    target: str
    role: EdgeRole


class DocCreateRequest(BaseModel):
    name: str
    body: str = ""
    parentPath: str | None = None


class DocBodyUpdate(BaseModel):
    body: str


class DocMoveRequest(BaseModel):
    name: str | None = None
    parentPath: str | None = None


class DocMetaUpdate(BaseModel):
    """Partial update of a note's ``meta.yaml`` tags/status.

    Each field is independently optional; ``None`` means "leave untouched", which
    maps onto ``Note.set_tags`` / ``Note.set_status`` each preserving the sibling
    field. A request with both ``None`` is a no-op that returns the current summary.
    """

    tags: list[str] | None = None
    status: str | None = None


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
    """Build a :class:`NoteSummary` for *note* (identity + excerpt + tags/status).

    ``tags``/``status`` come from the 05 :class:`~molexp.workspace.note_meta.NoteMeta`
    helpers (an untagged note reads back ``[]`` / ``"active"``).
    """
    body = note.body() or ""
    return NoteSummary(
        name=note.name,
        relPath=bundle.rel_path(note),
        excerpt=body[:_EXCERPT_CHARS],
        tags=note.tags(),
        status=note.status(),
    )


def _note_detail(bundle: Bundle, workspace: Workspace, note: Note, path: str) -> NoteDetailResponse:
    """Build a :class:`NoteDetailResponse` for *note* at its identity *path*.

    Enriches the response with ``cards`` — one :class:`EntityCard` per typed
    out-edge that resolves to a live entity (Run / Experiment / Reference / Note
    / Asset), each projected by the 05 entity-summary resolver
    (:meth:`~molexp.workspace.bundle.Bundle.entity_summary`).
    """
    return NoteDetailResponse(
        name=note.name,
        relPath=path,
        body=note.body(),
        links=list(note.out_edges()),
        cards=_resolve_cards(bundle, workspace, note),
    )


def _resolve_cards(bundle: Bundle, workspace: Workspace, note: Note) -> list[EntityCard]:
    """Resolve *note*'s typed out-edges to :class:`EntityCard` summary cards.

    Each edge target is resolved back to its live entity; an edge that resolves
    to no entity is skipped (it cannot be summarized as a card).
    """
    cards: list[EntityCard] = []
    for edge in note.typed_out_edges():
        entity = _resolve_edge_entity(bundle, workspace, edge.target)
        if entity is None:
            continue
        summary = bundle.entity_summary(entity)
        cards.append(
            EntityCard(
                kind=summary.kind,
                id=summary.id,
                title=summary.title,
                relPath=_entity_rel_path(bundle, entity),
                status=_entity_status(entity),
            )
        )
    return cards


def _resolve_edge_entity(
    bundle: Bundle, workspace: Workspace, target: str
) -> Folder | Asset | None:
    """Resolve a typed out-edge *target* path back to its live entity, or ``None``.

    A Concept dir (``meta.yaml`` present) resolves to its typed ``Folder`` via
    the bundle; an asset record dir (``<scope>/assets/<asset_id>/``) resolves to
    its :class:`~molexp.workspace.assets.base.Asset` via the manifest scanner.
    """
    from molexp.workspace.assets.scan import get_asset
    from molexp.workspace.errors import ConceptNotFoundError

    abs_path = _StdPath(str(target))
    try:
        rel = abs_path.relative_to(bundle.root).as_posix()
    except ValueError:
        return None
    try:
        return bundle.get(rel)
    except ConceptNotFoundError:
        pass
    if abs_path.parent.name == "assets":
        return get_asset(workspace.root, abs_path.name)
    return None


def _entity_rel_path(bundle: Bundle, entity: Folder | Asset) -> str | None:
    """A ``Folder`` entity's bundle-relative identity path; ``None`` for an ``Asset``."""
    from molexp.workspace.assets.base import Asset

    if isinstance(entity, Asset):
        return None
    return bundle.rel_path(entity)


def _entity_status(entity: Folder | Asset) -> str | None:
    """A ``Note`` entity's lifecycle status; ``None`` for anything else."""
    from molexp.workspace.concepts import Note

    return entity.status() if isinstance(entity, Note) else None


def _resolve_embed_target(
    bundle: Bundle, workspace: Workspace, target_kind: str, target: str
) -> Folder | Asset:
    """Resolve an embed ``(target_kind, target)`` to a live ``Folder`` / ``Asset``.

    Mirrors :func:`_resolve_note`'s not-found → 404 policy: an unknown run /
    experiment / asset / reference (or a target that resolves to no entity) is a
    404, never a silent miss.
    """
    from molexp.workspace.errors import ConceptNotFoundError

    if target_kind == "reference":
        try:
            return bundle.get(target)
        except ConceptNotFoundError as exc:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"reference {target!r} not found"
            ) from exc

    entity: Folder | Asset | None
    if target_kind == "run":
        entity = _find_run(workspace, target)
    elif target_kind == "experiment":
        entity = _find_experiment(workspace, target)
    else:  # "asset" — the Literal on EmbedRequest guarantees no other value
        from molexp.workspace.assets.scan import get_asset

        entity = get_asset(workspace.root, target)
    if entity is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{target_kind} {target!r} not found")
    return entity


def _find_run(workspace: Workspace, run_id: str) -> Run | None:
    """Find a ``Run`` by id across every project/experiment, or ``None``."""
    for project in workspace.list_projects():
        for experiment in project.list_experiments():
            if experiment.has_run(run_id):
                return experiment.get_run(run_id)
    return None


def _find_experiment(workspace: Workspace, experiment_id: str) -> Experiment | None:
    """Find an ``Experiment`` by id across every project, or ``None``."""
    for project in workspace.list_projects():
        if project.has_experiment(experiment_id):
            return project.get_experiment(experiment_id)
    return None


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
def list_knowledge(
    tag: Annotated[str | None, Query(description="Only notes carrying this tag.")] = None,
    status: Annotated[
        str | None, Query(description="Only notes with this lifecycle status.")
    ] = None,
    workspace: Workspace = Depends(get_workspace),
) -> KnowledgeListResponse:
    """List every Note + ReferenceConcept in the active workspace's bundle.

    Optional ``tag`` / ``status`` query params AND-narrow the note list (both
    read from the 05 :class:`~molexp.workspace.note_meta.NoteMeta` fields).
    """
    bundle = _bundle(workspace)

    notes: list[NoteSummary] = []
    for note in bundle.notes():
        if tag is not None and tag not in note.tags():
            continue
        if status is not None and note.status() != status:
            continue
        notes.append(_note_summary(bundle, note))

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
    """Return one note's full body (its ``index.md``) + its outgoing links + cards."""
    bundle = _bundle(workspace)
    concept = _resolve_note(bundle, path)
    return _note_detail(bundle, workspace, concept, path)


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


@router.post(
    "/doc/embed",
    response_model=EmbedResponse,
    dependencies=[Depends(_require_writable)],
)
def embed_doc(
    body: EmbedRequest,
    path: str = Query(..., description="The source note Concept's bundle-relative path."),
    workspace: Workspace = Depends(get_workspace),
) -> EmbedResponse:
    """Embed a live entity into a document — delegates to ``Bundle.embed``.

    Resolves the source ``Note`` (404 on miss / non-note) and the target entity
    (``run`` / ``experiment`` / ``asset`` / ``reference``; 404 on miss), then
    writes ONE typed provenance edge via ``Bundle.embed`` — the same verb the CLI
    uses, so the edge-writing logic is never re-built at the HTTP boundary.
    """
    from molexp.workspace.doc_embed import default_role_for

    bundle = _bundle(workspace)
    note = _resolve_note(bundle, path)
    target = _resolve_embed_target(bundle, workspace, body.target_kind, body.target)
    # Resolve the effective role the same way ``Bundle.embed`` will, so the
    # echoed response reports the edge that was actually written.
    role = body.role if body.role is not None else default_role_for(target)
    bundle.embed(note, target, role=role)
    return EmbedResponse(srcPath=path, target=body.target, role=role)


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
    return _note_detail(bundle, workspace, note, path)


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


@router.patch(
    "/doc/meta",
    response_model=NoteSummary,
    dependencies=[Depends(_require_writable)],
)
def update_doc_meta(
    payload: DocMetaUpdate,
    path: str = Query(..., description="The note Concept's bundle-relative path (its identity)."),
    workspace: Workspace = Depends(get_workspace),
) -> NoteSummary:
    """Update a note's tags/status — delegates to ``Note.set_tags`` / ``Note.set_status``.

    Each field is applied only when present (``None`` = leave untouched), so a
    partial update preserves the sibling field. The write logic is never re-built
    here: the same ``Note`` verbs the CLI uses own it (the Python==UI invariant).
    """
    bundle = _bundle(workspace)
    note = _resolve_note(bundle, path)
    if payload.tags is not None:
        note.set_tags(payload.tags)
    if payload.status is not None:
        note.set_status(payload.status)
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
