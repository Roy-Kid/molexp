"""Library routes — per-scope notes + literature/references under ``/api/library``.

A scope's :class:`~molexp.workspace.library.library.Library` owns a ``library/``
directory of markdown notes (each a ``NoteAsset``) plus a molexp-native
``references.json``.  These routes expose the derived index (notes + references)
and let the UI add notes / references; note *content* is served by the existing
``/api/assets/{id}/content`` endpoint and rendered by the markdown preview.

The scope is selected by the optional ``project_id`` / ``experiment_id`` /
``run_id`` query params — the deepest one present wins; none → the workspace
scope.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from molexp.workspace import (
    ExperimentNotFoundError,
    ProjectNotFoundError,
    RunNotFoundError,
)
from molexp.workspace.library import Library, LibraryIndex, Reference
from molexp.workspace.library.zotero import ZoteroImportError

from ..dependencies import get_workspace
from .schemas_library import (
    AddNoteRequest,
    NoteResponse,
    UpdateNoteRequest,
    ZoteroImportRequest,
    ZoteroImportResponse,
)

router = APIRouter(prefix="/library", tags=["library"])


def _note_response(note) -> NoteResponse:  # noqa: ANN001
    return NoteResponse(
        asset_id=note.asset_id,
        title=note.title,
        path=note.path.as_posix(),
        summary=note.summary,
        tags=tuple(note.tags.keys()),
        refs=note.refs,
    )


def _resolve_library(
    workspace,  # noqa: ANN001
    project_id: str | None,
    experiment_id: str | None,
    run_id: str | None,
) -> Library:
    """Return the :class:`Library` for the deepest scope the ids select.

    Raises a typed 404 when any named project/experiment/run is missing.
    """
    try:
        if project_id is None:
            return workspace.library
        project = workspace.get_project(project_id)
        if experiment_id is None:
            return project.library
        experiment = project.get_experiment(experiment_id)
        if run_id is None:
            return experiment.library
        return experiment.get_run(run_id).library
    except (ProjectNotFoundError, ExperimentNotFoundError, RunNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("", response_model=LibraryIndex)
def get_library(
    project_id: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> LibraryIndex:
    """Return the notes + references index for the selected scope.

    Best-effort syncs first: loose ``*.md`` files (e.g. a ``README.md``) are
    discovered and the on-disk index is refreshed so the agent stays current.
    Falls back to a pure read if the workspace is not writable.
    """
    library = _resolve_library(workspace, project_id, experiment_id, run_id)
    try:
        return library.build_index()
    except OSError:
        return library.index()


@router.post("/notes", response_model=NoteResponse, status_code=201)
def add_note(
    body: AddNoteRequest,
    project_id: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> NoteResponse:
    """Create (or overwrite by slug) a markdown note in the scope's library."""
    library = _resolve_library(workspace, project_id, experiment_id, run_id)
    note = library.add_note(
        body.title,
        body.content,
        slug=body.slug,
        summary=body.summary,
        tags=body.tags,
        refs=body.refs,
    )
    library.build_index()
    return _note_response(note)


@router.put("/notes/{asset_id}", response_model=NoteResponse)
def update_note(
    asset_id: str,
    body: UpdateNoteRequest,
    project_id: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> NoteResponse:
    """Edit a note's markdown body (preserves title/summary/tags/refs)."""
    library = _resolve_library(workspace, project_id, experiment_id, run_id)
    try:
        note = library.update_note(asset_id, body.content)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    library.build_index()
    return _note_response(note)


@router.post("/references", response_model=Reference, status_code=201)
def add_reference(
    body: Reference,
    project_id: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> Reference:
    """Insert or replace a bibliographic record (idempotent on ``key``)."""
    library = _resolve_library(workspace, project_id, experiment_id, run_id)
    ref = library.add_reference(body)
    library.build_index()
    return ref


@router.delete("/references/{key}", status_code=204)
def delete_reference(
    key: str,
    project_id: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> None:
    """Remove a reference by citation key (404 if absent)."""
    library = _resolve_library(workspace, project_id, experiment_id, run_id)
    if not library.references.remove(key):
        raise HTTPException(404, f"no reference with key {key!r}")
    library.build_index()


@router.post("/index", response_model=LibraryIndex)
def rebuild_index(
    project_id: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> LibraryIndex:
    """Re-derive and persist ``index.json`` + ``INDEX.md`` for the scope."""
    return _resolve_library(workspace, project_id, experiment_id, run_id).build_index()


@router.post("/zotero/import", response_model=ZoteroImportResponse)
def import_zotero(
    body: ZoteroImportRequest,
    project_id: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> ZoteroImportResponse:
    """Link a local Zotero library: import its items as references (no copy).

    The Zotero ``storage/`` PDFs are *pointed at* via each reference's
    ``pdf_path``; molexp imports no bytes.
    """
    library = _resolve_library(workspace, project_id, experiment_id, run_id)
    try:
        refs = library.import_zotero(body.path)
    except ZoteroImportError as exc:
        raise HTTPException(400, str(exc)) from exc
    library.build_index()
    return ZoteroImportResponse(imported=len(refs), references=refs)


@router.get("/sources", response_model=list[dict])
def list_sources(
    project_id: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> list[dict]:
    """External libraries (e.g. Zotero) linked into the selected scope."""
    return _resolve_library(workspace, project_id, experiment_id, run_id).list_sources()
