"""Request/response schemas for the ``/api/library`` routes."""

from __future__ import annotations

from pydantic import BaseModel, Field

from molexp.workspace.library import Reference


class AddNoteRequest(BaseModel):
    """Body for ``POST /api/library/notes``."""

    title: str
    content: str
    slug: str | None = None
    summary: str = ""
    tags: tuple[str, ...] = ()
    refs: tuple[str, ...] = ()


class NoteResponse(BaseModel):
    """A created note's index-shaped view (body is fetched via the asset API)."""

    asset_id: str
    title: str
    path: str
    summary: str = ""
    tags: tuple[str, ...] = ()
    refs: tuple[str, ...] = Field(default=())


class UpdateNoteRequest(BaseModel):
    """Body for ``PUT /api/library/notes/{asset_id}`` — edit a note's body."""

    content: str


class ZoteroImportRequest(BaseModel):
    """Body for ``POST /api/library/zotero/import``."""

    path: str
    """Path to ``zotero.sqlite`` or the Zotero data directory containing it."""


class ZoteroImportResponse(BaseModel):
    """Result of a Zotero link/import."""

    imported: int
    references: list[Reference]
