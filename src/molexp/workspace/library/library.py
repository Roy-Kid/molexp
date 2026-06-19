"""Library — notes + references storage surface for one Folder scope.

A peer of :class:`~molexp.workspace.assets.data.DataAssetLibrary`, bound to a
scope's on-disk directory.  It owns the ``library/`` subtree::

    <scope_dir>/library/
    ├── notes/<slug>.md      # each registered as a NoteAsset (Asset == file)
    ├── references.json      # ReferenceStore (bib records, not files)
    ├── index.json           # derived: machine-readable index
    └── INDEX.md             # derived: human/agent-readable index

Notes are registered into the scope's **authoritative** ``assets.json``
manifest (and the derived catalog), so they survive a catalog rebuild like
any other asset.  References live only in ``references.json``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

from ..assets.base import AssetScope
from ..assets.manifest import AssetManifest
from ..assets.note import NoteAsset
from ..utils import compute_content_hash, generate_asset_id, slugify
from .index import (
    INDEX_JSON_FILENAME,
    INDEX_MD_FILENAME,
    LibraryIndex,
    NoteEntry,
)
from .reference import REFERENCES_FILENAME, Reference, ReferenceStore
from .zotero import read_zotero_references

if TYPE_CHECKING:
    from ..catalog.index import AssetCatalog

LIBRARY_DIRNAME = "library"
NOTES_DIRNAME = "notes"
SOURCES_FILENAME = "sources.json"
DISCOVERED_TAG = "discovered"


def _parse_markdown_title(path: Path) -> str | None:
    """Return the first ``# H1`` heading of a markdown file, if any."""
    try:
        with path.open(encoding="utf-8") as fh:
            for _ in range(50):
                line = fh.readline()
                if not line:
                    break
                stripped = line.strip()
                if stripped.startswith("# "):
                    return stripped[2:].strip()
    except (OSError, UnicodeDecodeError):
        return None
    return None


class Library:
    """Notes + references surface for one scope.

    ``catalog`` is optional — passing ``None`` registers notes only into the
    per-scope manifest (useful for tests); production callers pass the
    workspace catalog so live queries see new notes immediately.
    """

    def __init__(
        self,
        scope_dir: str | PathLike[str],
        scope: AssetScope,
        catalog: AssetCatalog | None = None,
    ) -> None:
        self.scope_dir = Path(scope_dir)
        self.scope = scope
        self.catalog = catalog
        self.library_dir = self.scope_dir / LIBRARY_DIRNAME
        self.notes_dir = self.library_dir / NOTES_DIRNAME
        self._references = ReferenceStore(self.library_dir / REFERENCES_FILENAME)
        self._manifest = AssetManifest(self.scope_dir)

    # ── Notes ───────────────────────────────────────────────────────────────

    @property
    def references(self) -> ReferenceStore:
        """The scope's bibliographic store (``references.json``)."""
        return self._references

    def add_note(
        self,
        title: str,
        content: str,
        *,
        slug: str | None = None,
        summary: str = "",
        tags: Sequence[str] = (),
        refs: Sequence[str] = (),
    ) -> NoteAsset:
        """Write a markdown note and register it as a :class:`NoteAsset`.

        Idempotent on ``slug`` (derived from ``title`` when omitted): writing
        a note with an existing slug overwrites the file and re-registers a
        fresh asset record under the same path.

        Args:
            title: Human title; also the asset ``name``.
            content: Markdown body written verbatim to the ``.md`` file.
            slug: Optional explicit filename stem; defaults to a slug of ``title``.
            summary: One-line gloss surfaced in the index.
            tags: Topical labels (stored in the asset's ``tags`` map).
            refs: Citation keys (into ``references.json``) the note cites.

        Returns:
            The registered :class:`NoteAsset`.
        """
        stem = slug or slugify(title) or "note"
        rel_path = Path(LIBRARY_DIRNAME) / NOTES_DIRNAME / f"{stem}.md"
        abs_path = self.scope_dir / rel_path

        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")

        now = datetime.now()
        existing = self._find_note_by_path(rel_path)
        asset = NoteAsset(
            asset_id=existing.asset_id if existing else generate_asset_id(),
            name=title,
            scope=self.scope,
            path=rel_path,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            content_hash=compute_content_hash(abs_path),
            tags=dict.fromkeys(tags, ""),
            title=title,
            summary=summary,
            refs=tuple(refs),
        )

        self._manifest.register(asset)
        if self.catalog is not None:
            self.catalog.register(asset)
        return asset

    def discover_notes(self) -> list[NoteAsset]:
        """Register loose ``*.md`` files in the scope dir as in-place notes.

        Catches documentation the user dropped in directly — e.g. a
        ``README.md`` at the workspace or project root — without copying it:
        the :class:`NoteAsset` ``path`` points at the file where it already
        lives, tagged ``discovered``. Idempotent (keyed by path); a previously
        discovered note whose file has since been deleted is deregistered.

        Scans only the scope directory's top level (not child scopes, not the
        managed ``library/`` subtree), so each scope owns only its own files.

        Returns:
            The discovered notes registered this pass.
        """
        now = datetime.now()
        found: dict[str, Path] = {}
        if self.scope_dir.exists():
            for p in sorted(self.scope_dir.glob("*.md")):
                if p.is_file():
                    found[p.name] = p

        result: list[NoteAsset] = []
        for name, abs_path in found.items():
            rel_path = Path(name)
            title = _parse_markdown_title(abs_path) or abs_path.stem
            prior = self._find_note_by_path(rel_path)
            asset = NoteAsset(
                asset_id=prior.asset_id if prior else generate_asset_id(),
                name=title,
                scope=self.scope,
                path=rel_path,
                created_at=prior.created_at if prior else now,
                updated_at=now,
                content_hash=compute_content_hash(abs_path),
                tags={DISCOVERED_TAG: "1"},
                title=title,
                summary="",
                refs=(),
            )
            self._manifest.register(asset)
            if self.catalog is not None:
                self.catalog.register(asset)
            result.append(asset)

        # Prune discovered notes whose backing file is gone.
        for note in self.list_notes():
            if note.tags.get(DISCOVERED_TAG) != "1":
                continue
            if not (self.scope_dir / note.path).is_file():
                self._manifest.deregister(note.asset_id)
                if self.catalog is not None:
                    self.catalog.deregister_asset(note.asset_id)
        return result

    def list_notes(self) -> list[NoteAsset]:
        """All notes in this scope (from the authoritative manifest)."""
        return [a for a in self._manifest.list() if isinstance(a, NoteAsset)]

    def get_note(self, slug: str) -> NoteAsset | None:
        rel = (Path(LIBRARY_DIRNAME) / NOTES_DIRNAME / f"{slug}.md").as_posix()
        for note in self.list_notes():
            if Path(note.path).as_posix() == rel:
                return note
        return None

    def read_note(self, note: NoteAsset) -> str:
        """Return a note's markdown body."""
        return (self.scope_dir / note.path).read_text(encoding="utf-8")

    def update_note(self, asset_id: str, content: str) -> NoteAsset:
        """Rewrite an existing note's markdown body, preserving its metadata.

        Writes to wherever the note's ``path`` points — so editing a
        discovered ``README.md`` updates the file in place. Title / summary /
        tags / refs are kept; only the body, ``content_hash`` and
        ``updated_at`` change.

        Raises:
            KeyError: No note with ``asset_id`` in this scope.
        """
        prior = next((n for n in self.list_notes() if n.asset_id == asset_id), None)
        if prior is None:
            raise KeyError(f"no note {asset_id!r} in scope {self.scope.urn}")

        abs_path = self.scope_dir / prior.path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")

        asset = prior.model_copy(
            update={
                "updated_at": datetime.now(),
                "content_hash": compute_content_hash(abs_path),
            }
        )
        self._manifest.register(asset)
        if self.catalog is not None:
            self.catalog.register(asset)
        return asset

    # ── References ──────────────────────────────────────────────────────────

    def add_reference(self, ref: Reference) -> Reference:
        """Insert or replace a bibliographic record (idempotent on ``key``)."""
        return self._references.add(ref)

    def list_references(self) -> list[Reference]:
        return self._references.list()

    def import_zotero(self, path: str | PathLike[str]) -> list[Reference]:
        """Link a local Zotero library: import its items as references.

        Reads ``zotero.sqlite`` read-only and adds each item as a
        :class:`Reference` (``source="zotero"``); attached PDFs are *pointed
        at* via ``pdf_path``, never copied. Records the association in
        ``library/sources.json`` so it can be re-synced. Idempotent — a
        re-import overwrites the same keys.

        Args:
            path: The ``zotero.sqlite`` file or its containing data directory.

        Returns:
            The references imported this pass.
        """
        refs = read_zotero_references(path)
        for ref in refs:
            self._references.add(ref)
        self._record_source("zotero", str(Path(path).expanduser()), len(refs))
        return refs

    def list_sources(self) -> list[dict]:
        """External libraries linked into this scope (``sources.json``)."""
        sources_path = self.library_dir / SOURCES_FILENAME
        if not sources_path.exists():
            return []
        import json

        with sources_path.open() as fh:
            return json.load(fh).get("sources", [])

    def _record_source(self, kind: str, path: str, count: int) -> None:
        from ..base import _atomic_write_json

        sources = [s for s in self.list_sources() if not (s["kind"] == kind and s["path"] == path)]
        sources.append(
            {
                "kind": kind,
                "path": path,
                "count": count,
                "last_synced": datetime.now().isoformat(),
            }
        )
        self.library_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            self.library_dir / SOURCES_FILENAME,
            {"schema_version": 1, "sources": sources},
        )

    # ── Index ───────────────────────────────────────────────────────────────

    def index(self) -> LibraryIndex:
        """Derive the :class:`LibraryIndex` from the live sources (no I/O)."""
        notes = sorted(self.list_notes(), key=lambda n: n.title.lower())
        note_entries = tuple(
            NoteEntry(
                asset_id=n.asset_id,
                title=n.title,
                path=Path(n.path).as_posix(),
                summary=n.summary,
                tags=tuple(n.tags.keys()),
                refs=n.refs,
                updated_at=n.updated_at.isoformat(),
            )
            for n in notes
        )
        return LibraryIndex(
            scope=self.scope.urn,
            generated_at=datetime.now().isoformat(),
            notes=note_entries,
            references=tuple(self.list_references()),
        )

    def build_index(self) -> LibraryIndex:
        """Sync discovery, then derive + persist ``index.json`` + ``INDEX.md``.

        Runs :meth:`discover_notes` first so loose ``*.md`` files (e.g. a
        ``README.md``) are reflected in the index the agent reads.
        """
        self.discover_notes()
        index = self.index()

        from ..base import _atomic_write_json

        self.library_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.library_dir / INDEX_JSON_FILENAME, index.model_dump(mode="json"))
        (self.library_dir / INDEX_MD_FILENAME).write_text(index.to_markdown(), encoding="utf-8")
        return index

    # ── Internal ──────────────────────────────────────────────────────────

    def _find_note_by_path(self, rel_path: Path) -> NoteAsset | None:
        target = rel_path.as_posix()
        for note in self.list_notes():
            if Path(note.path).as_posix() == target:
                return note
        return None
