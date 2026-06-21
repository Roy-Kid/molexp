"""``Library`` — the OKF bundle façade for ``molexp.knowledge``.

A :class:`Library` wraps a *bundle root* (a directory) and exposes the whole
Concept tree beneath it — at any depth — as one management entry point. A
directory is a **Concept** iff it directly holds ``meta.yaml`` (okf-01-03);
the ``_ops/`` operational sidecar and its descendants are never Concepts, and
plain organizational dirs (no ``meta.yaml``) are walked *through* but never
*yielded*.

It is a thin runtime container (explicit ``__init__``, no pydantic), mirroring
the lazy-façade idiom of ``molexp.workspace.library.Library``: it records the
root path and does **no** disk I/O on construction. Every Concept it produces
is an okf-01-03 :class:`Folder`; the semantic graph lives in markdown
(``index.md`` links), so :meth:`link` round-trips through
:meth:`Folder.out_edges`.

Beyond the okf-01-04 core (``walk`` / ``get`` / ``put`` / ``link``), it derives
a bundle rollup via ``build_index()`` (→ ``index.json`` + ``INDEX.md``) and
filters it via ``search()`` (okf-03). references/notes (okf-07) surface through
the generic Concept machinery.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import cast

import molexp.atomicio as atomicio
from molexp.ids import slugify

from .concepts import Note, Reference
from .errors import ConceptNotFoundError
from .folder import OPS_DIR, Folder, _is_concept_dir, append_link, concept_from_dir
from .index import (
    INDEX_JSON_FILENAME,
    INDEX_MD_FILENAME,
    ConceptIndexEntry,
    LibraryIndex,
    extract_title,
)
from .models import ConceptMeta
from .ops import _utcnow
from .references import ReferenceMeta
from .zotero import read_zotero_items

__all__ = ["Library"]

REFERENCES_GROUP = "references"
SOURCES_FILENAME = "sources.json"


class Library:
    """A management façade over an OKF bundle (a Concept-directory tree)."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        """Record the bundle *root*; perform no disk I/O (lazy)."""
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """The bundle root directory."""
        return self._root

    # ── identity helpers ─────────────────────────────────────────────────

    def rel_path(self, concept: Folder) -> str:
        """Return *concept*'s identity: its POSIX path relative to the root."""
        return Path(concept.resolve()).relative_to(self._root).as_posix()

    def _folder_for(self, path: Path) -> Folder:
        """Build the typed Concept whose identity is the Concept dir *path*."""
        return concept_from_dir(path, root=path.parent)

    # ── walk / get / put / link ──────────────────────────────────────────

    def walk(self) -> Iterator[Folder]:
        """Yield every Concept under the root, depth-first (preorder).

        A dir is yielded iff it holds ``meta.yaml``. The ``_ops/`` sidecar
        (and everything beneath it) is skipped; non-Concept organizational
        dirs are descended into but not yielded.
        """
        yield from self._walk_dir(self._root)

    def _walk_dir(self, directory: Path) -> Iterator[Folder]:
        if not directory.is_dir():
            return
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir() or entry.name == OPS_DIR:
                continue
            if _is_concept_dir(entry):
                yield self._folder_for(entry)
            yield from self._walk_dir(entry)

    def get(self, rel_path: str | os.PathLike[str]) -> Folder:
        """Resolve a bundle-relative path to its :class:`Folder`.

        Raises:
            ConceptNotFoundError: if *rel_path* is not a Concept dir.
        """
        rel = PurePosixPath(os.fspath(rel_path))
        target = self._root.joinpath(*rel.parts)
        if not _is_concept_dir(target):
            raise ConceptNotFoundError(str(rel_path))
        return self._folder_for(target)

    def put(self, concept: Folder) -> Folder:
        """Idempotently materialize *concept* (write ``meta.yaml`` if absent)."""
        if not _is_concept_dir(concept.resolve()):
            concept.write_meta(ConceptMeta(type=concept.concept_type))
        return concept

    def link(self, src: Folder, dst: Folder, *, text: str | None = None) -> None:
        """Record a semantic edge ``src → dst`` as a markdown link in ``src``.

        Appends a real markdown link (relative to ``src``) to ``src/index.md``
        so :meth:`Folder.out_edges` resolves it back to *dst*. The graph lives
        in markdown, never in ``meta.yaml``. Appends unconditionally; link
        dedup/edge-typing remain a future enhancement.
        """
        append_link(src, dst, text=text)

    def references(self) -> list[Reference]:
        """Every Reference Concept in the bundle (typed view of :meth:`walk`)."""
        return [c for c in self.walk() if isinstance(c, Reference)]

    def notes(self) -> list[Note]:
        """Every Note Concept in the bundle (typed view of :meth:`walk`)."""
        return [c for c in self.walk() if isinstance(c, Note)]

    # ── Zotero import (okf-07-02) ────────────────────────────────────────

    def import_zotero(
        self,
        path: str | os.PathLike[str],
        *,
        under: Folder | None = None,
        now: datetime | None = None,
    ) -> list[Reference]:
        """Link a local Zotero library (read-only) as ``Reference`` Concepts.

        Each Zotero item becomes a ``Reference`` under *under* (default: a
        ``references/`` group at the bundle root); its PDF is *pointed at* via
        ``ReferenceMeta.pdf_path`` (no bytes copied). Idempotent on
        ``source_key`` — re-importing updates a reference's ``meta.yaml`` in
        place rather than duplicating it. Records the link in ``sources.json``.
        """
        host = under if under is not None else Folder(name=REFERENCES_GROUP, root=self._root)
        refs: list[Reference] = []
        items = read_zotero_items(path)
        for item in items:
            ref = cast(
                Reference,
                host.add_folder(slugify(item.key) or item.key, concept_type="reference"),
            )
            ref.write_ref_meta(
                ReferenceMeta(
                    title=item.title,
                    authors=item.authors,
                    year=item.year,
                    doi=item.doi,
                    url=item.url,
                    pdf_path=item.pdf_path,
                    source="zotero",
                    source_key=item.key,
                )
            )
            refs.append(ref)
        self._record_source("zotero", str(path), len(items), now=now)
        return refs

    def _record_source(self, source: str, path: str, count: int, *, now: datetime | None) -> None:
        sources_path = self._root / SOURCES_FILENAME
        existing: list[dict] = []
        if sources_path.is_file():
            existing = json.loads(sources_path.read_text(encoding="utf-8"))
        existing = [
            e for e in existing if not (e.get("source") == source and e.get("path") == path)
        ]
        existing.append(
            {
                "source": source,
                "path": path,
                "count": count,
                "imported_at": (now or _utcnow()).isoformat(),
            }
        )
        atomicio.atomic_write_json(sources_path, existing)

    # ── derived index + search (okf-03) ──────────────────────────────────

    def _entry_for(self, concept: Folder) -> ConceptIndexEntry:
        meta = concept.read_meta()
        title = extract_title(concept.read_index()) or concept.name
        links = tuple(
            Path(target).relative_to(self._root).as_posix() for target in concept.out_edges()
        )
        return ConceptIndexEntry(
            path=self.rel_path(concept),
            type=meta.type,
            id=meta.id,
            title=title,
            tags=tuple(meta.tags),
            links=links,
        )

    def build_index(self, *, now: datetime | None = None) -> LibraryIndex:
        """Rebuild the derived bundle index and write its two sibling files.

        Walks every Concept, rolls its identity into a :class:`LibraryIndex`,
        and atomically writes ``index.json`` (machine) + ``INDEX.md`` (human)
        at the bundle root. Always a fresh, full rebuild — never authoritative.
        """
        index = LibraryIndex(
            generated_at=now or _utcnow(),
            entries=tuple(self._entry_for(c) for c in self.walk()),
        )
        atomicio.atomic_write_json(self._root / INDEX_JSON_FILENAME, index.model_dump(mode="json"))
        atomicio.atomic_write_text(self._root / INDEX_MD_FILENAME, index.to_markdown())
        return index

    def _load_index(self) -> LibraryIndex:
        path = self._root / INDEX_JSON_FILENAME
        if not path.is_file():
            return self.build_index()
        return LibraryIndex.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def search(
        self,
        text: str | None = None,
        *,
        concept_type: str | None = None,
        tag: str | None = None,
        rebuild: bool = True,
    ) -> list[ConceptIndexEntry]:
        """Filter the index by *concept_type* / *tag* / *text* (AND semantics).

        *text* matches a case-insensitive substring of ``path`` or ``title``.
        ``rebuild`` (default) refreshes the index first; otherwise the last
        written ``index.json`` is used (built on demand if absent).
        """
        index = self.build_index() if rebuild else self._load_index()
        needle = text.lower() if text else None
        matches: list[ConceptIndexEntry] = []
        for entry in index.entries:
            if concept_type is not None and entry.type != concept_type:
                continue
            if tag is not None and tag not in entry.tags:
                continue
            if (
                needle is not None
                and needle not in entry.path.lower()
                and needle not in entry.title.lower()
            ):
                continue
            matches.append(entry)
        return matches
