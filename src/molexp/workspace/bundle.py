"""``Bundle`` — the OKF bundle façade over the ``workspace.Folder`` tree.

A *bundle* is a directory subtree whose Concept dirs (dirs that directly hold
``meta.yaml``) form a knowledge graph via ``index.md`` markdown links. A
:class:`Bundle` wraps a *bundle root* and exposes the whole subtree — at any
depth — as one management entry point: :meth:`walk` (depth-first Concept
enumeration), :meth:`get` (path-as-identity resolution), :meth:`put`
(idempotent materialization), :meth:`link` (a semantic edge written as a
markdown link, round-tripping through :meth:`Folder.out_edges`), plus a derived
rollup :meth:`build_index` (→ ``index.json`` machine + ``INDEX.md`` human/agent)
filtered by :meth:`search`.

It is a thin runtime container (explicit ``__init__``, no pydantic): it records
the root path + filesystem and does **no** disk I/O on construction. The
semantic graph lives in markdown (``index.md`` links), so :meth:`link`
round-trips through :meth:`Folder.out_edges`.

``Bundle`` indexes the OKF Concept tree itself (``Note`` / ``ReferenceConcept``
directories whose path is their identity), superseding the legacy per-scope
``Library`` that was removed in wsokf-11.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path as _StdPath
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, NamedTuple, cast

from molexp.ids import slugify

from .bundle_index import (
    INDEX_JSON_FILENAME,
    INDEX_MD_FILENAME,
    BundleIndex,
    ConceptIndexEntry,
    extract_title,
)
from .edges import DEFAULT_EDGE_ROLE, EdgeRole
from .errors import ConceptNotFoundError
from .folder import (
    META_YAML_FILENAME,
    OPS_DIR,
    Folder,
    append_link,
    concept_from_dir,
)
from .fs import FileSystem, PathArg
from .fs_local import LocalFileSystem
from .reference_meta import ReferenceMeta
from .zotero_concepts import read_zotero_items

if TYPE_CHECKING:
    from .assets.base import Asset
    from .concepts import Note, ReferenceConcept
    from .doc_embed import EntitySummary

__all__ = ["Backlink", "Bundle"]

REFERENCES_GROUP = "references"
SOURCES_FILENAME = "sources.json"


class Backlink(NamedTuple):
    """One resolved reverse edge, as returned by :meth:`Bundle.backlinks`.

    A ``NamedTuple`` wrapper — **not** a bare Concept: it pairs the *source*
    Concept whose ``index.md`` holds the link with the edge's declared *role*
    (the inverse of a :meth:`Folder.typed_out_edges` :class:`Edge`). Unpack it
    as ``source, role = backlink`` or read the fields by name.

    Attributes:
        source: The Concept (a :class:`Folder` subclass) whose ``index.md``
            holds the edge pointing at the queried target.
        role: The declared :class:`~molexp.workspace.edges.EdgeRole` of that
            edge (``"references"`` for untyped legacy links).
    """

    source: Folder
    role: EdgeRole


def _utcnow() -> datetime:
    """Return the current time as an aware-UTC ``datetime``."""
    return datetime.now(UTC)


def _is_concept_dir(path: PathArg, fs: FileSystem) -> bool:
    """Return ``True`` iff *path* is a dir that directly holds ``meta.yaml``."""
    return fs.is_dir(path) and fs.exists(fs.join(path, META_YAML_FILENAME))


class Bundle:
    """A management façade over an OKF bundle (a Concept-directory tree)."""

    def __init__(self, root: PathArg, *, fs: FileSystem | None = None) -> None:
        """Record the bundle *root* + filesystem; perform no disk I/O (lazy).

        Args:
            root: The bundle root directory.
            fs: The filesystem backing this bundle's I/O (default: local).
        """
        self._root = _StdPath(os.fspath(root))
        self._fs: FileSystem = fs if fs is not None else LocalFileSystem()

    @property
    def root(self) -> _StdPath:
        """The bundle root directory."""
        return self._root

    @property
    def fs(self) -> FileSystem:
        """The filesystem backing this bundle's I/O."""
        return self._fs

    # ── identity helpers ─────────────────────────────────────────────────

    def rel_path(self, concept: Folder) -> str:
        """Return *concept*'s identity: its POSIX path relative to the root."""
        return _StdPath(str(concept.resolve())).relative_to(self._root).as_posix()

    def _folder_for(self, path: PathArg) -> Folder:
        """Build the typed Concept whose identity is the Concept dir *path*.

        ``concept_from_dir`` reconstructs *path* against a *parent* Folder, and
        typed subclasses chain their ``resolve()`` through that parent (an
        ``Experiment`` resolves via its owning ``Project``). So the bundle hands
        it the nearest **ancestor Concept** — itself reconstructed recursively —
        as the parent, walking up past non-Concept organizational dirs.

        When no ancestor Concept exists within the bundle, a path-only base
        ``Folder`` anchored at *path*'s immediate parent dir carries ``_fs`` +
        path. That is correct for a base Concept — whose ``resolve()`` chains
        only its own name — but a **layout-aware entity** (``Project`` /
        ``Experiment`` / ``Run``) replays its container segment (``projects/`` …)
        on top of that anchor, doubling the path (``.../projects/projects/<id>``).
        This is the nested-mount bug ``server/plan_runtime/record.py`` root-mounts
        to dodge. We detect the replay (``resolve() != child_dir``) and re-anchor
        the thin parent the exact number of levels up that makes the entity's own
        layout replay reproduce *child_dir* — without touching entity
        ``resolve()`` (the frozen Layout naming law).
        """
        child_dir = str(path)
        ancestor = self._nearest_ancestor_concept(child_dir)
        if ancestor is not None:
            return concept_from_dir(child_dir, ancestor)
        thin = self._thin_parent(child_dir)
        concept = concept_from_dir(child_dir, thin)
        if self._norm(concept.resolve()) == self._norm(child_dir):
            return concept
        reanchored = self._reanchored_parent(concept, thin, child_dir)
        return concept_from_dir(child_dir, reanchored)

    @staticmethod
    def _norm(path: PathArg) -> str:
        """Normalize *path* to a canonical POSIX string for identity comparison."""
        return str(PurePosixPath(os.path.normpath(str(path))))

    def _reanchored_parent(self, concept: Folder, thin: Folder, child_dir: str) -> Folder:
        """Re-anchor a layout-replaying entity so ``resolve() == child_dir``.

        *thin* resolved to ``dirname(child_dir)``; the entity then re-added ``k``
        container segments on top of it. Count ``k`` from the overshoot and pin a
        fresh anchor ``k`` levels above *child_dir*, so the entity's own replay
        lands back on *child_dir* exactly.
        """
        resolved = PurePosixPath(self._norm(concept.resolve()))
        anchor = PurePosixPath(self._norm(thin.resolve()))
        target = PurePosixPath(self._norm(child_dir))
        try:
            overshoot = len(resolved.relative_to(anchor).parts)
        except ValueError:  # unexpected resolve() shape — don't guess
            return thin
        if not 1 <= overshoot <= len(target.parents):
            return thin
        return self._pinned_parent(str(target.parents[overshoot - 1]))

    def _nearest_ancestor_concept(self, child_dir: str) -> Folder | None:
        """Reconstruct the closest enclosing Concept dir, or ``None`` if none.

        Walks up from *child_dir* (still inside the bundle root); the first
        ancestor holding ``meta.yaml`` is reconstructed (recursively, so the
        whole typed chain is built) and returned.
        """
        root = str(self._root)
        current = self._fs.dirname(child_dir)
        while current != root and self._is_within_root(current):
            if _is_concept_dir(current, self._fs):
                return self._folder_for(current)
            parent = self._fs.dirname(current)
            if parent == current:  # reached the filesystem root — stop
                break
            current = parent
        return None

    def _is_within_root(self, directory: str) -> bool:
        """Return ``True`` iff *directory* is a strict descendant of the root."""
        try:
            return bool(_StdPath(directory).relative_to(self._root).parts)
        except ValueError:
            return False

    def _thin_parent(self, child_dir: PathArg) -> Folder:
        """A path-only base Folder anchored at *child_dir*'s immediate parent dir.

        The base case for a top-level Concept (one with no enclosing Concept): a
        base ``Folder.resolve()`` chains a single segment (the Concept's own
        name) onto this anchor. See :meth:`_pinned_parent`.
        """
        return self._pinned_parent(self._fs.dirname(str(child_dir)))

    def _pinned_parent(self, directory: PathArg) -> Folder:
        """A path-only base Folder whose ``resolve()`` equals *directory* exactly.

        Built without ``__init__`` to bypass name/kind validation — its sole job
        is to carry ``_fs`` and a fixed ``resolve()`` value into
        ``concept_from_dir`` as a Concept's parent anchor.
        """
        directory = str(directory)
        parent = Folder.__new__(Folder)
        parent._parent = None
        parent._name = self._fs.basename(directory)
        parent._kind = "bundle.parent"
        parent._root_path = _StdPath(self._fs.dirname(directory))
        parent._fs = self._fs
        parent._children_cache = {}
        return parent

    # ── walk / get / put / link ──────────────────────────────────────────

    def walk(self) -> Iterator[Folder]:
        """Yield every Concept under the root, depth-first (preorder).

        A dir is yielded iff it holds ``meta.yaml``. The ``_ops/`` sidecar (and
        everything beneath it) is skipped; non-Concept organizational dirs are
        descended into but not yielded; loose files are inherently skipped.
        """
        yield from self._walk_dir(str(self._root))

    def _walk_dir(self, directory: str) -> Iterator[Folder]:
        if not self._fs.is_dir(directory):
            return
        for name in sorted(self._fs.listdir(directory)):
            if name == OPS_DIR:
                continue
            entry = self._fs.join(directory, name)
            if not self._fs.is_dir(entry):
                continue
            if _is_concept_dir(entry, self._fs):
                yield self._folder_for(entry)
            yield from self._walk_dir(entry)

    def get(self, rel_path: PathArg) -> Folder:
        """Resolve a bundle-relative path to its :class:`Folder`.

        Args:
            rel_path: A bundle-relative POSIX path (the Concept's identity).

        Returns:
            The typed :class:`Folder` Concept at *rel_path*.

        Raises:
            ConceptNotFoundError: if *rel_path* is not a Concept dir.
        """
        rel = PurePosixPath(os.fspath(rel_path))
        target = self._fs.join(str(self._root), *rel.parts)
        if not _is_concept_dir(target, self._fs):
            raise ConceptNotFoundError(str(rel_path))
        return self._folder_for(target)

    def put(self, concept: Folder) -> Folder:
        """Idempotently materialize *concept* (write ``meta.yaml`` if absent).

        Args:
            concept: The Concept to materialize.

        Returns:
            The same *concept* (now backed by a ``meta.yaml`` on disk).
        """
        if not _is_concept_dir(concept.resolve(), self._fs):
            concept.write_meta()
        return concept

    def link(
        self,
        src: Folder,
        dst: Folder,
        *,
        text: str | None = None,
        role: EdgeRole = DEFAULT_EDGE_ROLE,
    ) -> None:
        """Record a typed semantic edge ``src → dst`` as a markdown link in *src*.

        Appends a real markdown link (relative to *src*) to ``src/index.md`` so
        :meth:`Folder.out_edges` resolves it back to *dst* and
        :meth:`Folder.typed_out_edges` recovers *role*. The graph lives in
        markdown, never in ``meta.yaml``. A thin delegator over the single
        :func:`~molexp.workspace.folder.append_link` writer.

        Args:
            src: The Concept the edge originates from.
            dst: The Concept the edge points to.
            text: Optional link label; defaults to *dst*'s name.
            role: The declared :class:`~molexp.workspace.edges.EdgeRole`; defaults
                to :data:`~molexp.workspace.edges.DEFAULT_EDGE_ROLE`.
        """
        append_link(src, dst, text=text, role=role)

    # ── document CRUD verbs (knowledge-docs-01) ──────────────────────────
    #
    # The shared, workspace-owned document surface: CLI and server both call
    # these same ``Bundle`` verbs, so the CRUD logic lives in one place and is
    # never re-implemented at the HTTP boundary (the Python==UI invariant).

    def create_note(self, name: str, *, parent: Folder | None = None, body: str = "") -> Note:
        """Idempotently create (or fetch) a :class:`Note` document Concept.

        *name* is slugified to the directory name (path is identity). With
        *parent* ``None`` the note mounts at the bundle root; otherwise it mounts
        as a child Concept under *parent* (recursive nesting, since a ``Note`` is
        a ``Folder``). Repeat calls on the same slug return the existing Concept
        — no duplicate dir — mirroring :meth:`import_zotero`'s idempotent-create.
        A non-empty *body* is written to ``index.md``.

        Args:
            name: Human name; slugified to the Concept dir name.
            parent: Concept to nest under, or ``None`` for the bundle root.
            body: Initial ``index.md`` narrative (written only when non-empty).

        Returns:
            The mounted :class:`Note` (the existing one on a repeat call).
        """
        from .concepts import Note

        host = parent if parent is not None else self._pinned_parent(self._root)
        slug = slugify(name) or name
        note = cast("Note", host.add_folder(Note(parent=host, name=slug)))
        if body:
            note.set_body(body)
        return note

    def rename_note(self, concept: Folder, new_name: str) -> None:
        """Rename *concept* in place (same parent), preserving body + child docs.

        Delegates to :meth:`Folder.move_to` with the current parent, so the whole
        subtree moves as one and the Concept resolves at its new identity path.

        Args:
            concept: The Concept to rename.
            new_name: The new human name; slugified to the new dir name.

        Raises:
            ValueError: If *concept* is unmounted (has no parent).
        """
        parent = concept.parent
        if parent is None:
            raise ValueError(f"cannot rename unmounted concept {concept.name!r}")
        # Slugify at the verb boundary (as create_note does) so callers may pass
        # a human name; move_to itself requires an already-valid id.
        concept.move_to(parent, new_name=slugify(new_name) or new_name)

    def move_note(self, concept: Folder, new_parent: Folder) -> None:
        """Move *concept* under *new_parent*, preserving body + child docs.

        Delegates to :meth:`Folder.move_to`; the Concept resolves under
        *new_parent* afterward. Existing relative links inside the moved
        ``index.md`` travel verbatim (they are not rewritten) — a derived
        :meth:`backlinks` recompute reflects the Concept's new identity.

        Args:
            concept: The Concept to move.
            new_parent: The Concept (or host Folder) to mount it under.
        """
        concept.move_to(new_parent)

    def delete_note(self, concept: Folder) -> None:
        """Delete *concept* — remove its directory subtree and drop it from the tree.

        Delegates to :meth:`Folder.delete` (recursive removal + parent-cache
        eviction). After deletion :meth:`walk` no longer yields the Concept.

        Args:
            concept: The Concept to delete.
        """
        concept.delete()

    def backlinks(self, concept: Folder) -> list[Backlink]:
        """Return the :class:`Backlink` rows whose edge points at *concept*.

        Each result is a ``Backlink(source, role)`` wrapper — not the bare
        source Concept: ``source`` is the Concept whose ``index.md`` holds the
        edge, ``role`` its declared :class:`~molexp.workspace.edges.EdgeRole`
        (the default ``references`` role is included, never dropped). Computed
        by walking the bundle and reading each Concept's
        :meth:`Folder.typed_out_edges` — a rebuildable derived view; no reverse
        index is persisted (one-source-of-truth law).

        Args:
            concept: The link target to find backlinks for.

        Returns:
            One :class:`Backlink` per edge resolving to *concept*.
        """
        target = self._norm(concept.resolve())
        result: list[Backlink] = []
        for source in self.walk():
            if self._norm(source.resolve()) == target:
                continue  # a self-edge is not a backlink
            for edge in source.typed_out_edges():
                if self._norm(edge.target) == target:
                    result.append(Backlink(source=source, role=edge.role))
        return result

    def export_markdown(self, concept: Folder, *, include_children: bool = True) -> str:
        """Return *concept*'s ``index.md`` as portable Markdown (optionally folding children).

        With *include_children* the descendant :class:`Note` bodies are appended
        depth-first in document order, each under a ``## <bundle-relative-path>``
        header. Read-only — writes nothing.

        Args:
            concept: The Concept whose Markdown to export.
            include_children: Fold descendant ``Note`` bodies in (default ``True``).

        Returns:
            A single portable Markdown string.
        """
        from .concepts import Note

        parts = [concept.read_index()]
        if include_children:
            for descendant in self._walk_dir(str(concept.resolve())):
                if isinstance(descendant, Note):
                    parts.append(f"\n\n## {self.rel_path(descendant)}\n\n{descendant.read_index()}")
        return "".join(parts)

    # ── document embed verb + entity summary (knowledge-docs-05) ─────────

    def embed(
        self,
        src_concept: Note,
        target: Folder | Asset,
        *,
        role: EdgeRole | None = None,
    ) -> None:
        """Embed *target* into document *src_concept* as one typed provenance edge.

        Writes a single typed markdown link from the ``Note`` *src_concept* to
        *target* through the sole edge-writer
        :func:`~molexp.workspace.folder.append_link` -- never a hand-built
        markdown string. *target* is a :class:`Folder` (a ``Run`` /
        ``Experiment`` / ``ReferenceConcept`` / ``Note``, handed to
        ``append_link`` directly) or an
        :class:`~molexp.workspace.assets.base.Asset` (resolved to its in-tree
        record dir ``<scope_dir>/assets/<asset_id>/`` and pointed at, never
        copied).

        With *role* ``None`` the per-kind default is used
        (:func:`~molexp.workspace.doc_embed.default_role_for`): a ``Run`` /
        ``Experiment`` -> ``records``, a ``ReferenceConcept`` -> ``cites``, any
        ``Asset`` (or other) -> ``references``; all from the frozen
        :class:`~molexp.workspace.edges.EdgeRole` vocabulary. An explicit *role*
        overrides the default.

        Args:
            src_concept: The document (a ``Note``) the edge originates from.
            target: The live entity to embed (a ``Folder`` or an ``Asset``).
            role: An explicit edge role; defaults to the per-kind default.

        Raises:
            TypeError: If *target* is neither a ``Folder`` nor an ``Asset``.
            FileNotFoundError: If an ``Asset`` target has no on-disk record dir
                (missing precondition -- raised, never silently downgraded).
        """
        from .doc_embed import default_role_for, resolve_embed_target

        dst = resolve_embed_target(target, root=self._root, pin=self._pinned_parent)
        edge_role = role if role is not None else default_role_for(target)
        append_link(src_concept, dst, role=edge_role)

    def entity_summary(self, target: Folder | Asset) -> EntitySummary:
        """Return a read-only :class:`EntitySummary` for *target* (a UI-card projection).

        Convenience wrapper over
        :func:`~molexp.workspace.doc_embed.summarize_entity` that passes this
        bundle's root as the anchor (needed only to locate an ``Asset``'s record
        dir). Writes nothing.

        Args:
            target: The entity to summarize (a ``Folder`` or an ``Asset``).

        Returns:
            The read-only :class:`EntitySummary`.
        """
        from .doc_embed import summarize_entity

        return summarize_entity(target, root=self._root)

    # ── typed filtered views + Zotero import (wsokf-05) ──────────────────

    def references(self) -> list[ReferenceConcept]:
        """Every OKF ``Reference`` Concept in the bundle (typed view of walk)."""
        from .concepts import ReferenceConcept

        return [c for c in self.walk() if isinstance(c, ReferenceConcept)]

    def notes(self) -> list[Note]:
        """Every OKF ``Note`` Concept in the bundle (typed view of walk)."""
        from .concepts import Note

        return [c for c in self.walk() if isinstance(c, Note)]

    def import_zotero(
        self,
        path: PathArg,
        *,
        under: Folder | None = None,
        now: datetime | None = None,
    ) -> list[ReferenceConcept]:
        """Link a local Zotero library (read-only) as ``Reference`` Concepts.

        Each Zotero item becomes a :class:`ReferenceConcept` under *under*
        (default: a ``references/`` group at the bundle root); its PDF is
        *pointed at* via ``ReferenceMeta.pdf_path`` — no bytes are copied.
        Idempotent on ``source_key``: re-importing an item updates its
        ``meta.yaml`` in place (the slugified Zotero key is the dir name)
        rather than duplicating it. Records the link in ``sources.json``.

        Args:
            path: The ``zotero.sqlite`` to read (opened read-only).
            under: The Concept to mount references beneath (default: a
                ``references/`` group materialized at the bundle root).
            now: Import timestamp; defaults to aware-UTC ``datetime.now``.

        Returns:
            The :class:`ReferenceConcept` records created or updated.
        """
        from .concepts import ReferenceConcept

        host = under if under is not None else self._references_group()
        items = read_zotero_items(path)
        refs: list[ReferenceConcept] = []
        for item in items:
            slug = slugify(item.key) or item.key
            ref = cast(
                "ReferenceConcept",
                host.add_folder(ReferenceConcept(parent=host, name=slug)),
            )
            ref.write_reference_meta(
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

    def _references_group(self) -> Folder:
        """Materialize (idempotently) the default ``references/`` host Folder."""
        group = Folder(
            name=REFERENCES_GROUP,
            kind="bundle.references",
            root_path=str(self._root),
            fs=self._fs,
        )
        if not self._fs.is_dir(group.resolve()):
            group.materialize()
            group.write_meta()
        return group

    def _record_source(self, source: str, path: str, count: int, *, now: datetime | None) -> None:
        """Append (dedup on source+path) a linked-source row into ``sources.json``."""
        sources_path = self._fs.join(str(self._root), SOURCES_FILENAME)
        existing: list[dict[str, object]] = []
        if self._fs.is_file(sources_path):
            raw = json.loads(self._fs.read_text(sources_path))
            if isinstance(raw, list):
                existing = [e for e in raw if isinstance(e, dict)]
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
        self._fs.atomic_write_json(sources_path, existing)

    # ── derived index + search ───────────────────────────────────────────

    def _entry_for(self, concept: Folder) -> ConceptIndexEntry:
        meta = concept.read_meta()
        raw_type = meta.get("type")
        raw_id = meta.get("id")
        raw_tags = meta.get("tags")
        title = extract_title(concept.read_index()) or concept.name
        links = tuple(
            _StdPath(str(target)).relative_to(self._root).as_posix()
            for target in concept.out_edges()
        )
        tags = tuple(str(t) for t in raw_tags) if isinstance(raw_tags, list) else ()
        return ConceptIndexEntry(
            path=self.rel_path(concept),
            type=str(raw_type) if raw_type is not None else "",
            id=str(raw_id) if raw_id is not None else None,
            title=title,
            tags=tags,
            links=links,
        )

    def build_index(self, *, now: datetime | None = None) -> BundleIndex:
        """Rebuild the derived bundle index and write its two sibling files.

        Walks every Concept, rolls its identity into a :class:`BundleIndex`, and
        atomically writes ``index.json`` (machine) + ``INDEX.md`` (human/agent)
        at the bundle root. Always a fresh, full rebuild — never authoritative
        (``meta.yaml`` + ``index.md`` remain the source of truth).

        Args:
            now: Build timestamp; defaults to aware-UTC ``datetime.now``.

        Returns:
            The freshly built :class:`BundleIndex`.
        """
        index = BundleIndex(
            generated_at=now or _utcnow(),
            entries=tuple(self._entry_for(c) for c in self.walk()),
        )
        self._fs.atomic_write_json(
            self._fs.join(str(self._root), INDEX_JSON_FILENAME),
            index.model_dump(mode="json"),
        )
        self._fs.atomic_write_text(
            self._fs.join(str(self._root), INDEX_MD_FILENAME),
            index.to_markdown(),
        )
        return index

    def _load_index(self) -> BundleIndex:
        path = self._fs.join(str(self._root), INDEX_JSON_FILENAME)
        if not self._fs.is_file(path):
            return self.build_index()
        return BundleIndex.model_validate(json.loads(self._fs.read_text(path)))

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

        Args:
            text: Case-insensitive substring matched against ``path``/``title``.
            concept_type: Exact ``type`` to match.
            tag: A tag that must be present on the Concept.
            rebuild: Refresh the index first (default); otherwise reuse the last
                written ``index.json`` (built on demand if absent).

        Returns:
            The matching :class:`ConceptIndexEntry` rows.
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
