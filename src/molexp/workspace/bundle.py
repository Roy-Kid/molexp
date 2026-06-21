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

``Bundle`` is a NEW, DISTINCT class from the per-scope ``workspace.Library``
(notes + literature). ``Library`` indexes documents inside one scope; ``Bundle``
indexes the Concept tree itself. They share only the loose word "library" in the
reference tree; conflating them would overload one name with two unrelated
surfaces and break the four ``Folder.library`` callers.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path as _StdPath
from pathlib import PurePosixPath

from .bundle_index import (
    INDEX_JSON_FILENAME,
    INDEX_MD_FILENAME,
    BundleIndex,
    ConceptIndexEntry,
    extract_title,
)
from .errors import ConceptNotFoundError
from .folder import (
    META_YAML_FILENAME,
    OPS_DIR,
    Folder,
    concept_from_dir,
)
from .fs import FileSystem, PathArg
from .fs_local import LocalFileSystem

__all__ = ["Bundle"]


def _utcnow() -> datetime:
    """Return the current time as an aware-UTC ``datetime``."""
    return datetime.now(UTC)


def _is_concept_dir(path: PathArg, fs: FileSystem) -> bool:
    """Return ``True`` iff *path* is a dir that directly holds ``meta.yaml``."""
    return fs.is_dir(path) and fs.exists(fs.join(path, META_YAML_FILENAME))


def append_link(src: Folder, dst: Folder, *, text: str | None = None) -> None:
    """Append a relative markdown link ``src → dst`` to ``src``'s ``index.md``.

    Writes a real markdown link (relative to *src*'s dir) so
    :meth:`Folder.out_edges` resolves it back to *dst*. The graph lives in
    markdown, never in ``meta.yaml``. Appends unconditionally; link dedup and
    edge-typing remain a future enhancement.

    Args:
        src: The Concept the edge originates from.
        dst: The Concept the edge points to.
        text: Optional link label; defaults to *dst*'s name.
    """
    rel = os.path.relpath(str(dst.resolve()), str(src.resolve()))
    rel_posix = PurePosixPath(rel).as_posix()
    label = text if text is not None else dst.name
    existing = src.read_index()
    prefix = existing if not existing or existing.endswith("\n") else existing + "\n"
    src.write_index(f"{prefix}- [{label}]({rel_posix})\n")


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
        as the parent, walking up past non-Concept organizational dirs. When no
        ancestor Concept exists within the bundle, a path-only base ``Folder``
        (resolving to the immediate parent dir) carries ``_fs`` + path so the
        identity stays the on-disk path.
        """
        child_dir = str(path)
        ancestor = self._nearest_ancestor_concept(child_dir)
        parent = ancestor if ancestor is not None else self._thin_parent(child_dir)
        return concept_from_dir(child_dir, parent)

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
        """A path-only base Folder whose ``resolve()`` equals *child_dir*'s dir.

        Built without ``__init__`` to bypass name/kind validation — its sole job
        is to carry ``_fs`` and a ``resolve()`` value into ``concept_from_dir``
        for a top-level Concept (one with no enclosing Concept).
        """
        directory = self._fs.dirname(str(child_dir))
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

    def link(self, src: Folder, dst: Folder, *, text: str | None = None) -> None:
        """Record a semantic edge ``src → dst`` as a markdown link in *src*.

        Appends a real markdown link (relative to *src*) to ``src/index.md`` so
        :meth:`Folder.out_edges` resolves it back to *dst*. The graph lives in
        markdown, never in ``meta.yaml``.

        Args:
            src: The Concept the edge originates from.
            dst: The Concept the edge points to.
            text: Optional link label; defaults to *dst*'s name.
        """
        append_link(src, dst, text=text)

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
