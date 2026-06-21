"""``Folder`` — the OKF Concept-on-disk base for ``molexp.knowledge``.

A Concept is a directory whose **path is its identity**. Its on-disk form is
physically split into three files (never a frontmatter blob):

    <concept>/
    ├── meta.yaml      # structured metadata (ConceptMeta) — authority
    ├── index.md       # narrative + markdown-link knowledge graph
    ├── log.md         # optional chronological history (append-only)
    └── _ops/          # operational sidecar — hot machine state, NOT knowledge
        └── <name>.json

``Folder`` mirrors the method shapes of ``molexp.workspace.folder.Folder``
(side-effect-free :meth:`resolve`, lazy :meth:`path`, five-verb CRUD) and is
``meta.yaml``-authoritative. All disk I/O routes through an injectable
:class:`FileSystem` (okf-05-04) so a bundle can live on a remote backend; the
default :class:`LocalFileSystem` reproduces the prior stdlib + ``atomicio``
behavior. A child inherits its parent's filesystem.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import ClassVar, NamedTuple, cast

from molexp.ids import slugify

from .errors import ConceptExistsError, ConceptNotFoundError
from .fs import FileSystem, LocalFileSystem
from .models import ConceptMeta
from .types import concept_type, resolve_concept_type

META_FILE = "meta.yaml"
INDEX_FILE = "index.md"
LOG_FILE = "log.md"
OPS_DIR = "_ops"
DEFAULT_CONCEPT_TYPE = "folder"

# Standard markdown inline link: [text](target)
_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


class LinkScan(NamedTuple):
    """Resolved out-links of a Concept's ``index.md``.

    Attributes:
        concepts: Targets that resolve to in-tree Concept dirs (have
            ``meta.yaml``) — the knowledge-graph out-edges.
        external: ``http(s)://`` links.
        other: In-tree targets that are not Concepts (no ``meta.yaml``).
    """

    concepts: list[Path]
    external: list[str]
    other: list[str]


def _is_concept_dir(path: Path, fs: FileSystem) -> bool:
    return fs.is_file(path / META_FILE)


@concept_type(DEFAULT_CONCEPT_TYPE)
class Folder:
    """Base class for every OKF Concept directory.

    Construct with either ``root`` (a root Concept anchored at ``root/<id>``)
    or ``parent`` (a child Concept), never both. ``__init__`` performs no
    disk I/O; the directory is materialized lazily by :meth:`path` or any
    ``write_*`` call. The on-disk identity is ``slugify(name)``. A root folder
    defaults to a :class:`LocalFileSystem`; a child inherits its parent's.
    """

    _not_found_error_cls: ClassVar[type[Exception]] = ConceptNotFoundError
    _exists_error_cls: ClassVar[type[Exception]] = ConceptExistsError

    def __init__(
        self,
        *,
        name: str,
        parent: Folder | None = None,
        root: str | os.PathLike[str] | None = None,
        concept_type: str = DEFAULT_CONCEPT_TYPE,
        fs: FileSystem | None = None,
    ) -> None:
        if parent is None and root is None:
            raise ValueError("Folder requires either parent or root")
        if parent is not None and root is not None:
            raise ValueError("Folder: parent and root are mutually exclusive")
        self._name = slugify(name) or name
        self._parent = parent
        self._root = Path(root) if root is not None else None
        self._type = concept_type
        self._children: dict[str, Folder] = {}
        if fs is not None:
            self._fs: FileSystem = fs
        elif parent is not None:
            self._fs = parent._fs
        else:
            self._fs = LocalFileSystem()

    # ── Identity / path ──────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """The on-disk identity (slugified)."""
        return self._name

    @property
    def parent(self) -> Folder | None:
        return self._parent

    @property
    def fs(self) -> FileSystem:
        """The filesystem backing this Concept's I/O."""
        return self._fs

    @property
    def concept_type(self) -> str:
        """The Concept's declared ``type`` (the required ``meta.yaml`` field)."""
        return self._type

    def resolve(self) -> Path:
        """Compute the on-disk path without any I/O (no mkdir)."""
        if self._parent is None:
            assert self._root is not None  # guaranteed by __init__
            return self._root / self._name
        return self._parent.resolve() / self._name

    def path(self) -> Path:
        """Return the on-disk path, creating the directory if absent (lazy)."""
        target = self.resolve()
        self._fs.mkdir(target)
        return target

    # ── meta.yaml ────────────────────────────────────────────────────────

    def read_meta(self) -> ConceptMeta:
        """Load and validate this Concept's ``meta.yaml``."""
        return ConceptMeta.from_yaml(self._fs.read_text(self.resolve() / META_FILE))

    def write_meta(self, meta: ConceptMeta) -> None:
        """Atomically write this Concept's ``meta.yaml``."""
        self._fs.write_text(self.path() / META_FILE, meta.to_yaml())

    # ── index.md / log.md ────────────────────────────────────────────────

    def read_index(self) -> str:
        """Return ``index.md`` body, or ``""`` if absent."""
        p = self.resolve() / INDEX_FILE
        return self._fs.read_text(p) if self._fs.is_file(p) else ""

    def write_index(self, text: str) -> None:
        """Atomically write ``index.md`` (narrative + markdown links)."""
        self._fs.write_text(self.path() / INDEX_FILE, text)

    def read_log(self) -> str:
        """Return ``log.md`` body, or ``""`` if absent."""
        p = self.resolve() / LOG_FILE
        return self._fs.read_text(p) if self._fs.is_file(p) else ""

    def append_log(self, entry: str, *, timestamp: datetime | None = None) -> None:
        """Append a timestamped line to ``log.md`` (atomic rewrite)."""
        ts = (timestamp or datetime.now()).isoformat(timespec="seconds")
        line = f"- {ts} {entry}\n"
        self._fs.write_text(self.path() / LOG_FILE, self.read_log() + line)

    # ── Markdown-link knowledge graph ────────────────────────────────────

    def links(self) -> LinkScan:
        """Parse ``index.md`` markdown links, classified.

        Targets are resolved relative to this Concept's dir; ``index.md``
        suffixes are stripped to the containing dir. See :class:`LinkScan`.
        """
        base = self.resolve()
        concepts: list[Path] = []
        external: list[str] = []
        other: list[str] = []
        for target in _MD_LINK.findall(self.read_index()):
            if target.startswith(("http://", "https://")):
                external.append(target)
                continue
            norm = Path(os.path.normpath(base / target))
            concept_dir = norm.parent if norm.name == INDEX_FILE else norm
            if _is_concept_dir(concept_dir, self._fs):
                concepts.append(concept_dir)
            else:
                other.append(target)
        return LinkScan(concepts=concepts, external=external, other=other)

    def out_edges(self) -> list[Path]:
        """In-tree Concept link targets — the knowledge-graph out-edges."""
        return self.links().concepts

    # ── Five-verb CRUD (child concept = subdir with meta.yaml) ────────────

    def _ensure_materialized(self) -> None:
        """Write this Concept's ``meta.yaml`` (with its own type) if absent.

        Adding a child promotes the parent to a real Concept, so a
        programmatically-built tree is fully walkable. (A dir created by hand —
        plain ``mkdir`` — stays an organizational non-Concept until written.)
        """
        if not _is_concept_dir(self.resolve(), self._fs):
            self.write_meta(ConceptMeta(type=self._type))

    def add_folder(self, name: str, *, concept_type: str = DEFAULT_CONCEPT_TYPE) -> Folder:
        """Create (or return existing) child Concept; idempotent on slug.

        The returned instance is the registry-resolved subclass for
        *concept_type* (unknown types fall back to base :class:`Folder`).
        """
        self._ensure_materialized()
        slug = slugify(name) or name
        cached = self._children.get(slug)
        if cached is not None:
            return cached
        cls = resolve_concept_type(concept_type, Folder)
        child = cls(name=name, parent=self, concept_type=concept_type)
        if not _is_concept_dir(child.resolve(), self._fs):
            child.write_meta(ConceptMeta(type=concept_type))
        self._children[slug] = child
        return child

    def get_folder(self, name: str) -> Folder:
        """Return the child Concept named *name* (raw or slug), correctly typed."""
        for candidate in (name, slugify(name)):
            if not candidate:
                continue
            cached = self._children.get(candidate)
            if cached is not None:
                return cached
            child_dir = self.resolve() / candidate
            if _is_concept_dir(child_dir, self._fs):
                child = concept_from_dir(child_dir, parent=self)
                self._children[child.name] = child
                return child
        raise self._not_found_error_cls(name)

    def has_folder(self, name: str) -> bool:
        """Whether a child Concept named *name* (raw or slug) exists."""
        for candidate in (name, slugify(name)):
            if not candidate:
                continue
            if candidate in self._children:
                return True
            if _is_concept_dir(self.resolve() / candidate, self._fs):
                return True
        return False

    def list_folders(self) -> list[Folder]:
        """All child Concepts (subdirs holding ``meta.yaml``), sorted by id."""
        base = self.resolve()
        if not self._fs.is_dir(base):
            return []
        out: list[Folder] = []
        for entry in sorted(self._fs.iterdir(base)):
            if entry.name == OPS_DIR or not self._fs.is_dir(entry):
                continue
            if not _is_concept_dir(entry, self._fs):
                continue
            child = self._children.get(entry.name) or concept_from_dir(entry, parent=self)
            self._children[entry.name] = child
            out.append(child)
        return out

    def remove_folder(self, name: str) -> None:
        """Delete a child Concept (and its subtree); evict from cache."""
        for candidate in (name, slugify(name)):
            if not candidate:
                continue
            child_dir = self.resolve() / candidate
            if _is_concept_dir(child_dir, self._fs):
                self._fs.rmtree(child_dir)
                self._children.pop(candidate, None)
                return
            if candidate in self._children:
                self._children.pop(candidate, None)
                return
        raise self._not_found_error_cls(name)

    # ── _ops operational sidecar (hot machine state — never in meta.yaml) ─

    def ops_dir(self) -> Path:
        """Return the per-Concept ``_ops/`` dir, creating it if absent."""
        d = self.resolve() / OPS_DIR
        self._fs.mkdir(d)
        return d

    def read_ops_json(self, name: str) -> dict | None:
        """Read ``_ops/<name>.json``, or ``None`` if absent."""
        p = self.resolve() / OPS_DIR / f"{name}.json"
        return cast("dict", self._fs.read_json(p)) if self._fs.is_file(p) else None

    def write_ops_json(self, name: str, data: object) -> None:
        """Atomically write ``_ops/<name>.json`` (operational state)."""
        self._fs.write_json(self.ops_dir() / f"{name}.json", data)

    def update_ops_json(self, name: str, fn: Callable[[dict], dict]) -> dict:
        """Read-modify-write ``_ops/<name>.json`` under an advisory file lock."""
        ops = self.ops_dir()
        with self._fs.lock(ops / f"{name}.json.lock"):
            current = self.read_ops_json(name) or {}
            updated = fn(current)
            self._fs.write_json(ops / f"{name}.json", updated)
        return updated


def append_link(src: Folder, dst: Folder, *, text: str | None = None) -> None:
    """Append a markdown link ``src → dst`` to ``src``'s ``index.md``.

    The link is written relative to *src* so :meth:`Folder.out_edges` resolves
    it back to *dst*. The semantic graph lives in markdown, never in
    ``meta.yaml``. Shared by ``Library.link`` and ``Note.cite``.
    """
    rel = os.path.relpath(Path(dst.resolve()), Path(src.resolve()))
    label = text or dst.name
    src.write_index(src.read_index() + f"- [{label}]({rel})\n")


def concept_from_dir(
    directory: Path,
    *,
    parent: Folder | None = None,
    root: str | os.PathLike[str] | None = None,
    fs: FileSystem | None = None,
) -> Folder:
    """Reconstruct *directory* as its registry-resolved Concept subclass.

    Reads ``directory/meta.yaml``'s ``type`` and instantiates the matching
    class (unknown / unreadable type → base :class:`Folder`, forward-compat),
    carrying the on-disk type + filesystem through. Pass exactly one of
    *parent* / *root*; a child inherits *parent*'s fs, a root takes *fs*.
    """
    effective_fs = fs or (parent._fs if parent is not None else LocalFileSystem())
    meta_type = DEFAULT_CONCEPT_TYPE
    meta_path = directory / META_FILE
    if effective_fs.is_file(meta_path):
        try:
            meta_type = ConceptMeta.from_yaml(effective_fs.read_text(meta_path)).type
        except Exception:  # malformed meta degrades to base Folder (forward-compat)
            meta_type = DEFAULT_CONCEPT_TYPE
    cls = resolve_concept_type(meta_type, Folder)
    return cls(
        name=directory.name, parent=parent, root=root, concept_type=meta_type, fs=effective_fs
    )


__all__ = ["Folder", "LinkScan", "append_link", "concept_from_dir"]
