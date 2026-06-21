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
(side-effect-free :meth:`resolve`, lazy :meth:`path`, five-verb CRUD) but is
``meta.yaml``-authoritative and local-only — the ``FileSystem`` Protocol /
remote backends are deliberately out of scope for okf-01-03, so I/O uses
stdlib :class:`pathlib.Path`. All file writes route through the cross-layer
:mod:`molexp.atomicio` primitives (referenced via the module so spies that
patch ``molexp.atomicio.*`` intercept them).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import ClassVar, NamedTuple

import molexp.atomicio as atomicio
from molexp.ids import slugify

from .errors import ConceptExistsError, ConceptNotFoundError
from .models import ConceptMeta

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


def _is_concept_dir(path: Path) -> bool:
    return (path / META_FILE).is_file()


class Folder:
    """Base class for every OKF Concept directory.

    Construct with either ``root`` (a root Concept anchored at ``root/<id>``)
    or ``parent`` (a child Concept), never both. ``__init__`` performs no
    disk I/O; the directory is materialized lazily by :meth:`path` or any
    ``write_*`` call. The on-disk identity is ``slugify(name)``.
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

    # ── Identity / path ──────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """The on-disk identity (slugified)."""
        return self._name

    @property
    def parent(self) -> Folder | None:
        return self._parent

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
        target.mkdir(parents=True, exist_ok=True)
        return target

    # ── meta.yaml ────────────────────────────────────────────────────────

    def read_meta(self) -> ConceptMeta:
        """Load and validate this Concept's ``meta.yaml``."""
        text = (self.resolve() / META_FILE).read_text(encoding="utf-8")
        return ConceptMeta.from_yaml(text)

    def write_meta(self, meta: ConceptMeta) -> None:
        """Atomically write this Concept's ``meta.yaml``."""
        atomicio.atomic_write_text(self.path() / META_FILE, meta.to_yaml())

    # ── index.md / log.md ────────────────────────────────────────────────

    def read_index(self) -> str:
        """Return ``index.md`` body, or ``""`` if absent."""
        p = self.resolve() / INDEX_FILE
        return p.read_text(encoding="utf-8") if p.is_file() else ""

    def write_index(self, text: str) -> None:
        """Atomically write ``index.md`` (narrative + markdown links)."""
        atomicio.atomic_write_text(self.path() / INDEX_FILE, text)

    def read_log(self) -> str:
        """Return ``log.md`` body, or ``""`` if absent."""
        p = self.resolve() / LOG_FILE
        return p.read_text(encoding="utf-8") if p.is_file() else ""

    def append_log(self, entry: str, *, timestamp: datetime | None = None) -> None:
        """Append a timestamped line to ``log.md`` (atomic rewrite)."""
        ts = (timestamp or datetime.now()).isoformat(timespec="seconds")
        line = f"- {ts} {entry}\n"
        atomicio.atomic_write_text(self.path() / LOG_FILE, self.read_log() + line)

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
            if _is_concept_dir(concept_dir):
                concepts.append(concept_dir)
            else:
                other.append(target)
        return LinkScan(concepts=concepts, external=external, other=other)

    def out_edges(self) -> list[Path]:
        """In-tree Concept link targets — the knowledge-graph out-edges."""
        return self.links().concepts

    # ── Five-verb CRUD (child concept = subdir with meta.yaml) ────────────

    def add_folder(self, name: str, *, concept_type: str = DEFAULT_CONCEPT_TYPE) -> Folder:
        """Create (or return existing) child Concept; idempotent on slug."""
        slug = slugify(name) or name
        cached = self._children.get(slug)
        if cached is not None:
            return cached
        child = Folder(name=name, parent=self, concept_type=concept_type)
        if not _is_concept_dir(child.resolve()):
            child.write_meta(ConceptMeta(type=concept_type))
        self._children[slug] = child
        return child

    def get_folder(self, name: str) -> Folder:
        """Return the child Concept named *name* (raw or slug)."""
        for candidate in (name, slugify(name)):
            if not candidate:
                continue
            cached = self._children.get(candidate)
            if cached is not None:
                return cached
            child = Folder(name=candidate, parent=self)
            if _is_concept_dir(child.resolve()):
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
            if _is_concept_dir(self.resolve() / candidate):
                return True
        return False

    def list_folders(self) -> list[Folder]:
        """All child Concepts (subdirs holding ``meta.yaml``), sorted by id."""
        base = self.resolve()
        if not base.is_dir():
            return []
        out: list[Folder] = []
        for entry in sorted(base.iterdir()):
            if entry.name == OPS_DIR or not entry.is_dir():
                continue
            if not _is_concept_dir(entry):
                continue
            child = self._children.get(entry.name) or Folder(name=entry.name, parent=self)
            self._children[entry.name] = child
            out.append(child)
        return out

    def remove_folder(self, name: str) -> None:
        """Delete a child Concept (and its subtree); evict from cache."""
        for candidate in (name, slugify(name)):
            if not candidate:
                continue
            child_dir = self.resolve() / candidate
            if _is_concept_dir(child_dir):
                shutil.rmtree(child_dir)
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
        d.mkdir(parents=True, exist_ok=True)
        return d

    def read_ops_json(self, name: str) -> dict | None:
        """Read ``_ops/<name>.json``, or ``None`` if absent."""
        p = self.resolve() / OPS_DIR / f"{name}.json"
        if not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def write_ops_json(self, name: str, data: object) -> None:
        """Atomically write ``_ops/<name>.json`` (operational state)."""
        atomicio.atomic_write_json(self.ops_dir() / f"{name}.json", data)

    def update_ops_json(self, name: str, fn: Callable[[dict], dict]) -> dict:
        """Read-modify-write ``_ops/<name>.json`` under an advisory file lock."""
        ops = self.ops_dir()
        with atomicio.file_lock(ops / f"{name}.json.lock"):
            current = self.read_ops_json(name) or {}
            updated = fn(current)
            atomicio.atomic_write_json(ops / f"{name}.json", updated)
        return updated


__all__ = ["Folder", "LinkScan"]
