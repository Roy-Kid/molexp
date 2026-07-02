"""OKF ``Note`` + ``Reference`` Concepts on the ``workspace.Folder`` family (wsokf-05).

In OKF a Concept is a **directory** whose path is its identity:

- A :class:`Note` is a Folder whose body is its ``index.md`` and whose citations
  are markdown links (resolved by :meth:`Folder.out_edges`).
- A :class:`ReferenceConcept` is a Folder whose structured bib record lives in
  ``meta.yaml`` (:class:`ReferenceMeta`) and whose human citation text lives in
  ``index.md``. PDFs are *pointed at* via ``ReferenceMeta.pdf_path`` /
  ``pdf_asset_id`` — never copied.

Both register via ``@concept_type(...)`` (the open registry shared with the
``molexp.knowledge`` peer layer), so :func:`molexp.workspace.folder.concept_from_dir`
rebuilds the right subclass from a directory's ``meta.yaml`` ``type``.

The constructors match the workspace :class:`Folder` keyword contract
(``parent`` / ``name`` / ``kind`` / ``root_path`` / ``fs``) and default their
``kind`` to a dotted concept type (``note.note`` / ``reference.reference``),
consistent with the existing ``workspace.root`` / ``workspace.project`` family.

These OKF concepts are a port of ``molexp.knowledge.concepts`` (Note +
Reference). They are the going-forward home for notes + literature, reached
via the :class:`~molexp.workspace.bundle.Bundle` façade.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import ClassVar, cast

from molexp.knowledge.types import concept_type

from .edges import DEFAULT_EDGE_ROLE, EdgeRole
from .folder import META_YAML_FILENAME, Folder, append_link
from .fs import FileSystem, PathArg
from .note_meta import NOTE_TYPE, NoteMeta
from .reference_meta import ReferenceMeta

NOTE_KIND = NOTE_TYPE
REFERENCE_KIND = "reference.reference"


@concept_type(NOTE_KIND)
class Note(Folder):
    """A note Concept — narrative in ``index.md``, citations as markdown links.

    Mountable anywhere (convention, not enforcement). Its body is the
    ``index.md`` text; :meth:`cite` records a reference as a markdown link so
    :meth:`Folder.out_edges` resolves it back.
    """

    DEFAULT_KIND: ClassVar[str] = NOTE_KIND

    def __init__(
        self,
        *,
        parent: Folder | None = None,
        name: str,
        kind: str = NOTE_KIND,
        root_path: PathArg | None = None,
        fs: FileSystem | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name, kind=kind, root_path=root_path, fs=fs)

    def body(self) -> str:
        """Return the note body (its ``index.md``)."""
        return self.read_index()

    def set_body(self, text: str) -> None:
        """Set the note body (its ``index.md``)."""
        self.write_index(text)

    def cite(
        self,
        ref: Folder,
        *,
        text: str | None = None,
        role: EdgeRole = DEFAULT_EDGE_ROLE,
    ) -> None:
        """Cite *ref* — append a typed markdown link resolvable via :meth:`out_edges`.

        A thin delegator over the single :func:`~molexp.workspace.folder.append_link`
        writer; *role* is recovered by :meth:`Folder.typed_out_edges`.

        Args:
            ref: The Concept being cited.
            text: Optional link label; defaults to *ref*'s name.
            role: The declared :class:`~molexp.workspace.edges.EdgeRole`; defaults
                to :data:`~molexp.workspace.edges.DEFAULT_EDGE_ROLE`.
        """
        append_link(self, ref, text=text, role=role)

    # -- typed meta.yaml (tags / status) -- knowledge-docs-05 ------------------

    def read_note_meta(self) -> NoteMeta:
        """Load this note's typed document ``meta.yaml`` as a :class:`NoteMeta`.

        A legacy bare marker (only ``{type, id}``) -- or an absent ``meta.yaml``
        -- reads back with the additive defaults (``tags == []`` /
        ``status == "active"``), so no migration is needed.
        """
        fpath = self._fs.join(self.resolve(), META_YAML_FILENAME)
        if not self._fs.exists(fpath):
            return NoteMeta(type=self._kind, id=self._name)
        return cast("NoteMeta", NoteMeta.from_yaml(self._fs.read_text(fpath)))

    def write_note_meta(self, meta: NoteMeta) -> None:
        """Atomically write this note's typed document ``meta.yaml``.

        The on-disk ``type`` is stamped to this Concept's ``kind`` (``note.note``)
        and ``id`` to its name, mirroring
        :meth:`ReferenceConcept.write_reference_meta`, so :func:`concept_from_dir`
        rebuilds a :class:`Note` and identity stays path-derived. Any other keys
        on *meta* are preserved verbatim (``ConceptMeta`` is ``extra="allow"``).
        """
        meta = meta.model_copy(update={"type": self._kind, "id": self._name})
        fpath = self._fs.join(self.path(), META_YAML_FILENAME)
        self._fs.atomic_write_text(fpath, meta.to_yaml())

    def tags(self) -> list[str]:
        """Return this note's categorical tags (``[]`` when untagged)."""
        return list(self.read_note_meta().tags)

    def status(self) -> str:
        """Return this note's lifecycle status (``"active"`` by default)."""
        return self.read_note_meta().status

    def set_tags(self, tags: Iterable[str]) -> None:
        """Set this note's tags, preserving ``status`` and any other meta keys."""
        current = self.read_note_meta()
        self.write_note_meta(current.model_copy(update={"tags": list(tags)}))

    def set_status(self, status: str) -> None:
        """Set this note's status, preserving ``tags`` and any other meta keys."""
        current = self.read_note_meta()
        self.write_note_meta(current.model_copy(update={"status": status}))


@concept_type(REFERENCE_KIND)
class ReferenceConcept(Folder):
    """A reference Concept — a bibliographic record (one Concept per ref).

    Structured bib fields live in ``meta.yaml`` (:class:`ReferenceMeta`); the
    human-readable citation text lives in ``index.md``. PDFs are pointed at via
    ``ReferenceMeta.pdf_path`` / ``pdf_asset_id`` — never copied.

    Named ``ReferenceConcept`` (not ``Reference``) to keep the concept type
    name unambiguous in the workspace namespace.
    """

    DEFAULT_KIND: ClassVar[str] = REFERENCE_KIND

    def __init__(
        self,
        *,
        parent: Folder | None = None,
        name: str,
        kind: str = REFERENCE_KIND,
        root_path: PathArg | None = None,
        fs: FileSystem | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name, kind=kind, root_path=root_path, fs=fs)

    def read_ref_meta(self) -> ReferenceMeta:
        """Load this reference's typed bib ``meta.yaml`` as a :class:`ReferenceMeta`."""
        fpath = self._fs.join(self.resolve(), META_YAML_FILENAME)
        return cast("ReferenceMeta", ReferenceMeta.from_yaml(self._fs.read_text(fpath)))

    def write_reference_meta(self, meta: ReferenceMeta) -> None:
        """Atomically write this reference's typed bib ``meta.yaml``.

        The on-disk ``type`` is stamped to this Concept's ``kind``
        (``reference.reference``) — the registered concept type — so
        :func:`concept_from_dir` rebuilds a :class:`ReferenceConcept` (workspace
        uses dotted kinds; ``ReferenceMeta.type`` itself defaults to the bare
        ``"reference"`` for the OKF bib payload).
        """
        meta = meta.model_copy(update={"type": self._kind, "id": self._name})
        fpath = self._fs.join(self.path(), META_YAML_FILENAME)
        self._fs.atomic_write_text(fpath, meta.to_yaml())

    def write_ref_meta(self, meta: ReferenceMeta) -> None:
        """Deprecated alias for :meth:`write_reference_meta`.

        Renamed to the full-word spelling that matches
        :meth:`Note.write_note_meta`; this alias warns and delegates.
        """
        warnings.warn(
            "ReferenceConcept.write_ref_meta is deprecated; "
            "use ReferenceConcept.write_reference_meta instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_reference_meta(meta)

    def citation(self) -> str:
        """Return the human-readable citation text (its ``index.md``)."""
        return self.read_index()

    def set_citation(self, text: str) -> None:
        """Set the human-readable citation text (its ``index.md``)."""
        self.write_index(text)


__all__ = ["NOTE_KIND", "REFERENCE_KIND", "Note", "ReferenceConcept"]
