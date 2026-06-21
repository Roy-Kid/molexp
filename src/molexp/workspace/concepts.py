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

from typing import ClassVar, cast

from molexp.knowledge.types import concept_type

from .folder import META_YAML_FILENAME, Folder, append_link
from .fs import FileSystem, PathArg
from .reference_meta import ReferenceMeta

NOTE_KIND = "note.note"
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

    def cite(self, ref: Folder, *, text: str | None = None) -> None:
        """Cite *ref* — append a markdown link resolvable via :meth:`out_edges`."""
        append_link(self, ref, text=text)


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

    def write_ref_meta(self, meta: ReferenceMeta) -> None:
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

    def citation(self) -> str:
        """Return the human-readable citation text (its ``index.md``)."""
        return self.read_index()

    def set_citation(self, text: str) -> None:
        """Set the human-readable citation text (its ``index.md``)."""
        self.write_index(text)


__all__ = ["NOTE_KIND", "REFERENCE_KIND", "Note", "ReferenceConcept"]
