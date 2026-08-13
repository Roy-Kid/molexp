"""``KnowledgeItem`` — a typed, source-linked OKF Concept.

A ``KnowledgeItem`` is a Note-shaped Concept whose ``meta.json`` carries a typed
head (:class:`KnowledgeMeta`: a :data:`KnowledgeKind` plus a **required,
non-empty** list of :class:`SourceRef`) and whose human narrative lives in
``index.md``.

**The load-bearing invariant** — every KnowledgeItem carries ≥1 ``SourceRef`` —
is enforced at :class:`KnowledgeMeta` construction: a sourceless meta **fails
loudly** (``ValidationError``). So no execution-derived knowledge can be created
unsourced (integration.md §5.2, coordination invariant #4).

**Sources vs edges.** ``meta.json`` (``KnowledgeMeta.sources``) is the
*authoritative* typed record of every source (any :data:`SourceKind`, incl.
content-hash / file references that have no in-tree home). Where a source *is* an
in-tree Folder (a ``Run`` / ``Experiment`` / Concept), :meth:`KnowledgeItem.cite`
also writes a **typed OKF out-edge** (reusing the P0.1 edge role) so the
knowledge graph is traversable and the item is *reachable from* what it cites.

Follows the ``ReferenceConcept`` / ``ReferenceMeta`` precedent; registered against
the shared ``@concept_type`` registry so :func:`concept_from_dir` rebuilds it.
"""

from __future__ import annotations

from typing import ClassVar, Literal, cast, get_args

from pydantic import BaseModel, field_validator

from molexp.knowledge.types import concept_type

from .concept_meta import ConceptMeta
from .edges import EdgeRole
from .folder import META_JSON_FILENAME, Folder, append_link
from .fs import FileSystem, PathArg

KNOWLEDGE_ITEM_KIND = "knowledge.item"

KnowledgeKind = Literal[
    "Observation",
    "Decision",
    "Assumption",
    "Constraint",
    "Finding",
    "FailureAnalysis",
    "ProtocolNote",
    "ParameterRationale",
    "OpenQuestion",
]
"""The typed category of a knowledge item."""

KNOWLEDGE_KINDS: tuple[KnowledgeKind, ...] = get_args(KnowledgeKind)
"""The KnowledgeKind vocabulary, in declaration order."""


def parse_knowledge_kind(value: str) -> KnowledgeKind:
    """Validate an untrusted ``kind`` string against :data:`KnowledgeKind`.

    Agent tools and CLI flags carry the kind as a free ``str``. Passing that
    straight into a ``Literal``-typed API only defers the failure to model
    construction, mid-turn; validating at the boundary yields one clear error
    naming the whole vocabulary.

    Raises:
        ValueError: If *value* is not one of :data:`KNOWLEDGE_KINDS`.
    """
    for kind in KNOWLEDGE_KINDS:
        if value == kind:
            return kind
    raise ValueError(
        f"unknown knowledge kind {value!r}; expected one of: {', '.join(KNOWLEDGE_KINDS)}"
    )


SourceKind = Literal[
    "artifact",
    "run",
    "experiment",
    "file",
    "decision",
    "agent_action",
    "reference",
]
"""The kind of canonical object a :class:`SourceRef` points at."""


class SourceRef(BaseModel, frozen=True):
    """A typed pointer to an existing canonical object a KnowledgeItem derives from.

    Attributes:
        kind: What sort of object *ref* names.
        ref: The identifier — a content hash, ``run_id``, path, proposal id, or
            reference id, per *kind*.
        span: An optional sub-location (line range / cell / figure region).
    """

    kind: SourceKind
    ref: str
    span: str | None = None


class KnowledgeMeta(ConceptMeta):
    """The typed ``meta.json`` head of a :class:`KnowledgeItem`.

    Extends :class:`ConceptMeta` (frozen, ``extra="allow"``). ``sources`` is
    **required and non-empty** — the single chokepoint that makes a sourceless
    knowledge item impossible.
    """

    type: str = KNOWLEDGE_ITEM_KIND
    kind: KnowledgeKind
    sources: list[SourceRef]
    status: Literal["active", "stale", "superseded", "conflicting"] = "active"
    supersedes: tuple[str, ...] = ()
    confidence: float | None = None
    created_by: str

    @field_validator("sources")
    @classmethod
    def _require_at_least_one_source(cls, value: list[SourceRef]) -> list[SourceRef]:
        """Reject an empty ``sources`` list — the source-attribution invariant."""
        if not value:
            raise ValueError(
                "a KnowledgeItem must carry at least one SourceRef — source "
                "attribution is required (no unsourced knowledge)"
            )
        return value


@concept_type(KNOWLEDGE_ITEM_KIND)
class KnowledgeItem(Folder):
    """A typed, source-linked knowledge Concept (see the module docstring)."""

    DEFAULT_KIND: ClassVar[str] = KNOWLEDGE_ITEM_KIND

    def __init__(
        self,
        *,
        parent: Folder | None = None,
        name: str,
        kind: str = KNOWLEDGE_ITEM_KIND,
        root_path: PathArg | None = None,
        fs: FileSystem | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name, kind=kind, root_path=root_path, fs=fs)

    # ── narrative body (index.md) ─────────────────────────────────────────

    def body(self) -> str:
        """Return the human narrative (its ``index.md``)."""
        return self.read_index()

    def set_body(self, text: str) -> None:
        """Set the human narrative (its ``index.md``)."""
        self.write_index(text)

    # ── typed head (meta.json) ────────────────────────────────────────────

    def read_knowledge_meta(self) -> KnowledgeMeta:
        """Load this item's typed :class:`KnowledgeMeta` from ``meta.json``.

        Raises loudly (``ValidationError``) if the on-disk head is not a valid
        KnowledgeMeta — e.g. missing ``kind`` or an empty ``sources`` list.
        """
        fpath = self._fs.join(self.resolve(), META_JSON_FILENAME)
        if not self._fs.exists(fpath):
            legacy = self._fs.join(self.resolve(), "meta.json")
            return cast("KnowledgeMeta", KnowledgeMeta.from_yaml(self._fs.read_text(legacy)))
        return cast("KnowledgeMeta", KnowledgeMeta.from_json(self._fs.read_text(fpath)))

    def write_knowledge_meta(self, meta: KnowledgeMeta) -> None:
        """Atomically write this item's typed ``meta.json``.

        The on-disk ``type`` / ``id`` are stamped to this Concept's ``kind`` /
        ``name`` (mirroring ``ReferenceConcept.write_reference_meta``) so
        :func:`concept_from_dir` rebuilds a :class:`KnowledgeItem`.
        """
        meta = meta.model_copy(update={"type": self._kind, "id": self._name})
        fpath = self._fs.join(self.path(), META_JSON_FILENAME)
        self._fs.atomic_write_text(fpath, meta.to_json())

    # ── typed provenance edge (reuse P0.1) ────────────────────────────────

    def cite(self, source: Folder, *, role: EdgeRole = "derived_from") -> None:
        """Record a typed provenance out-edge to an in-tree Folder *source*.

        A thin delegator over :func:`~molexp.workspace.folder.append_link` (the
        single markdown-edge writer). Use for in-tree sources (a ``Run`` /
        ``Experiment`` / Concept); content-hash / file sources have no Folder to
        point at and live only in ``meta.json``.

        Args:
            source: The in-tree Folder this item derives from / cites.
            role: The declared :class:`~molexp.workspace.edges.EdgeRole`
                (default ``"derived_from"``).
        """
        append_link(self, source, role=role)


__all__ = [
    "KNOWLEDGE_ITEM_KIND",
    "KNOWLEDGE_KINDS",
    "KnowledgeItem",
    "KnowledgeKind",
    "KnowledgeMeta",
    "SourceKind",
    "SourceRef",
    "parse_knowledge_kind",
]
