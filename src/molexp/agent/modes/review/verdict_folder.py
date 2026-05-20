"""On-disk layout for a single ReviewMode verdict — ``ReviewVerdictFolder``.

:class:`ReviewVerdictFolder` is the agent-layer
:class:`molexp.workspace.Folder` subclass ReviewMode owns
(``kind = "agent.review"``). It is the read-only mode's **only** write
surface: ReviewMode produces a verdict and persists it here, and writes
nowhere else — never ``src/`` / ``tests/`` / ``ir/``, never a reviewed
target.

The folder mounts on any workspace ``Folder`` (typically the plan's
:class:`~molexp.agent.modes.plan.plan_folder.PlanFolder`) via the
generic ``add_folder`` and holds two files:

- ``verdict.yaml`` — the structured :class:`ReviewVerdict`.
- ``verdict.md`` — a human-readable rendering of the same verdict.

Construction is side-effect free; ``parent.add_folder(folder)`` /
:meth:`Folder.path` create directories lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

from molexp._typing import JSONValue
from molexp.agent.modes.review.renderers import render_verdict_markdown
from molexp.agent.modes.review.verdict import ReviewVerdict
from molexp.path import Path as MolexpPath
from molexp.workspace import Folder, FolderMetadata, atomic_write_text
from molexp.workspace.base import _load_metadata, _reconstruct, _save_metadata

if TYPE_CHECKING:
    from molexp.workspace.fs import PathArg

__all__ = [
    "AGENT_REVIEW_KIND",
    "REVIEW_METADATA_FILENAME",
    "ReviewVerdictFolder",
    "ReviewVerdictFolderMetadata",
]

AGENT_REVIEW_KIND = "agent.review"
"""Folder ``kind`` for a single ReviewMode verdict workspace."""

REVIEW_METADATA_FILENAME = "review_verdict_folder.json"
"""Per-:class:`ReviewVerdictFolder` metadata file (auto-derived from class name)."""

VERDICT_YAML_FILENAME = "verdict.yaml"
"""The structured :class:`ReviewVerdict` file under a :class:`ReviewVerdictFolder`."""

VERDICT_MD_FILENAME = "verdict.md"
"""The human-readable verdict file under a :class:`ReviewVerdictFolder`."""


class ReviewVerdictFolderMetadata(FolderMetadata, frozen=True):
    """Frozen metadata for a :class:`ReviewVerdictFolder`.

    Extends :class:`FolderMetadata` (so the workspace layer treats it as
    opaque ``Folder`` metadata) with the reviewed plan id.

    Attributes:
        plan_id: ``id`` of the plan whose artefact was reviewed, or ``""``
            before it is bound.
    """

    plan_id: str = ""


class ReviewVerdictFolder(Folder):
    """One ReviewMode verdict workspace — ``kind = "agent.review"``.

    Construct with an explicit ``name`` (the verdict-folder id) and mount
    via the generic :meth:`Folder.add_folder` on the plan's
    ``PlanFolder``::

        verdict_folder = plan_folder.add_folder(ReviewVerdictFolder(name="review"))
        verdict_folder.write_verdict(verdict)
    """

    def __init__(
        self,
        *,
        parent: Folder | None = None,
        name: str,
        kind: str = AGENT_REVIEW_KIND,
        plan_id: str = "",
        _entity_metadata: ReviewVerdictFolderMetadata | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name, kind=kind)
        meta = (
            _entity_metadata
            if _entity_metadata is not None
            else ReviewVerdictFolderMetadata(id=self._name, name=name, kind=kind, plan_id=plan_id)
        )
        self._entity_metadata: ReviewVerdictFolderMetadata = meta

    # ── Folder hooks ─────────────────────────────────────────────────────

    def resolve(self) -> MolexpPath:
        if self._parent is None:
            raise RuntimeError(
                f"ReviewVerdictFolder {self._name!r} is unmounted — mount via parent.add_folder()"
            )
        return type(self).child_dir(self._parent, self._name)

    @classmethod
    def child_dir(cls, parent: Folder, derived_id: str) -> MolexpPath:
        """Verdict folders live under ``<parent>/reviews/<id>/``."""
        return MolexpPath(parent._fs.join(parent.path(), "reviews", derived_id))

    @classmethod
    def from_disk(cls, child_dir: PathArg, parent: Folder) -> ReviewVerdictFolder:
        meta_path = parent._fs.join(child_dir, REVIEW_METADATA_FILENAME)
        meta = _load_metadata(ReviewVerdictFolderMetadata, meta_path, fs=parent._fs)
        folder_meta = FolderMetadata(
            id=meta.id,
            name=meta.name,
            kind=AGENT_REVIEW_KIND,
            created_at=meta.created_at,
            updated_at=meta.updated_at,
        )
        attrs = cls.base_from_disk_attrs(parent, folder_meta) | {"_entity_metadata": meta}
        return _reconstruct(cls, attrs)

    def materialize(self) -> None:
        self._fs.mkdir(self.path(), parents=True, exist_ok=True)
        _save_metadata(
            self._entity_metadata,
            self._fs.join(self.path(), REVIEW_METADATA_FILENAME),
            fs=self._fs,
        )

    def save(self) -> None:
        _save_metadata(
            self._entity_metadata,
            self._fs.join(self.path(), REVIEW_METADATA_FILENAME),
            fs=self._fs,
        )

    def _to_index_row(self) -> dict[str, JSONValue]:
        return cast("dict[str, JSONValue]", self._entity_metadata.model_dump(mode="json"))

    @property
    def metadata(self) -> ReviewVerdictFolderMetadata:  # type: ignore[override]
        return self._entity_metadata

    @property
    def plan_id(self) -> str:
        """The plan id this verdict folder is anchored to."""
        return self._entity_metadata.plan_id

    # ── Directory helper ─────────────────────────────────────────────────

    def _root(self) -> Path:
        """Resolve + mkdir the verdict-folder root, returning a local :class:`Path`."""
        path = Path(self.path())
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ── Verdict writers / readers ────────────────────────────────────────

    def write_verdict(self, verdict: ReviewVerdict) -> tuple[Path, Path]:
        """Persist ``verdict`` as ``verdict.yaml`` + ``verdict.md``.

        The YAML round-trips back through :meth:`ReviewVerdict.model_validate`;
        the Markdown is a human-readable rendering.

        Returns:
            The ``(yaml_path, md_path)`` pair written.
        """
        root = self._root()
        yaml_path = root / VERDICT_YAML_FILENAME
        md_path = root / VERDICT_MD_FILENAME
        payload = verdict.model_dump(mode="json")
        atomic_write_text(yaml_path, yaml.safe_dump(payload, sort_keys=False))
        atomic_write_text(md_path, render_verdict_markdown(verdict))
        return yaml_path, md_path

    def read_verdict(self) -> ReviewVerdict:
        """Load the persisted :class:`ReviewVerdict` from ``verdict.yaml``."""
        path = self._root() / VERDICT_YAML_FILENAME
        data = yaml.safe_load(path.read_text())
        return ReviewVerdict.model_validate(data)
