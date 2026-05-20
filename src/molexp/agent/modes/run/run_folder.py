"""On-disk layout for a single RunMode execution — ``RunFolder``.

:class:`RunFolder` is the agent-layer :class:`molexp.workspace.Folder`
subclass RunMode owns (``kind = "agent.run"``). It is anchored to the
plan — mounted on the plan's :class:`~molexp.agent.modes.plan.plan_folder.PlanFolder`
(or any other workspace ``Folder``) via the generic ``add_folder`` — and
holds the *agent-facing* run summary:

- ``run_report.yaml`` — the typed :class:`RunReport`.
- ``progress/`` — projected :class:`~molexp.agent.modes.run.monitor.RunProgress`
  snapshots.
- ``repairs/`` — emitted :class:`~molexp.agent.modes._planning.PlanDiff` /
  :class:`~molexp.agent.modes.run.repair.RepairEscalation` records.

Execution records and asset / artifact lineage stay on the workspace
``Run`` / ``AssetCatalog`` — :class:`RunFolder` does *not* duplicate
them; it only adds the agent-facing report.

Construction is side-effect free; ``parent.add_folder(run)`` / paths
create directories lazily.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field

from molexp._typing import JSONValue
from molexp.agent.modes._planning import PlanDiff
from molexp.agent.modes.run.monitor import RunProgress
from molexp.agent.modes.run.repair import RepairEscalation
from molexp.agent.types import utc_now
from molexp.path import Path as MolexpPath
from molexp.workspace import Folder, FolderMetadata, atomic_write_text
from molexp.workspace.base import _load_metadata, _reconstruct, _save_metadata

if TYPE_CHECKING:
    from molexp.workspace.fs import PathArg

__all__ = ["AGENT_RUN_KIND", "RunFolder", "RunFolderMetadata", "RunReport"]

AGENT_RUN_KIND = "agent.run"
"""Folder ``kind`` for a single RunMode execution workspace."""

RUN_METADATA_FILENAME = "run_folder.json"
"""Per-:class:`RunFolder` metadata file (auto-derived from class name)."""

RUN_REPORT_FILENAME = "run_report.yaml"
"""The typed :class:`RunReport` file under a :class:`RunFolder`."""


class RunFolderMetadata(FolderMetadata, frozen=True):
    """Frozen metadata for a :class:`RunFolder`.

    Extends :class:`FolderMetadata` (so the workspace layer treats it as
    opaque ``Folder`` metadata) with the bound plan id.

    Attributes:
        plan_id: ``id`` of the plan this run executes, or ``""`` before
            it is bound.
    """

    plan_id: str = ""


class RunReport(BaseModel):
    """The agent-facing summary of one RunMode execution.

    Persisted to ``run_report.yaml`` and consumed by the review queue.

    Attributes:
        plan_id: The plan that was executed.
        status: The terminal outcome — ``"completed"`` / ``"failed"`` /
            ``"needs_clarification"``.
        run_id: The workspace ``Run`` id the workflow executed against,
            or ``None`` when execution never started.
        execution_id: The workflow execution id, or ``None``.
        progress: The final :class:`RunProgress` projection.
        repair_diffs: Any :class:`PlanDiff`\\ s emitted on unrecoverable
            failure.
        escalation: A :class:`RepairEscalation` when the run needs
            AuthorMode re-entry, else ``None``.
        started_at: When the run began.
        finished_at: When the run finished.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str
    status: str
    run_id: str | None = None
    execution_id: str | None = None
    progress: RunProgress
    repair_diffs: tuple[PlanDiff, ...] = ()
    escalation: RepairEscalation | None = None
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)


class RunFolder(Folder):
    """One RunMode execution workspace — ``kind = "agent.run"``.

    Construct with an explicit ``name`` (the run-report id) and mount via
    the generic :meth:`Folder.add_folder` on the plan's ``PlanFolder``::

        run_folder = plan_folder.add_folder(RunFolder(name="run-1"))
        run_folder.write_run_report(report)
    """

    def __init__(
        self,
        *,
        parent: Folder | None = None,
        name: str,
        kind: str = AGENT_RUN_KIND,
        plan_id: str = "",
        _entity_metadata: RunFolderMetadata | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name, kind=kind)
        meta = (
            _entity_metadata
            if _entity_metadata is not None
            else RunFolderMetadata(id=self._name, name=name, kind=kind, plan_id=plan_id)
        )
        self._entity_metadata: RunFolderMetadata = meta

    # ── Folder hooks ─────────────────────────────────────────────────────

    def resolve(self) -> MolexpPath:
        if self._parent is None:
            raise RuntimeError(
                f"RunFolder {self._name!r} is unmounted — mount via parent.add_folder()"
            )
        return type(self).child_dir(self._parent, self._name)

    @classmethod
    def child_dir(cls, parent: Folder, derived_id: str) -> MolexpPath:
        """Run folders live under ``<parent>/runs/<run_id>/``."""
        return MolexpPath(parent._fs.join(parent.path(), "runs", derived_id))

    @classmethod
    def from_disk(cls, child_dir: PathArg, parent: Folder) -> RunFolder:
        meta_path = parent._fs.join(child_dir, RUN_METADATA_FILENAME)
        meta = _load_metadata(RunFolderMetadata, meta_path, fs=parent._fs)
        folder_meta = FolderMetadata(
            id=meta.id,
            name=meta.name,
            kind=AGENT_RUN_KIND,
            created_at=meta.created_at,
            updated_at=meta.updated_at,
        )
        attrs = cls.base_from_disk_attrs(parent, folder_meta) | {"_entity_metadata": meta}
        return _reconstruct(cls, attrs)

    def materialize(self) -> None:
        self._fs.mkdir(self.path(), parents=True, exist_ok=True)
        _save_metadata(
            self._entity_metadata,
            self._fs.join(self.path(), RUN_METADATA_FILENAME),
            fs=self._fs,
        )

    def save(self) -> None:
        _save_metadata(
            self._entity_metadata,
            self._fs.join(self.path(), RUN_METADATA_FILENAME),
            fs=self._fs,
        )

    def _to_index_row(self) -> dict[str, JSONValue]:
        return cast("dict[str, JSONValue]", self._entity_metadata.model_dump(mode="json"))

    @property
    def metadata(self) -> RunFolderMetadata:  # type: ignore[override]
        return self._entity_metadata

    @property
    def plan_id(self) -> str:
        """The plan id this run folder is anchored to."""
        return self._entity_metadata.plan_id

    # ── Directory helper ─────────────────────────────────────────────────

    def _root(self) -> Path:
        """Resolve + mkdir the run-folder root, returning a local :class:`Path`."""
        path = Path(self.path())
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ── Report / progress writers ────────────────────────────────────────

    def write_run_report(self, report: RunReport) -> Path:
        """Persist the typed :class:`RunReport` to ``run_report.yaml``.

        The report round-trips back through :meth:`RunReport.model_validate`.
        """
        path = self._root() / RUN_REPORT_FILENAME
        payload = report.model_dump(mode="json")
        atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))
        return path

    def read_run_report(self) -> RunReport:
        """Load the persisted :class:`RunReport` from ``run_report.yaml``."""
        path = self._root() / RUN_REPORT_FILENAME
        data = yaml.safe_load(path.read_text())
        return RunReport.model_validate(data)

    def write_progress(self, label: str, progress: RunProgress) -> Path:
        """Persist one :class:`RunProgress` snapshot to ``progress/<label>.json``."""
        progress_dir = self._root() / "progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        path = progress_dir / f"{label}.json"
        atomic_write_text(path, progress.model_dump_json(indent=2))
        return path

    def write_repair_escalation(self, escalation: RepairEscalation) -> Path:
        """Persist a :class:`RepairEscalation` to ``repairs/escalation.json``."""
        repairs_dir = self._root() / "repairs"
        repairs_dir.mkdir(parents=True, exist_ok=True)
        path = repairs_dir / "escalation.json"
        atomic_write_text(path, escalation.model_dump_json(indent=2))
        return path
