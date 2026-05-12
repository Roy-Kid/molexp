"""Experiment entity — one directory per parameter combination.

Inherits :class:`Folder` (sub-spec 02) so it participates in the
unified workspace folder abstraction: ``kind`` is
:data:`WORKSPACE_EXPERIMENT_KIND`, ``parent`` is the owning
:class:`Project`.

An Experiment is a parameter-space container plus replica configuration
(``n_replicas`` × ``seeds``). Replicas under the same Experiment share
parameters; they differ only in their random seed.

Workspace does **not** know about workflows — pairing an Experiment
with a workflow is the caller's concern. Use the workflow layer to
build a ``WorkflowSpec`` and pass the workspace ``Run`` to its
``execute(run=...)`` method:

    >>> exp = project.Experiment("lr-1e-3", params={"lr": 1e-3})
    >>> run = exp.Run()
    >>> result = await my_workflow_spec.execute(run=run)

Construction is side-effect free; ``project.Experiment(...)``
materializes on disk at call-time (idempotent: if an experiment with
the same slug already exists, it is loaded and returned).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .project import Project
    from .workspace import Workspace

from molexp._typing import JSONValue

from .assets import AssetScope, AssetsView, DataAssetLibrary
from .base import (
    _load_metadata,
    _rebuild_container_index,
    _reconstruct,
    _save_metadata,
)
from .errors import ExperimentExistsError, ExperimentNotFoundError
from .folder import (
    WORKSPACE_EXPERIMENT_KIND,
    WORKSPACE_RUN_KIND,
    Folder,
)
from .models import ExperimentMetadata, FolderMetadata
from .run import Run
from .utils import generate_id

# Default replica seeds — deterministic, well-separated
_DEFAULT_SEEDS = [42, 123, 456, 789, 1234]


class Experiment(Folder):
    """Repeatable experiment — a parameter-space container.

    Example::

        exp = project.Experiment(
            "lr-1e-3",
            params={"lr": 1e-3},
            n_replicas=3,
        )
        run = exp.Run()
        # Workflow execution is the caller's concern; workspace just
        # provides the Run that workflow.execute(run=...) operates on.
    """

    _exists_error_cls = ExperimentExistsError
    _not_found_error_cls = ExperimentNotFoundError

    def __init__(
        self,
        *,
        parent: Project | None = None,
        name: str,
        kind: str = WORKSPACE_EXPERIMENT_KIND,
        project: Project | None = None,
        id: str | None = None,
        params: dict[str, JSONValue] | None = None,
        n_replicas: int = 1,
        seeds: list[int] | None = None,
        workflow_source: str | None = None,
        workflow_type: str | None = None,
        git_commit: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
        default_target: str | None = None,
        _entity_metadata: ExperimentMetadata | None = None,
    ) -> None:
        resolved_parent = parent if parent is not None else project
        if resolved_parent is None:
            raise ValueError("Experiment: parent (or project) is required")

        meta = (
            _entity_metadata
            if _entity_metadata is not None
            else ExperimentMetadata(
                id=id if id is not None else generate_id(),
                name=name,
                description=description,
                tags=list(tags) if tags is not None else [],
                workflow_source=workflow_source,
                workflow_type=workflow_type,
                parameter_space=dict(params) if params else {},
                git_commit=git_commit,
                n_replicas=n_replicas,
                seeds=list(seeds) if seeds is not None else None,
                default_target=default_target,
            )
        )

        self._parent = resolved_parent
        self._name = meta.id
        self._kind = kind
        self._root_path = None
        self._metadata = FolderMetadata(
            id=meta.id,
            name=meta.name,
            kind=kind,
            created_at=meta.created_at,
            updated_at=meta.created_at,
        )
        self._children_cache = {}

        # Entity-specific state
        self._entity_metadata: ExperimentMetadata = meta
        self._data_assets: DataAssetLibrary | None = None

    # ── Folder hooks ─────────────────────────────────────────────────────

    def _compute_path(self) -> Path:
        return self.experiment_dir

    @classmethod
    def _child_dir(cls, parent: Folder, derived_id: str) -> Path:
        """:class:`Folder.attach` hook — experiments live under ``experiments/<id>/``."""
        return parent.path() / "experiments" / derived_id

    @classmethod
    def _from_disk(cls, child_dir: Path, parent: Folder) -> Experiment:
        """:class:`Folder.attach` hook — load ``experiment.json`` and rebuild entity state."""
        meta = _load_metadata(ExperimentMetadata, child_dir / "experiment.json")
        return _reconstruct(
            cls,
            {
                "_parent": parent,
                "_name": meta.id,
                "_kind": WORKSPACE_EXPERIMENT_KIND,
                "_root_path": None,
                "_metadata": FolderMetadata(
                    id=meta.id,
                    name=meta.name,
                    kind=WORKSPACE_EXPERIMENT_KIND,
                    created_at=meta.created_at,
                    updated_at=meta.created_at,
                ),
                "_children_cache": {},
                "_entity_metadata": meta,
                "_data_assets": None,
            },
        )

    # ── Properties (entity-specific) ─────────────────────────────────────

    @property
    def project(self) -> Project:
        """The owning :class:`Project` (alias for :attr:`Folder.parent`)."""
        if self._parent is None:  # pragma: no cover — Experiment always has a parent
            raise RuntimeError("Experiment has no parent project")
        return cast("Project", self._parent)

    @property
    def metadata(self) -> ExperimentMetadata:  # type: ignore[override]
        return self._entity_metadata

    @metadata.setter
    def metadata(self, value: ExperimentMetadata) -> None:
        self._entity_metadata = value

    @property
    def id(self) -> str:
        return self._entity_metadata.id

    @property
    def name(self) -> str:
        return self._entity_metadata.name

    @property
    def created_at(self):
        return self._entity_metadata.created_at

    @property
    def description(self) -> str:
        return self._entity_metadata.description

    @property
    def tags(self) -> list[str]:
        return self._entity_metadata.tags

    @property
    def workflow_source(self) -> str | None:
        return self._entity_metadata.workflow_source

    @property
    def parameter_space(self) -> dict[str, JSONValue]:
        return self._entity_metadata.parameter_space

    @property
    def params(self) -> dict[str, JSONValue]:
        """Concrete parameter dict bound to this experiment."""
        return self._entity_metadata.parameter_space

    @property
    def n_replicas(self) -> int:
        return self._entity_metadata.n_replicas

    @property
    def seeds(self) -> list[int] | None:
        return self._entity_metadata.seeds

    @property
    def workspace(self) -> Workspace:
        return self.project.workspace

    @property
    def experiment_dir(self) -> Path:
        return self.project.project_dir / "experiments" / self.id

    @property
    def scope(self) -> AssetScope:
        return AssetScope(kind="experiment", ids=(self.project.id, self.id))

    @property
    def assets(self) -> AssetsView:
        """Scope-filtered catalog view (read-only queries)."""
        return AssetsView(self.project.workspace.catalog, self.scope)

    @property
    def data_assets(self) -> DataAssetLibrary:
        if self._data_assets is None:
            self._data_assets = DataAssetLibrary(
                self.experiment_dir, self.scope, self.project.workspace.catalog
            )
        return self._data_assets

    def get_seeds(self) -> list[int]:
        """Return replica seeds (length == ``n_replicas``)."""
        seeds = self._entity_metadata.seeds
        if seeds is not None:
            return list(seeds[: self.n_replicas])
        out = list(_DEFAULT_SEEDS)
        while len(out) < self.n_replicas:
            out.append(out[-1] + 111)
        return out[: self.n_replicas]

    # ── Persistence ─────────────────────────────────────────────────────

    def materialize(self) -> None:
        """Create filesystem structure and persist metadata (non-recursive)."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        _save_metadata(self._entity_metadata, self.experiment_dir / "experiment.json")
        self._catalog_upsert()

    def save(self) -> None:
        """Persist current metadata to disk."""
        _save_metadata(self._entity_metadata, self.experiment_dir / "experiment.json")
        self._catalog_upsert()

    def _catalog_upsert(self) -> None:
        ws = self.project.workspace
        meta = self._entity_metadata
        ws.catalog.upsert_experiment(
            {
                "experiment_id": meta.id,
                "project_id": self.project.id,
                "name": meta.name,
                "description": meta.description,
                "tags": list(meta.tags),
                "parameter_space": dict(meta.parameter_space),
                "n_replicas": meta.n_replicas,
                "workflow_source": meta.workflow_source,
                "workflow_type": meta.workflow_type,
                "path": str(self.experiment_dir.relative_to(ws.root)),
                "created_at": meta.created_at.isoformat(),
                "updated_at": meta.created_at.isoformat(),
            }
        )

    # ── Run operations (typed wrappers over attach/create_child/get_child) ──

    def Run(
        self,
        parameters: dict[str, JSONValue] | None = None,
        *,
        id: str | None = None,
        target: str | None = None,
        workflow_snapshot: dict[str, JSONValue] | None = None,
    ) -> Run:
        """Idempotent constructor — return existing run if found, else create.

        If a run with the same ID already exists on disk, load and
        return it. Otherwise, construct + materialize a new Run.

        ``workflow_snapshot`` is an opaque JSON-shaped payload — the
        canonical structure lives in
        :class:`molexp.workflow.snapshot_ref.WorkflowSnapshotRef` but
        workspace doesn't know that type.

        For "must be new" semantics, use :meth:`create_run`. For
        "must already exist", use :meth:`run`.
        """
        resolved_id = id if id is not None else generate_id()
        resolved_target = target if target is not None else self._entity_metadata.default_target
        r = self.attach(
            resolved_id,
            kind=WORKSPACE_RUN_KIND,
            child_cls=Run,
            id=resolved_id,
            parameters=parameters,
            workflow_snapshot=workflow_snapshot,
            target=resolved_target,
        )
        if not isinstance(r, Run):  # pragma: no cover — defensive
            raise TypeError(f"attach returned {type(r).__name__}, expected Run")
        self._refresh_runs_index()
        return r

    def create_run(
        self,
        parameters: dict[str, JSONValue] | None = None,
        *,
        id: str | None = None,
        target: str | None = None,
        workflow_snapshot: dict[str, JSONValue] | None = None,
    ) -> Run:
        """Strict constructor — raise :class:`RunExistsError` if exists."""
        resolved_id = id if id is not None else generate_id()
        resolved_target = target if target is not None else self._entity_metadata.default_target
        r = self.create_child(
            resolved_id,
            kind=WORKSPACE_RUN_KIND,
            child_cls=Run,
            id=resolved_id,
            parameters=parameters,
            workflow_snapshot=workflow_snapshot,
            target=resolved_target,
        )
        if not isinstance(r, Run):  # pragma: no cover — defensive
            raise TypeError(f"create_child returned {type(r).__name__}, expected Run")
        self._refresh_runs_index()
        return r

    def run(self, run_id: str) -> Run:
        """Strict getter — raise :class:`RunNotFoundError` if absent."""
        r = self.get_child(run_id, kind=WORKSPACE_RUN_KIND, child_cls=Run)
        if not isinstance(r, Run):  # pragma: no cover — defensive
            raise TypeError(f"get_child returned {type(r).__name__}, expected Run")
        return r

    def list_runs(self) -> list[Run]:
        """List all runs by scanning the ``runs/`` directory."""
        result: list[Run] = []
        runs_dir = self.experiment_dir / "runs"
        if not runs_dir.exists():
            return result
        for entry in sorted(runs_dir.iterdir()):
            if entry.is_dir() and (entry / "run.json").exists():
                result.append(Run._from_disk(entry, self))
        return result

    def children(self, kind: str | None = None) -> list[Folder]:
        """List child folders, optionally filtered by ``kind``.

        Experiment's only entity children are :class:`Run` instances
        under ``runs/``.
        """
        if kind is not None and kind != WORKSPACE_RUN_KIND:
            return []
        return list(self.list_runs())

    def delete_run(self, run_id: str) -> None:
        """Delete a run directory and cascade-drop its catalog rows.

        Raises:
            KeyError: If the run is not found.
        """
        import shutil

        run_dir = self.experiment_dir / "runs" / f"run-{run_id}"
        if not run_dir.exists():
            raise KeyError(f"Run '{run_id}' not found")
        shutil.rmtree(run_dir)
        self._children_cache.pop(run_id, None)
        self.project.workspace.catalog.remove_run(run_id)
        self._refresh_runs_index()

    # ── Internal ────────────────────────────────────────────────────────

    def _refresh_runs_index(self) -> None:
        _rebuild_container_index(
            container_dir=self.experiment_dir / "runs",
            index_filename="runs.json",
            metadata_filename="run.json",
            fields=["id", "status", "parameters", "profile", "created_at", "finished_at"],
        )
