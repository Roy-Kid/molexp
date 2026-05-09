"""Experiment entity — one directory per parameter combination.

An Experiment is a parameter-space container plus replica configuration
(``n_replicas`` × ``seeds``). Replicas under the same Experiment share
parameters; they differ only in their random seed.

Workspace does **not** know about workflows — pairing an Experiment
with a workflow is the caller's concern. Use the workflow layer to
build a ``WorkflowSpec`` and pass the workspace ``Run`` to its
``execute(run=...)`` method:

    >>> exp = project.experiment("lr-1e-3", params={"lr": 1e-3})
    >>> run = exp.run()
    >>> result = await my_workflow_spec.execute(run=run)

Construction is side-effect free; ``project.experiment(...)``
materializes on disk at call-time (idempotent: if an experiment with
the same slug already exists, it is loaded and returned).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .project import Project
    from .workspace import Workspace

from molexp._typing import JSONValue

from .assets import AssetScope, AssetsView, DataAssetLibrary
from .base import (
    _list_children,
    _load_metadata,
    _rebuild_container_index,
    _reconstruct,
    _save_metadata,
)
from .models import ExperimentMetadata, RunMetadata
from .run import Run
from .utils import generate_id

# Default replica seeds — deterministic, well-separated
_DEFAULT_SEEDS = [42, 123, 456, 789, 1234]


class Experiment:
    """Repeatable experiment — a parameter-space container.

    Example::

        exp = project.experiment(
            "lr-1e-3",
            params={"lr": 1e-3},
            n_replicas=3,
        )
        run = exp.run()
        # Workflow execution is the caller's concern; workspace just
        # provides the Run that workflow.execute(run=...) operates on.
    """

    def __init__(
        self,
        name: str,
        project: Project,
        id: str | None = None,
        *,
        params: dict[str, JSONValue] | None = None,
        n_replicas: int = 1,
        seeds: list[int] | None = None,
        workflow_source: str | None = None,
        workflow_type: str | None = None,
        git_commit: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
        default_target: str | None = None,
    ) -> None:
        self.project = project
        self.metadata = ExperimentMetadata(
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
        self._data_assets: DataAssetLibrary | None = None

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def id(self) -> str:
        return self.metadata.id

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def created_at(self):
        return self.metadata.created_at

    @property
    def description(self) -> str:
        return self.metadata.description

    @property
    def tags(self) -> list[str]:
        return self.metadata.tags

    @property
    def workflow_source(self) -> str | None:
        return self.metadata.workflow_source

    @property
    def parameter_space(self) -> dict[str, JSONValue]:
        return self.metadata.parameter_space

    @property
    def params(self) -> dict[str, JSONValue]:
        """Concrete parameter dict bound to this experiment."""
        return self.metadata.parameter_space

    @property
    def n_replicas(self) -> int:
        return self.metadata.n_replicas

    @property
    def seeds(self) -> list[int] | None:
        return self.metadata.seeds

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
        seeds = self.metadata.seeds
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
        _save_metadata(self.metadata, self.experiment_dir / "experiment.json")
        self._catalog_upsert()

    def save(self) -> None:
        """Persist current metadata to disk."""
        _save_metadata(self.metadata, self.experiment_dir / "experiment.json")
        self._catalog_upsert()

    def _catalog_upsert(self) -> None:
        ws = self.project.workspace
        ws.catalog.upsert_experiment(
            {
                "experiment_id": self.metadata.id,
                "project_id": self.project.id,
                "name": self.metadata.name,
                "description": self.metadata.description,
                "tags": list(self.metadata.tags),
                "parameter_space": dict(self.metadata.parameter_space),
                "n_replicas": self.metadata.n_replicas,
                "workflow_source": self.metadata.workflow_source,
                "workflow_type": self.metadata.workflow_type,
                "path": str(self.experiment_dir.relative_to(ws.root)),
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.created_at.isoformat(),
            }
        )

    # ── Run operations ──────────────────────────────────────────────────

    def run(
        self,
        parameters: dict[str, JSONValue] | None = None,
        *,
        id: str | None = None,
        target: str | None = None,
        workflow_snapshot: dict[str, JSONValue] | None = None,
    ) -> Run:
        """Get-or-create a run (idempotent, materialized immediately).

        If a run with the same ID already exists on disk, load and
        return it. Otherwise, construct + materialize a new Run.

        ``workflow_snapshot`` is an opaque JSON-shaped payload — the
        canonical structure lives in
        :class:`molexp.workflow.snapshot_ref.WorkflowSnapshotRef` but
        workspace doesn't know that type. Callers that want to record
        workflow provenance pass the snapshot's ``model_dump(mode="json")``
        directly; everyone else passes ``None`` (the default).
        """
        r = Run(
            experiment=self,
            parameters=parameters,
            id=id,
            workflow_snapshot=workflow_snapshot,
            target=target if target is not None else self.metadata.default_target,
        )
        run_dir = self.experiment_dir / "runs" / f"run-{r.id}"
        if run_dir.exists():
            return self._load_run_from_dir(run_dir)
        r.materialize()
        self._refresh_runs_index()
        return r

    def get_run(self, run_id: str) -> Run | None:
        """Get run by ID."""
        run_dir = self.experiment_dir / "runs" / f"run-{run_id}"
        if not run_dir.exists():
            return None
        return self._load_run_from_dir(run_dir)

    def list_runs(self) -> list[Run]:
        """List all runs by scanning the ``runs/`` directory."""
        return _list_children(
            children_dir=self.experiment_dir / "runs",
            metadata_filename="run.json",
            metadata_cls=RunMetadata,
            child_cls=Run,
            attrs_factory=lambda m: {"experiment": self, "metadata": m},
        )

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

    def _load_run_from_dir(self, run_dir: Path) -> Run:
        meta = _load_metadata(RunMetadata, run_dir / "run.json")
        return _reconstruct(Run, {"experiment": self, "metadata": meta})
