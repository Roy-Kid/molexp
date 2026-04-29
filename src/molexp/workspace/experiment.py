"""Experiment entity — one directory per parameter combination.

An Experiment binds a workflow to a concrete set of parameters plus a
replica configuration (``n_replicas`` × ``seeds``).  Replicas under the
same Experiment share parameters; they differ only in their random seed.

Construction is side-effect free; ``project.experiment(...)`` materializes
on disk at call-time (idempotent: if an experiment with the same slug
already exists, it is loaded and returned).
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .project import Project
    from .workspace import Workspace

from molexp.workflow.context import TaskContext
from molexp.workflow.spec import Workflow, WorkflowSpec
from molexp.workflow.task import Task

from .assets import AssetScope, AssetsView, DataAssetLibrary
from .base import (
    _atomic_write_json,
    _list_children,
    _load_metadata,
    _rebuild_container_index,
    _reconstruct,
    _save_metadata,
)
from .models import ExperimentMetadata, RunMetadata, WorkflowSnapshotRef
from .run import Run
from .utils import generate_id

# Default replica seeds — deterministic, well-separated
_DEFAULT_SEEDS = [42, 123, 456, 789, 1234]


class _EntryTask(Task):
    """Wraps a bare ``fn(RunContext) -> None`` into a workflow Task."""

    def __init__(self, fn: Callable) -> None:
        self._fn = fn

    async def execute(self, ctx: TaskContext) -> None:
        run_ctx = ctx.run_context
        if run_ctx is None:
            fn_name = getattr(self._fn, "__name__", None) or "anonymous"
            raise RuntimeError(
                f"{fn_name}() requires a RunContext, but the "
                "workflow was executed without a workspace run."
            )
        if asyncio.iscoroutinefunction(self._fn):
            await self._fn(run_ctx)
        else:
            # Run sync bodies in a worker thread so blocking I/O (e.g.
            # ``time.sleep``) does not stall sibling replicas in the same
            # event loop.  Preserve the original semantics where a sync
            # callable that returns an awaitable is still awaited.
            result = await asyncio.to_thread(self._fn, run_ctx)
            if asyncio.iscoroutine(result) or inspect.isawaitable(result):
                await result


def _promote_to_workflow(fn: Callable, name: str) -> WorkflowSpec:
    """Promote a bare ``fn(RunContext)`` to a single-Task WorkflowSpec."""
    fn_name = getattr(fn, "__name__", None) or "anonymous"
    return Workflow(name=name).add(_EntryTask(fn), name=fn_name).build()


def _resolve_callable_entrypoint(fn: Callable) -> str:
    """Return ``"<file>:<qualname>"`` for a module-level callable."""
    qualname = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    if qualname is None or "<locals>" in qualname or "<lambda>" in qualname:
        raise ValueError(
            f"cannot determine an importable entrypoint for {fn!r}: "
            "define it at module scope (not nested / lambda) so the "
            "worker can re-import it."
        )
    file_path = Path(inspect.getfile(fn)).resolve()
    return f"{file_path}:{qualname}"


def _resolve_spec_entrypoint(spec: WorkflowSpec) -> str:
    """Return ``"<file>:<varname>"`` for *spec*.

    A ``WorkflowSpec`` carries no name; the worker re-imports it by
    looking up the variable that holds it.  We find that module by
    asking the first registered task (which always lives in the same
    user module that assembled the spec) for its source, then scan
    that module's globals for a binding to *spec* by identity.
    """
    mod = inspect.getmodule(spec._tasks[0].fn_or_class)
    file_path = Path(inspect.getfile(mod)).resolve()
    for var, val in vars(mod).items():
        if val is spec:
            return f"{file_path}:{var}"
    raise ValueError(
        f"cannot determine an importable entrypoint: {spec!r} is not "
        f"bound to a module-level variable in {file_path}.  Assign the "
        "spec to a name at module scope so the worker can re-import it."
    )


class Experiment:
    """Repeatable experiment bound to a workflow and a concrete parameter set.

    Example::

        exp = project.experiment(
            "lr-1e-3",
            params={"lr": 1e-3},
            n_replicas=3,
        )
        exp.set_workflow(train_fn)
    """

    def __init__(
        self,
        name: str,
        project: Project,
        id: str | None = None,
        *,
        params: dict[str, Any] | None = None,
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
        self._workflow: WorkflowSpec | None = None
        # Captured at ``set_workflow()`` time as ``"<file>:<qualname>"`` so
        # the worker can pull the workflow object back without re-running
        # the entire user script.
        self._workflow_entrypoint: str | None = None
        # Snapshot of the JSON IR when bound from a dict; ``None`` when
        # bound from a Python ``WorkflowSpec`` / callable.
        self._workflow_ir: dict[str, Any] | None = None

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
    def parameter_space(self) -> dict[str, Any]:
        return self.metadata.parameter_space

    @property
    def params(self) -> dict[str, Any]:
        """Concrete parameter dict bound to this experiment."""
        return self.metadata.parameter_space

    @property
    def n_replicas(self) -> int:
        return self.metadata.n_replicas

    @property
    def seeds(self) -> list[int] | None:
        return self.metadata.seeds

    @property
    def workflow(self) -> WorkflowSpec | None:
        """The bound workflow (always ``WorkflowSpec`` or ``None``).

        Lazy-loads from ``experiment_dir/workflow.json`` (the IR file) on
        first access if no workflow has been bound in-process. This
        recovers IR-bound workflows after a server restart, where
        :meth:`set_workflow` was called in a previous process.
        """
        cached = getattr(self, "_workflow", None)
        if cached is not None:
            return cached
        ir = self._read_workflow_ir_from_disk()
        if ir is None:
            return None
        from molexp.workflow.spec import WorkflowSpec as _WorkflowSpec

        spec = _WorkflowSpec.from_dict(ir)
        self._workflow = spec
        self._workflow_ir = ir
        return spec

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

    # ── Workflow binding ────────────────────────────────────────────────

    def set_workflow(
        self,
        workflow: WorkflowSpec | Callable | dict[str, Any],
    ) -> None:
        """Bind a workflow to this experiment.

        Accepted forms:

        - A compiled :class:`WorkflowSpec`.
        - A bare ``fn(RunContext)`` callable; auto-promoted to a
          single-Task ``WorkflowSpec``.
        - A JSON IR ``dict`` matching ``schema/workflow.json``;
          compiled via :meth:`WorkflowSpec.from_dict` and persisted to
          ``experiment_dir/workflow.json`` so the binding survives
          process restarts.

        For Python-side bindings (spec / callable), captures an
        *entrypoint* — ``"<absolute_file>:<qualname>"`` — so a worker
        process can re-import the workflow without re-running the entire
        user script. IR bindings need no entrypoint: the IR itself is
        the durable artifact.

        Raises:
            TypeError: If *workflow* is not a ``WorkflowSpec``, callable,
                or dict.
            ValueError: If a workflow is already bound, the entrypoint
                cannot be resolved, or the IR fails to compile.
        """
        if getattr(self, "_workflow", None) is not None:
            raise ValueError(f"Experiment {self.name!r} already has a workflow bound.")
        if isinstance(workflow, dict):
            from molexp.workflow.spec import WorkflowSpec as _WorkflowSpec

            spec = _WorkflowSpec.from_dict(workflow)
            self._workflow = spec
            self._workflow_ir = dict(workflow)
            self._workflow_entrypoint = None
            self._persist_workflow_ir(self._workflow_ir)
            return
        if isinstance(workflow, WorkflowSpec):
            self._workflow = workflow
            self._workflow_entrypoint = _resolve_spec_entrypoint(workflow)
        elif callable(workflow):
            self._workflow = _promote_to_workflow(workflow, self.name)
            self._workflow_entrypoint = _resolve_callable_entrypoint(workflow)
        else:
            raise TypeError(
                f"Expected WorkflowSpec, callable, or IR dict, got {type(workflow).__name__}"
            )

    # ── Workflow IR persistence ─────────────────────────────────────────

    @property
    def workflow_ir_path(self) -> Path:
        """Path to the on-disk workflow IR file (may not exist)."""
        return self.experiment_dir / "workflow.json"

    def _persist_workflow_ir(self, ir: dict[str, Any]) -> None:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.workflow_ir_path, ir)

    def _read_workflow_ir_from_disk(self) -> dict[str, Any] | None:
        path = self.workflow_ir_path
        if not path.exists():
            return None
        try:
            with open(path, "r") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

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
        parameters: dict[str, Any] | None = None,
        *,
        id: str | None = None,
        target: str | None = None,
    ) -> Run:
        """Get-or-create a run (idempotent, materialized immediately).

        If a run with the same ID already exists on disk, load and return it.
        Otherwise, construct + materialize a new Run.
        """
        # Always emit a snapshot when an entrypoint was captured, even
        # without a recorded workflow_source — the entrypoint itself is
        # what the worker needs.
        source = self.metadata.workflow_source or self._workflow_entrypoint
        snapshot = None
        if source is not None or self._workflow_entrypoint is not None:
            snapshot = WorkflowSnapshotRef(
                source=source or "",
                entrypoint=self._workflow_entrypoint,
                git_commit=self.metadata.git_commit,
            )

        r = Run(
            experiment=self,
            parameters=parameters,
            id=id,
            workflow_snapshot=snapshot,
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
