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
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .project import Project
    from .workspace import Workspace

from molexp.workflow.context import TaskContext
from molexp.workflow.spec import WorkflowBuilder, WorkflowSpec
from molexp.workflow.task import Task

from .asset import AssetLibrary
from .base import _list_children, _load_metadata, _reconstruct, _save_metadata
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
            raise RuntimeError(
                f"{self._fn.__name__}() requires a RunContext, but the "
                "workflow was executed without a workspace run."
            )
        result = self._fn(run_ctx)
        if asyncio.iscoroutine(result) or inspect.isawaitable(result):
            await result


def _promote_to_workflow(fn: Callable, name: str) -> WorkflowSpec:
    """Promote a bare ``fn(RunContext)`` to a single-Task WorkflowSpec."""
    return WorkflowBuilder(name=name).add(_EntryTask(fn), name=fn.__name__).build()


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
    ) -> None:
        self.project = project
        self.metadata = ExperimentMetadata(
            id=id if id is not None else generate_id(),
            name=name,
            workflow_source=workflow_source,
            workflow_type=workflow_type,
            parameter_space=dict(params) if params else {},
            git_commit=git_commit,
            n_replicas=n_replicas,
            seeds=list(seeds) if seeds is not None else None,
        )
        self._assets_lib: AssetLibrary | None = None
        self._workflow: WorkflowSpec | None = None

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
        """The bound workflow (always ``WorkflowSpec`` or ``None``)."""
        return self._workflow

    @property
    def workspace(self) -> Workspace:
        return self.project.workspace

    @property
    def experiment_dir(self) -> Path:
        return self.project.project_dir / "experiments" / self.id

    @property
    def assets(self) -> AssetLibrary:
        if self._assets_lib is None:
            self._assets_lib = AssetLibrary(self.experiment_dir / "assets")
        return self._assets_lib

    # ── Workflow binding ────────────────────────────────────────────────

    def set_workflow(self, workflow: WorkflowSpec | Callable) -> None:
        """Bind a workflow to this experiment.

        Accepts a compiled :class:`WorkflowSpec` or a bare ``fn(RunContext)``
        callable; callables are auto-promoted to a single-Task ``WorkflowSpec``.

        Raises:
            TypeError: If *workflow* is not a ``WorkflowSpec`` or callable.
            ValueError: If a workflow is already bound.
        """
        if self._workflow is not None:
            raise ValueError(
                f"Experiment {self.name!r} already has a workflow bound."
            )
        if isinstance(workflow, WorkflowSpec):
            self._workflow = workflow
        elif callable(workflow):
            self._workflow = _promote_to_workflow(workflow, self.name)
        else:
            raise TypeError(
                f"Expected WorkflowSpec or callable, got {type(workflow).__name__}"
            )

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

    def save(self) -> None:
        """Persist current metadata to disk."""
        _save_metadata(self.metadata, self.experiment_dir / "experiment.json")

    # ── Run operations ──────────────────────────────────────────────────

    def run(
        self,
        parameters: dict[str, Any] | None = None,
        *,
        id: str | None = None,
    ) -> Run:
        """Get-or-create a run (idempotent, materialized immediately).

        If a run with the same ID already exists on disk, load and return it.
        Otherwise, construct + materialize a new Run.
        """
        snapshot = None
        if self.metadata.workflow_source:
            snapshot = WorkflowSnapshotRef(
                source=self.metadata.workflow_source,
                git_commit=self.metadata.git_commit,
            )

        r = Run(
            experiment=self,
            parameters=parameters,
            id=id,
            workflow_snapshot=snapshot,
        )
        run_dir = self.experiment_dir / "runs" / f"run-{r.id}"
        if run_dir.exists():
            return self._load_run_from_dir(run_dir)
        r.materialize()
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

    # ── Internal ────────────────────────────────────────────────────────

    def _load_run_from_dir(self, run_dir: Path) -> Run:
        meta = _load_metadata(RunMetadata, run_dir / "run.json")
        return _reconstruct(Run, {"experiment": self, "metadata": meta})
