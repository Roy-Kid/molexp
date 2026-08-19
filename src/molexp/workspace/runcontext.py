"""``RunContext`` — the run-execution context manager.

Entered via ``with run.start() as ctx:``; orchestrates the lifecycle, result
binding, checkpointing, artifact/log/metric I/O, and asset access for one run
attempt. The work is delegated to four collaborators (``RunLifecycle`` /
``ContextStore`` / ``ExecutionStore`` / ``RunAssets``); this facade wires them
together and re-exposes the typed accessor handles. Split out of ``run.py``
(where the :class:`Run` entity lives): ``Run.start`` constructs a
``RunContext`` and ``RunContext.open`` lazily reconstructs a ``Run`` via a
function-local import, so there is no module-level import cycle.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from molexp._typing import JSONValue, TaskOutput
from molexp.profile import ProfileConfig

from .assets import ArtifactAsset, Asset, AssetScope, Producer
from .assets.base import AssetKind
from .base import _load_metadata, _reconstruct
from .context import Context
from .metrics import MetricRecord
from .models import RunMetadata, RunStatus
from .run_assets import RunAssets
from .run_context import ContextStore
from .run_execution import ExecutionStore
from .run_lifecycle import RunLifecycle

if TYPE_CHECKING:
    from typing import Protocol

    from .run import Run
    from .workspace import Workspace

    class _WorkflowLike(Protocol):
        """Duck-typed shape of ``molexp.workflow.Workflow``.

        Declared here (rather than imported) because the workspace layer must
        not depend on the workflow layer (CLAUDE.md § *Workspace core-dependency
        boundary*). The workflow layer's real ``Workflow`` structurally
        satisfies this Protocol. Annotation-only, so it lives under
        ``TYPE_CHECKING`` and the runtime imports above stay E402-free.
        """

        workflow_id: str
        version: str

        def register(self, workspace: Workspace) -> None: ...


class RunContext:
    """Primary execution context.

    Entered via ``with run.start() as ctx:`` — manages lifecycle,
    result binding, checkpointing, artifact storage, and asset access.

    The active :class:`~molexp.profile.ProfileConfig` is fixed at
    construction time via ``profile_config``.  Late-binding after
    construction is not permitted.

    ``run_dir`` is the run's on-disk root as a :class:`~pathlib.Path`
    (same path as :attr:`Run.run_dir`). ``workdir`` is the execution-scoped
    scratch directory where the driver writes files
    (``<run_dir>/executions/<exec_id>/work``).
    """

    run: Run
    run_dir: Path

    def __init__(
        self,
        run: Run,
        *,
        profile_config: ProfileConfig | None = None,
        execution_id: str | None = None,
    ) -> None:
        self.run = run
        self.run_dir = Path(run.run_dir)
        self._profile_config = (
            profile_config if profile_config is not None else ProfileConfig({}, name=None)
        )
        self._entered = False
        self._start_time: datetime | None = None
        # ``execution_id`` is normally derived inside ``__enter__``; an
        # explicit override is used by molq submission to pre-allocate
        # the slot so stdout/stderr/jobs land under the same directory
        # the worker will use.
        self._explicit_execution_id: str | None = execution_id
        self._execution_id: str | None = None
        # Active task id (set via set_active_task) used for Producer.task_id
        self._active_task_id: str | None = None

        # ── Layered collaborators (workspace-slim-03) ───────────────────
        # The facade keeps the transient lifecycle state above; each
        # collaborator owns one cohesive responsibility. ``RunLifecycle``
        # orchestrates the others on enter/exit via a back-reference.
        scope = AssetScope(
            kind="run",
            ids=(run.experiment.project.id, run.experiment.id, run.id),
        )
        self._ctx_store = ContextStore(run, self.run_dir)
        self._executions = ExecutionStore(run, self.run_dir)
        self._assets = RunAssets(run, self.run_dir, scope, self._producer, self._get_execution_id)
        self._lifecycle = RunLifecycle(self)

        # Stream / chain accessors stay on the facade. File products go
        # through :meth:`register_artifact` — there is no ``ctx.artifact``.
        self.files = self._assets.files
        self.log = self._assets.log
        self.checkpoint = self._assets.checkpoint
        self.metrics = self._assets.metrics

    def _producer(self) -> Producer:
        return Producer(
            run_id=self.run.id,
            execution_id=self._execution_id,
            task_id=self._active_task_id,
        )

    def _get_execution_id(self) -> str | None:
        return self._execution_id

    def set_active_task(self, task_id: str | None) -> None:
        """Set the active task id so future accessor writes set ``Producer.task_id``."""
        self._active_task_id = task_id

    # ── Working directories ─────────────────────────────────────────────

    @property
    def workdir(self) -> Path:
        """Execution-scoped scratch directory for this attempt.

        ``<run_dir>/executions/<execution_id>/work``, created on access.
        Write files here, then :meth:`register_artifact` to publish them.

        Raises:
            RuntimeError: if no execution is active.
        """
        return self._assets.workdir

    def task_workdir(self, task_name: str) -> Path:
        """Scratch directory for one workflow task under this execution."""
        return self._assets.task_workdir(task_name)

    # ── Lifecycle ───────────────────────────────────────────────────────

    def __enter__(self) -> RunContext:
        self._lifecycle.enter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:  # noqa: ANN001
        try:
            return self._lifecycle.exit(exc_type, exc_val, exc_tb)
        finally:
            # Flush deferred metadata once, after the lifecycle has appended its
            # final run-log line. Per-op writes are deferred to avoid
            # O(lines x assets) manifest churn (logs) and O(records) index
            # rewrites per metric (metrics).
            self._assets.log.flush_all()
            self._assets.metrics.flush()

    # ── Async-context-manager protocol ──────────────────────────────────
    #
    # Thin wrappers over ``__enter__`` / ``__exit__``. The body is sync
    # (filesystem writes, metadata maintenance) so no real awaits happen
    # — but exposing ``__aenter__`` / ``__aexit__`` lets callers write
    # ``async with run.start() as ctx:`` directly inside an async
    # workflow body without juggling ``asyncio.to_thread`` themselves.

    async def __aenter__(self) -> RunContext:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:  # noqa: ANN001
        return self.__exit__(exc_type, exc_val, exc_tb)

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def params(self) -> dict[str, JSONValue]:
        """Shortcut for ``self.run.parameters``."""
        return self.run.parameters

    @property
    def config(self) -> ProfileConfig:
        """Active profile configuration (read-only mapping)."""
        return self._profile_config

    def get_data_dir(
        self,
        asset_name: str,
        *,
        fallback: str | Path | None = None,
    ) -> Path:
        """Resolve a data directory path (delegates to :class:`RunAssets`)."""
        return self._assets.get_data_dir(asset_name, fallback=fallback)

    def register_artifact(
        self,
        data: TaskOutput,
        *,
        name: str | None = None,
        mime: str | None = None,
        tags: dict[str, str] | None = None,
        consumed: Sequence[Asset | str] | None = None,
    ) -> ArtifactAsset:
        """Publish ``data`` as a run artifact under ``<run_dir>/artifacts/``.

        Args:
            data: A :class:`~pathlib.Path` (copied), bytes, JSON-serializable
                ``dict``/``list``, or any other value (str-cast).
            name: Filename under ``artifacts/``. Defaults to ``data.name``
                when *data* is a Path; required otherwise.
            mime: Optional MIME type hint.
            tags: Free-form metadata attached to the asset.
            consumed: Optional upstream assets (or asset-id strings) recorded
                as :attr:`Producer.inputs` lineage.

        Returns:
            The registered :class:`~molexp.workspace.assets.ArtifactAsset`.
        """
        return self._assets.register_artifact(
            data, name=name, mime=mime, tags=tags, consumed=consumed
        )

    def register(
        self,
        path: Path,
        *,
        kind: AssetKind = "artifact",
        name: str | None = None,
        mime: str | None = None,
        tags: dict[str, str] | None = None,
        consumed: Sequence[Asset | str] | None = None,
        **extra: object,
    ) -> Asset:
        """Catalog an existing file under this run. Does not rewrite the payload."""
        return self._assets.register(
            path, kind=kind, name=name, mime=mime, tags=tags, consumed=consumed, **extra
        )

    def register_metric(
        self,
        key: str,
        value: float,
        *,
        step: int | None = None,
        tags: dict | None = None,
    ) -> MetricRecord:
        """Append a scalar metric to this run's metrics WAL."""
        return self.metrics.scalar(key, value, step, tags=tags)

    def set_result(self, key: str, value: TaskOutput) -> None:
        self._ctx_store.set_result(key, value)

    def get_result(self, key: str) -> TaskOutput:
        return self._ctx_store.get_result(key)

    def bind_workflow_version(self, spec: _WorkflowLike) -> None:
        """Pin this run to a versioned :class:`~molexp.workflow.Workflow`.

        Persists ``workflow_id`` + ``workflow_version`` into
        ``RunMetadata`` and registers the spec's
        :class:`~molexp.workflow.WorkflowVersion` record under the
        workspace's ``.versions/workflows/`` directory. Idempotent on
        identical re-binds; raises
        :class:`~molexp.workflow.WorkflowVersionConflictError` when the
        same ``workflow_id`` is already labelled with a different
        version.

        Args:
            spec: The :class:`~molexp.workflow.Workflow` produced
                by ``Workflow.build()``.
        """
        ws = self.run.experiment.project.workspace
        spec.register(ws)
        self.run._update_metadata(
            workflow_id=spec.workflow_id,
            workflow_version=spec.version,
        )
        self._ctx_store.save()

    def set_workflow(self, workflow: BaseModel | dict) -> None:
        self._ctx_store.set_workflow(workflow)

    # ── Asset access ────────────────────────────────────────────────────

    def get_asset(self, name: str, scope: str = "project"):  # noqa: ANN201
        return self._assets.get_asset(name, scope)

    def find_asset(self, name: str):  # noqa: ANN201
        return self._assets.find_asset(name)

    # ── Constructor helpers ─────────────────────────────────────────────

    @classmethod
    def open(cls, run_dir: Path) -> RunContext:
        """Reconstruct a RunContext from an existing run directory.

        Traverses the directory hierarchy to rebuild the full
        Workspace → Project → Experiment → Run chain.

        Args:
            run_dir: Path to the run directory (contains ``run.json``).

        Returns:
            ``RunContext`` wrapping the reconstructed run.

        Raises:
            FileNotFoundError: If workspace, project, experiment or run metadata is missing.
        """
        from .run import Run  # local: run.py imports RunContext, so avoid a module cycle
        from .workspace import Workspace

        run_dir = Path(run_dir).resolve()
        # Layout: workspace_root/projects/proj_id/experiments/exp_id/runs/run-{id}
        workspace_root = run_dir.parents[5]
        project_id = run_dir.parents[3].name
        experiment_id = run_dir.parents[1].name

        workspace = Workspace.load(workspace_root)
        project = workspace.get_project(project_id)
        if project is None:
            raise FileNotFoundError(
                f"Project '{project_id}' not found in workspace at {workspace_root}"
            )
        experiment = project.get_experiment(experiment_id)
        if experiment is None:
            raise FileNotFoundError(
                f"Experiment '{experiment_id}' not found in project '{project_id}'"
            )
        # Load run metadata directly from run_dir/run.json.
        # Avoid experiment.get_run() which calls run_dir.exists() — on Lustre/NFS
        # filesystems that call can return False for newly-created directories due
        # to metadata cache staleness on compute nodes, even when the directory and
        # file are physically present.
        run_json = run_dir / "run.json"
        meta = None
        last_exc: Exception | None = None
        for _attempt in range(3):
            try:
                meta = _load_metadata(RunMetadata, run_json)
                break
            except FileNotFoundError as exc:
                last_exc = exc
                time.sleep(1)

        if meta is None:
            raise FileNotFoundError(f"Run metadata not found at {run_json}") from last_exc
        run = _reconstruct(Run, {"experiment": experiment, "metadata": meta})
        profile_cfg = ProfileConfig(run.metadata.config, name=run.metadata.profile)
        return cls(run, profile_config=profile_cfg)

    def mark_failed(self, error: str | None = None, traceback_text: str | None = None) -> None:
        """Record a run failure so an exception-free ``with ctx:`` exit still
        resolves the run-status to ``FAILED``.

        The workflow runtime calls this (through the ``RunContextLike``
        protocol) when a task body fails but the failure does not propagate as
        an exception out of ``execute()`` — the engine swallows the task
        exception and signals here instead. The lifecycle's ``exit`` consults
        ``context.status["run"]`` to pick ``SUCCEEDED`` vs ``FAILED``.

        ``traceback_text`` (optional) carries the formatted Python traceback of
        the underlying task failure; the lifecycle persists it into
        ``executions/<exec_id>/error.txt`` so the trace file holds the real
        stack, not a placeholder note. Like ``error``, only the FIRST recorded
        failure wins (``setdefault``).
        """
        context = self._ctx_store.context
        context.status["run"] = RunStatus.FAILED
        if error:
            record = {"message": error}
            if traceback_text:
                record["traceback"] = traceback_text
            context.errors.setdefault("run", record)

    def mark_succeeded(self) -> None:
        """Record a positive completion signal for this attempt.

        The workflow runtime calls this after a workflow finishes with no
        failed task, so the lifecycle's ``exit`` can distinguish "the workflow
        ran and succeeded" from "nothing ran at all". A recorded failure
        always wins — ``mark_succeeded`` never downgrades a prior
        :meth:`mark_failed` (run-recovery bug 1).
        """
        context = self._ctx_store.context
        if context.status.get("run") != RunStatus.FAILED:
            context.status["run"] = RunStatus.SUCCEEDED

    # ── Internal ────────────────────────────────────────────────────────

    @property
    def context(self) -> Context:
        return self._ctx_store.context
