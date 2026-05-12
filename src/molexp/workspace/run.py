"""Run entity and RunContext execution lifecycle.

A **Run** represents a single execution instance within an experiment.
**RunContext** is the context manager that handles lifecycle, artifacts,
checkpoints, and asset access during execution.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp._typing import (
    ChannelMessage,
    HashablePayload,
    JSONValue,
    TaskOutput,
)

if TYPE_CHECKING:
    from .workspace import Workspace


class _WorkflowLike(Protocol):
    """Duck-typed shape of ``molexp.workflow.Workflow``.

    Defined here (rather than imported) because the workspace layer must
    not depend on the workflow layer (CLAUDE.md § *Workspace core-dependency
    boundary*). The workflow layer's real ``Workflow`` structurally
    satisfies this Protocol.
    """

    workflow_id: str
    version: str

    def register(self, workspace: Workspace) -> None: ...


if TYPE_CHECKING:
    from .experiment import Experiment

from molexp.config import ProfileConfig

from .assets import (
    ArtifactAccessor,
    AssetCatalog,
    AssetManifest,
    AssetScope,
    CheckpointAccessor,
    ErrorTraceAsset,
    ImportAction,
    LogAccessor,
    Producer,
)
from .assets.manifest import MANIFEST_FILENAME  # noqa: F401 (imported for side effects in tests)
from .base import (
    _load_metadata,
    _rebuild_container_index,
    _reconstruct,
    _save_metadata,
)
from .context import Context
from .errors import RunExistsError, RunNotFoundError
from .folder import WORKSPACE_RUN_KIND, Folder
from .metrics import MetricsWriter
from .models import (
    ErrorInfo,
    ExecutionMetadata,
    ExecutionRecord,
    FolderMetadata,
    RunMetadata,
)
from .utils import generate_asset_id, generate_id

logger = get_logger(__name__)


_FINGERPRINT_HASH_HEX_LEN = 16


class RunFingerprint(BaseModel):
    """Content-addressed identifier for a :class:`Run`.

    Computed from the four inputs that fully determine a run's
    behavior: the compiled workflow spec id, the parameters, the
    upstream inputs, and the environment. Two runs with identical
    fingerprints are reproductions of the same experiment by
    definition; a fingerprint mismatch is grounds to invalidate any
    cached result.

    The existing UUID :attr:`Run.id` continues to identify a row in
    the workspace. The fingerprint is exposed alongside via
    :attr:`Run.fingerprint`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_spec_id: str
    parameters_hash: str
    inputs_hash: str
    environment_hash: str

    @property
    def fingerprint_id(self) -> str:
        canonical = json.dumps(
            {
                "workflow_spec_id": self.workflow_spec_id,
                "parameters_hash": self.parameters_hash,
                "inputs_hash": self.inputs_hash,
                "environment_hash": self.environment_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_FINGERPRINT_HASH_HEX_LEN]


def _hash_payload(payload: HashablePayload) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_FINGERPRINT_HASH_HEX_LEN]


def _environment_signature() -> str:
    """Deterministic, dependency-free hash of the active runtime environment."""
    return _hash_payload(
        {
            "python": sys.version.split()[0],
            "platform": platform.platform(terse=True),
        }
    )


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ── RunContext ──────────────────────────────────────────────────────────────


class RunContext:
    """Primary execution context.

    Entered via ``with run.start() as ctx:`` — manages lifecycle,
    result binding, checkpointing, artifact storage, and asset access.

    The active :class:`~molexp.config.ProfileConfig` is fixed at
    construction time via ``profile_config``.  Late-binding after
    construction is not permitted.
    """

    def __init__(
        self,
        run: Run,
        *,
        profile_config: ProfileConfig | None = None,
        execution_id: str | None = None,
    ) -> None:
        self.run = run
        self.work_dir = run.run_dir
        self._profile_config = (
            profile_config if profile_config is not None else ProfileConfig({}, name=None)
        )
        self._entered = False
        self._context: Context = Context(
            run_id=run.id,
            experiment_id=run.experiment.id,
            project_id=run.experiment.project.id,
            work_dir=self.work_dir,
        )
        self._start_time: datetime | None = None
        # ``execution_id`` is normally derived inside ``__enter__``; an
        # explicit override is used by molq submission to pre-allocate
        # the slot so stdout/stderr/jobs land under the same directory
        # the worker will use.
        self._explicit_execution_id: str | None = execution_id
        self._execution_id: str | None = None
        # Actor message-passing infrastructure — channel name → asyncio.Queue.
        # Queues hold user-domain messages (typed ``ChannelMessage`` because
        # the workspace doesn't constrain the wire format).
        self._channels: dict[str, asyncio.Queue[ChannelMessage]] = {}
        # Active task id (set via set_active_task) used for Producer.task_id
        self._active_task_id: str | None = None
        # Walltime chunking — set by ``suspend()`` so ``__exit__`` keeps the
        # run resumable instead of marking it SUCCEEDED.
        self._suspended: bool = False
        self._suspended_at_step: int | None = None

        # Asset plumbing
        self._scope = AssetScope(
            kind="run",
            ids=(
                run.experiment.project.id,
                run.experiment.id,
                run.id,
            ),
        )
        self._manifest = AssetManifest(self.work_dir)
        workspace_root = run.experiment.project.workspace.root
        self._catalog: AssetCatalog = AssetCatalog(workspace_root)

        self.artifact = ArtifactAccessor(
            self.work_dir,
            self._scope,
            self._manifest,
            self._catalog,
            self._producer,
        )
        self.log = LogAccessor(
            self.work_dir,
            self._scope,
            self._manifest,
            self._catalog,
            self._producer,
            self._get_execution_id,
        )
        self.checkpoint = CheckpointAccessor(
            self.work_dir,
            self._scope,
            self._manifest,
            self._catalog,
            self._producer,
        )
        self.metrics = MetricsWriter(self.work_dir)

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

    # ── Lifecycle ───────────────────────────────────────────────────────

    def __enter__(self) -> RunContext:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing_results()
        self._apply_profile_metadata()
        self._claim_ownership()
        self.run._set_status(RunStatus.RUNNING)
        self._start_time = datetime.now()
        self._entered = True

        # Determine which execution attempt this is and record it.
        self._execution_id = self._explicit_execution_id or self._next_execution_id()
        new_record = ExecutionRecord(
            execution_id=self._execution_id,
            started_at=self._start_time,
        )
        self.run._update_metadata(
            execution_history=[*self.run.metadata.execution_history, new_record]
        )
        self._write_execution_metadata(
            ExecutionMetadata(
                execution_id=self._execution_id,
                run_id=self.run.id,
                started_at=self._start_time,
                status=RunStatus.RUNNING.value,
            )
        )
        self._refresh_executions_index()
        self._append_run_log(f"execution started  exec_id={self._execution_id}")

        self._save_context()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        now = datetime.now()
        labels = self._labels_without_ownership()
        error_info: ErrorInfo | None = None
        if exc_type is None and self._suspended:
            # Walltime chunking — keep the run resume-eligible (PENDING) so
            # the next ``with run.start()`` picks up at ``last_step``. The
            # current attempt is still recorded as a finished execution
            # entry so the execution_history reflects every chunk.
            final = RunStatus.PENDING
            self.run._update_metadata(
                status=final,
                labels=labels,
                execution_history=self._close_execution_record("suspended", now),
            )
        elif exc_type is None:
            workflow_status = self._context.status.get("run")
            final = RunStatus.FAILED if workflow_status == RunStatus.FAILED else RunStatus.SUCCEEDED
            self.run._update_metadata(
                status=final,
                finished_at=now,
                labels=labels,
                execution_history=self._close_execution_record(final.value, now),
            )
        else:
            final = RunStatus.FAILED
            error_info = ErrorInfo(
                type=exc_type.__name__,
                message=str(exc_val),
                timestamp=now,
            )
            self.run._update_metadata(
                status=final,
                finished_at=now,
                labels=labels,
                error=error_info,
                execution_history=self._close_execution_record(final.value, now),
            )
            self._save_error_details(exc_type, exc_val, exc_tb)
        self._update_execution_metadata(
            finished_at=now,
            status=final.value,
            error=error_info,
        )
        self._refresh_executions_index()
        self.run.experiment._refresh_runs_index()
        self._append_run_log(
            f"execution finished exec_id={self._execution_id}  status={final.value}"
        )
        self._save_context()
        self._entered = False
        return False

    # ── Async-context-manager protocol ──────────────────────────────────
    #
    # Thin wrappers over ``__enter__`` / ``__exit__``. The body is sync
    # (filesystem writes, metadata maintenance) so no real awaits happen
    # — but exposing ``__aenter__`` / ``__aexit__`` lets callers write
    # ``async with run.start() as ctx:`` directly inside an async
    # workflow body without juggling ``asyncio.to_thread`` themselves.

    async def __aenter__(self) -> RunContext:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
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
        """Resolve a data directory path.

        Searches the asset hierarchy first. If no asset is found and
        *fallback* is given, creates ``workspace_root / fallback`` and
        returns it.  All return values are :class:`~pathlib.Path`.

        Args:
            asset_name: Name of the asset to look up.
            fallback: Relative path under workspace root to create when the
                asset is not found.

        Returns:
            Resolved data directory path.

        Raises:
            FileNotFoundError: If no asset found and no fallback specified.
        """
        asset = self.find_asset(asset_name)
        if asset is not None:
            return Path(asset.path)
        if fallback is not None:
            fallback = Path(fallback)
            data_dir = self.run.experiment.project.workspace.root / fallback
            data_dir.mkdir(parents=True, exist_ok=True)
            return data_dir
        raise FileNotFoundError(f"Asset {asset_name!r} not found and no fallback specified.")

    def set_result(self, key: str, value: TaskOutput) -> None:
        self._context.results[key] = value

    def get_result(self, key: str) -> TaskOutput:
        return self._context.results.get(key)

    # ── Walltime chunking ──────────────────────────────────────────────

    @property
    def resumed_step(self) -> int:
        """Step number to resume iteration from.

        Reads ``RunMetadata.last_step`` recorded by previous executions of
        this run (chunks); ``0`` for a fresh run.  Used together with
        :meth:`checkpoint_step` and :meth:`suspend` to drive walltime
        chunking under a single ``run.json``.
        """
        return self.run.metadata.last_step or 0

    def checkpoint_step(self, step: int, *, data: dict[str, JSONValue] | None = None) -> None:
        """Record the latest completed step on the run metadata.

        ``step`` is the *next* step to start at — i.e. after completing
        steps 0..N-1, callers pass ``N``.  Optional ``data`` is forwarded
        into the run's checkpoint JSON for diagnostic snapshotting; it
        does not influence resumption.
        """
        self.run._update_metadata(last_step=int(step))
        # Persist results/context so set_result data isn't lost on suspend
        self._save_context()
        if data is not None:
            self.checkpoint(name=f"step_{step}", data=data)

    def suspend(self, *, at_step: int | None = None) -> None:
        """Mark the run for resumption rather than completion.

        Subsequent ``__exit__`` (without an exception) leaves the run in
        :class:`RunStatus.PENDING` instead of :class:`RunStatus.SUCCEEDED`.
        The caller is expected to ``return`` after this call so the
        ``with`` block exits normally.
        """
        self._suspended = True
        if at_step is not None:
            self._suspended_at_step = int(at_step)
            self.run._update_metadata(last_step=int(at_step))

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
        self._save_context()

    def set_workflow(self, workflow: BaseModel | dict) -> None:
        if isinstance(workflow, BaseModel):
            self._context.workflow = workflow.model_dump()
        elif isinstance(workflow, dict):
            self._context.workflow = workflow
        else:
            raise TypeError("workflow must be a Pydantic BaseModel or dict")

    # ── Asset access ────────────────────────────────────────────────────

    def register_asset(
        self,
        name: str,
        src: Path | str,
        action: ImportAction = "copy",
        meta: dict | None = None,
    ):
        """Import a ``DataAsset`` into this run's experiment scope."""
        return self.run.experiment.data_assets.import_asset(
            name=name, src=src, action=action, meta=meta or {}
        )

    def get_asset(self, name: str, scope: str = "project"):
        if scope == "experiment":
            return self.run.experiment.data_assets.get(name)
        if scope == "project":
            return self.run.experiment.project.data_assets.get(name)
        if scope == "workspace":
            return self.run.experiment.project.workspace.data_assets.get(name)
        raise ValueError(f"Unknown scope: {scope!r}")

    def find_asset(self, name: str):
        for scope in ("experiment", "project", "workspace"):
            asset = self.get_asset(name, scope=scope)
            if asset is not None:
                return asset
        return None

    # ── Actor message passing ───────────────────────────────────────────

    async def receive(self, channel: str) -> ChannelMessage:
        if channel not in self._channels:
            raise KeyError(f"Channel '{channel}' not found")
        return await self._channels[channel].get()

    async def emit(self, channel: str, message: ChannelMessage) -> None:
        if channel not in self._channels:
            logger.warning(f"emit() to non-existent channel {channel!r} — dropped")
            return
        await self._channels[channel].put(message)

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

    def _register_channel(self, name: str, queue: asyncio.Queue[ChannelMessage]) -> None:
        self._channels[name] = queue

    # ── Internal ────────────────────────────────────────────────────────

    @property
    def context(self) -> Context:
        return self._context

    def _load_existing_results(self) -> None:
        from .schema_version import read_versioned_json

        run_json = self.work_dir / "run.json"
        if not run_json.exists() or run_json.stat().st_size == 0:
            return
        data = read_versioned_json(run_json)
        for key, value in data.get("context", {}).get("results", {}).items():
            if key not in self._context.results:
                self._context.results[key] = value

    def _save_context(self) -> None:
        from .schema_version import write_versioned_json

        write_versioned_json(
            self.work_dir / "run.json",
            {
                **self.run.metadata.model_dump(mode="json"),
                "context": self._context.model_dump(mode="json"),
            },
        )

    def _execution_metadata_path(self) -> Path:
        assert self._execution_id is not None
        return self.work_dir / "executions" / self._execution_id / "execution.json"

    def _write_execution_metadata(self, meta: ExecutionMetadata) -> None:
        from .schema_version import write_versioned_json

        target = self._execution_metadata_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        write_versioned_json(target, meta.model_dump(mode="json"))

    def _update_execution_metadata(self, **updates: object) -> None:
        """Merge *updates* into the on-disk execution.json (read-modify-write).

        Values flow through pydantic's per-field validators on
        :class:`ExecutionMetadata`; the parameter type is the structural
        top-type ``object`` because the values are forwarded as-is
        without inspection.
        """
        from .schema_version import read_versioned_json, write_versioned_json

        target = self._execution_metadata_path()
        if not target.exists():
            return
        current = ExecutionMetadata(**read_versioned_json(target))
        merged = current.model_copy(update=updates)
        write_versioned_json(target, merged.model_dump(mode="json"))

    def _refresh_executions_index(self) -> None:
        _rebuild_container_index(
            container_dir=self.work_dir / "executions",
            index_filename="executions.json",
            metadata_filename="execution.json",
            fields=[
                "execution_id",
                "run_id",
                "status",
                "started_at",
                "finished_at",
                "scheduler_job_id",
            ],
        )

    def _next_execution_id(self) -> str:
        """Return the execution_id for this attempt.

        Mirrors the logic in the workflow runtime so that the directory
        created by RunStorePersistence and the execution_history entry
        share the same identifier.
        """
        run_id = self.run.id
        base = f"exec-{run_id}"
        exec_root = self.work_dir / "executions"
        if not exec_root.exists():
            return base
        existing = [p for p in exec_root.iterdir() if p.name.startswith(base)]
        if not existing:
            return base
        return f"{base}-{len(existing) + 1}"

    def _close_execution_record(self, status: str, finished_at: datetime) -> list[ExecutionRecord]:
        """Return execution_history with the current record closed."""
        history = list(self.run.metadata.execution_history)
        for i, entry in enumerate(history):
            if entry.execution_id == self._execution_id:
                history[i] = entry.model_copy(update={"finished_at": finished_at, "status": status})
                return history
        return history

    def _append_run_log(self, message: str) -> None:
        """Append a single timestamped line to the ``run`` LogAsset."""
        ts = datetime.now().isoformat(timespec="seconds")
        self.log("run").append(f"{ts}  {message}")

    def _apply_profile_metadata(self) -> None:
        """Persist the active profile name / data / hash into RunMetadata."""
        cfg = self._profile_config
        self.run._update_metadata(
            profile=cfg.name,
            config=cfg.to_dict(),
            config_hash=cfg.content_hash() if len(cfg) > 0 or cfg.name else None,
            labels=dict(self.run.metadata.labels),
        )

    def _claim_ownership(self) -> None:
        """Stamp the run with the current process identity.

        Stored in ``labels`` as ``pid`` / ``host`` / ``heartbeat``.  A later
        ``molexp run`` invocation can consult these to tell a live run from a
        zombie left behind by a crashed process.
        """
        labels = dict(self.run.metadata.labels)
        labels["pid"] = str(os.getpid())
        labels["host"] = platform.node()
        labels["heartbeat"] = datetime.now().isoformat()
        self.run._update_metadata(labels=labels)

    def _labels_without_ownership(self) -> dict[str, str]:
        """Return labels with the ownership stamp removed."""
        labels = dict(self.run.metadata.labels)
        for key in ("pid", "host", "heartbeat"):
            labels.pop(key, None)
        return labels

    def _save_error_details(self, exc_type, exc_val, exc_tb) -> None:
        """Persist an ``ErrorTraceAsset`` for the current execution."""
        tb_lines = traceback.format_exception(exc_type, exc_val, exc_tb)
        exec_id = self._execution_id or "unbound"
        rel_path = Path("executions") / exec_id / "error.txt"
        target = self.work_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            f"Error: {datetime.now().isoformat()}\n"
            f"Type: {exc_type.__name__}\n"
            f"Message: {exc_val}\n\n" + "".join(tb_lines)
        )

        now = datetime.now()
        asset = ErrorTraceAsset(
            asset_id=generate_asset_id(),
            name=f"error_{exec_id}",
            scope=self._scope,
            path=rel_path,
            created_at=now,
            updated_at=now,
            producer=self._producer(),
            exception_type=exc_type.__name__,
            message=str(exc_val),
            execution_id=exec_id,
        )
        self._manifest.register(asset)
        self._catalog.register(asset)


# ── Run ─────────────────────────────────────────────────────────────────────


class Run(Folder):
    """Single execution instance within an experiment.

    Inherits :class:`Folder` (sub-spec 02): ``kind`` is
    :data:`WORKSPACE_RUN_KIND`, ``parent`` is the owning
    :class:`Experiment`. The on-disk directory uses the ``run-<id>``
    prefix preserved from the pre-refactor layout — see
    :meth:`_child_dir`.

    Example::

        run = experiment.Run(parameters={"lr": 0.001})
        with run.start() as ctx:
            result = my_workflow(ctx)
            ctx.set_result("output", result)
    """

    _exists_error_cls = RunExistsError
    _not_found_error_cls = RunNotFoundError

    def __init__(
        self,
        *,
        parent: Experiment | None = None,
        name: str | None = None,
        kind: str = WORKSPACE_RUN_KIND,
        experiment: Experiment | None = None,
        parameters: dict[str, JSONValue] | None = None,
        id: str | None = None,
        workflow_snapshot: dict[str, JSONValue] | None = None,
        target: str | None = None,
        _entity_metadata: RunMetadata | None = None,
    ) -> None:
        resolved_parent = parent if parent is not None else experiment
        if resolved_parent is None:
            raise ValueError("Run: parent (or experiment) is required")
        # ``name`` (Folder convention) is the Run's id — Run has no
        # human-readable name distinct from its slug.
        meta = (
            _entity_metadata
            if _entity_metadata is not None
            else RunMetadata(
                id=id or name or generate_id(),
                parameters=parameters or {},
                workflow_snapshot=workflow_snapshot,
                target=target,
            )
        )

        self._parent = resolved_parent
        self._name = meta.id
        self._kind = kind
        self._root_path = None
        self._metadata = FolderMetadata(
            id=meta.id,
            name=meta.id,  # Run has no separate display name
            kind=kind,
            created_at=meta.created_at,
            updated_at=meta.created_at,
        )
        self._children_cache = {}

        # Entity-specific state
        self._entity_metadata: RunMetadata = meta

    # ── Folder hooks ─────────────────────────────────────────────────────

    def _compute_path(self) -> Path:
        return self.run_dir

    @classmethod
    def _child_dir(cls, parent: Folder, derived_id: str) -> Path:
        """:class:`Folder.attach` hook — runs live under ``runs/run-<id>/``.

        The ``run-`` prefix is preserved from the pre-refactor layout
        to keep legacy workspaces loadable without migration.
        """
        return parent.path() / "runs" / f"run-{derived_id}"

    @classmethod
    def _from_disk(cls, child_dir: Path, parent: Folder) -> Run:
        """:class:`Folder.attach` hook — load ``run.json`` and rebuild entity state."""
        meta = _load_metadata(RunMetadata, child_dir / "run.json")
        return _reconstruct(
            cls,
            {
                "_parent": parent,
                "_name": meta.id,
                "_kind": WORKSPACE_RUN_KIND,
                "_root_path": None,
                "_metadata": FolderMetadata(
                    id=meta.id,
                    name=meta.id,
                    kind=WORKSPACE_RUN_KIND,
                    created_at=meta.created_at,
                    updated_at=meta.created_at,
                ),
                "_children_cache": {},
                "_entity_metadata": meta,
            },
        )

    def children(self, kind: str | None = None) -> list[Folder]:
        """Run has no entity children — executions live under ``executions/``
        but are not Folder-tracked (sub-spec 03 may revisit)."""
        return []

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def experiment(self) -> Experiment:
        """The owning :class:`Experiment` (alias for :attr:`Folder.parent`)."""
        if self._parent is None:  # pragma: no cover — Run always has a parent
            raise RuntimeError("Run has no parent experiment")
        return cast("Experiment", self._parent)

    @property
    def metadata(self) -> RunMetadata:  # type: ignore[override]
        return self._entity_metadata

    @metadata.setter
    def metadata(self, value: RunMetadata) -> None:
        self._entity_metadata = value

    @property
    def id(self) -> str:
        return self._entity_metadata.id

    @property
    def parameters(self) -> dict[str, JSONValue]:
        return self._entity_metadata.parameters

    @property
    def fingerprint(self) -> RunFingerprint:
        """Content-addressed :class:`RunFingerprint` for this run.

        Independent of the UUID :attr:`id`; two runs with identical
        ``(workflow_spec_id, parameters, inputs_hash, environment_hash)``
        share the same fingerprint id.
        """
        snapshot = self.metadata.workflow_snapshot
        # ``workflow_snapshot`` is an opaque JSON dict (see RunMetadata);
        # we only sip a single field for fingerprint composition.
        workflow_spec_id = ""
        if isinstance(snapshot, dict):
            value = snapshot.get("workflow_id")
            if isinstance(value, str):
                workflow_spec_id = value
        return RunFingerprint(
            workflow_spec_id=workflow_spec_id,
            parameters_hash=_hash_payload(self.metadata.parameters),
            inputs_hash=_hash_payload({}),  # populated when input-asset model lands
            environment_hash=_environment_signature(),
        )

    @property
    def status(self) -> str:
        return self.metadata.status

    @property
    def run_dir(self) -> Path:
        return self.experiment.experiment_dir / "runs" / f"run-{self.id}"

    @property
    def scope(self):
        from .assets import AssetScope

        return AssetScope(
            kind="run",
            ids=(self.experiment.project.id, self.experiment.id, self.id),
        )

    @property
    def assets(self):
        """Scope-filtered catalog view (read-only queries) for this run."""
        from .assets import AssetsView

        return AssetsView(self.experiment.project.workspace.catalog, self.scope)

    def get_result(self, key: str) -> TaskOutput:
        """Read a result value persisted by ``RunContext.set_result``.

        Returns ``None`` when the run has not been executed yet, when the
        key is absent, or when ``run.json`` does not exist on disk.
        """
        from .schema_version import read_versioned_json

        run_json = self.run_dir / "run.json"
        if not run_json.exists() or run_json.stat().st_size == 0:
            return None
        try:
            data = read_versioned_json(run_json)
        except (OSError, ValueError):
            return None
        return data.get("context", {}).get("results", {}).get(key)

    # ── Persistence ─────────────────────────────────────────────────────

    def materialize(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _save_metadata(self.metadata, self.run_dir / "run.json")
        self._catalog_upsert()

    def save(self) -> None:
        _save_metadata(self.metadata, self.run_dir / "run.json")
        self._catalog_upsert()

    def _catalog_upsert(self) -> None:
        ws = self.experiment.project.workspace
        record = {
            "run_id": self.metadata.id,
            "experiment_id": self.experiment.id,
            "status": self.metadata.status,
            "parameters": dict(self.metadata.parameters),
            "profile": self.metadata.profile,
            "config_hash": self.metadata.config_hash,
            "labels": dict(self.metadata.labels),
            "path": str(self.run_dir.relative_to(ws.root)),
            "created_at": self.metadata.created_at.isoformat(),
            "finished_at": (
                self.metadata.finished_at.isoformat() if self.metadata.finished_at else None
            ),
            "workflow_snapshot": (
                dict(self.metadata.workflow_snapshot) if self.metadata.workflow_snapshot else None
            ),
        }
        ws.catalog.upsert_run(record)
        # Upsert every execution record in history
        for rec in self.metadata.execution_history:
            ws.catalog.upsert_execution(
                {
                    "execution_id": rec.execution_id,
                    "run_id": self.metadata.id,
                    "status": rec.status,
                    "started_at": rec.started_at.isoformat(),
                    "finished_at": (rec.finished_at.isoformat() if rec.finished_at else None),
                    "scheduler_job_id": rec.scheduler_job_id,
                }
            )

    # ── Execution ───────────────────────────────────────────────────────

    def start(
        self,
        profile_config: ProfileConfig | None = None,
        *,
        execution_id: str | None = None,
    ) -> RunContext:
        """Return a context manager for executing this run.

        *profile_config* selects the active molcfg profile; when omitted
        the run executes with an empty (defaults-only) :class:`ProfileConfig`.

        *execution_id* pre-allocates the execution slot — used by external
        submitters (e.g. molq) that need to know the per-attempt directory
        ahead of worker startup.

        The returned :class:`RunContext` supports both ``with`` and
        ``async with`` — choose whichever matches the caller's body.
        For the no-arg case, ``Run`` itself is also a context manager
        (sugar that calls ``self.start()`` internally); see
        :meth:`__enter__` / :meth:`__aenter__`.
        """
        return RunContext(self, profile_config=profile_config, execution_id=execution_id)

    # ── Sugar: ``with run as ctx:`` / ``async with run as ctx:`` ────────
    #
    # Equivalent to ``with run.start() as ctx:`` / ``async with``. Sugar
    # form does not accept ``profile_config`` / ``execution_id``; for
    # those, call ``run.start(...)`` explicitly. Internally we cache the
    # ``RunContext`` on first ``__enter__`` so ``__exit__`` sees the
    # same instance.

    def __enter__(self) -> RunContext:
        self._sugar_ctx = self.start()
        return self._sugar_ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        ctx = self._sugar_ctx
        del self._sugar_ctx
        return ctx.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> RunContext:
        self._sugar_ctx = self.start()
        return await self._sugar_ctx.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        ctx = self._sugar_ctx
        del self._sugar_ctx
        return await ctx.__aexit__(exc_type, exc_val, exc_tb)

    def cancel(self) -> None:
        """Mark the run as cancelled in workspace metadata."""
        labels = dict(self.metadata.labels)
        for key in ("pid", "host", "heartbeat"):
            labels.pop(key, None)
        self._update_metadata(
            status=RunStatus.CANCELLED,
            finished_at=datetime.now(),
            labels=labels,
        )

    def delete_execution(self, execution_id: str) -> None:
        """Delete a single execution attempt from this run.

        Removes ``executions/<execution_id>/`` on disk, pops the matching
        entry from ``execution_history``, and drops the catalog row.  The
        run itself is left intact.

        Raises:
            KeyError: If the execution id is not present under this run.
        """
        import shutil

        exec_dir = self.run_dir / "executions" / execution_id
        history = list(self.metadata.execution_history)
        matched_idx = next(
            (i for i, rec in enumerate(history) if rec.execution_id == execution_id),
            None,
        )
        if matched_idx is None and not exec_dir.exists():
            raise KeyError(f"Execution '{execution_id}' not found under run '{self.id}'")
        if exec_dir.exists():
            shutil.rmtree(exec_dir)
        if matched_idx is not None:
            history.pop(matched_idx)
            self._update_metadata(execution_history=history)
        self.experiment.project.workspace.catalog.remove_execution(execution_id)
        _rebuild_container_index(
            container_dir=self.run_dir / "executions",
            index_filename="executions.json",
            metadata_filename="execution.json",
            fields=[
                "execution_id",
                "run_id",
                "status",
                "started_at",
                "finished_at",
                "scheduler_job_id",
            ],
        )

    # ── Internal (frozen-metadata mutation helpers) ──────────────────────

    def _set_status(self, status: RunStatus) -> None:
        self.metadata = self.metadata.model_copy(update={"status": status.value})
        self.save()

    def _update_metadata(self, **updates: object) -> None:
        """Forward partial-field updates into ``RunMetadata.model_copy``.

        Values flow through pydantic's per-field validators; the parameter
        type is the true Python top-type ``object`` (not ``Any`` — the
        function does not interact with the values, it only forwards
        them, and pydantic owns the per-field type contract).
        """
        self.metadata = self.metadata.model_copy(update=updates)
        self.save()
