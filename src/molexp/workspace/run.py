"""Run entity and RunContext execution lifecycle.

A **Run** represents a single execution instance within an experiment.
**RunContext** is the context manager that handles lifecycle, artifacts,
checkpoints, and asset access during execution.
"""

from __future__ import annotations

import json
import os
import platform
import time
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mollog import get_logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from .experiment import Experiment

from molexp.config import ProfileConfig
from molexp.plugins.metrics import MetricsWriter

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
from .base import _atomic_write_json, _load_metadata, _reconstruct, _save_metadata
from .context import Context
from .models import ErrorInfo, ExecutionRecord, RunMetadata, WorkflowSnapshotRef
from .utils import generate_asset_id, generate_id

logger = get_logger(__name__)


class RunStatus(str, Enum):
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
    ) -> None:
        self.run = run
        self.work_dir = run.run_dir
        self._profile_config = (
            profile_config if profile_config is not None else ProfileConfig({}, name=None)
        )
        self._entered = False
        self._context = Context(
            run_id=run.id,
            experiment_id=run.experiment.id,
            project_id=run.experiment.project.id,
            work_dir=self.work_dir,
        )
        self._start_time: datetime | None = None
        self._execution_id: str | None = None
        # Actor message-passing infrastructure
        self._channels: dict[str, Any] = {}
        # Active task id (set via set_active_task) used for Producer.task_id
        self._active_task_id: str | None = None

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
        self._execution_id = self._next_execution_id()
        new_record = ExecutionRecord(
            execution_id=self._execution_id,
            started_at=self._start_time,
        )
        self.run._update_metadata(
            execution_history=[*self.run.metadata.execution_history, new_record]
        )
        self._append_run_log(f"execution started  exec_id={self._execution_id}")

        self._save_context()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        now = datetime.now()
        labels = self._labels_without_ownership()
        if exc_type is None:
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
            self.run._update_metadata(
                status=final,
                finished_at=now,
                labels=labels,
                error=ErrorInfo(
                    type=exc_type.__name__,
                    message=str(exc_val),
                    timestamp=now,
                ),
                execution_history=self._close_execution_record(final.value, now),
            )
            self._save_error_details(exc_type, exc_val, exc_tb)
        self._append_run_log(
            f"execution finished exec_id={self._execution_id}  status={final.value}"
        )
        self._save_context()
        self._entered = False
        return False

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def params(self) -> dict[str, Any]:
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

    def set_result(self, key: str, value: Any) -> None:
        self._context.results[key] = value

    def get_result(self, key: str) -> Any:
        return self._context.results.get(key)

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
        elif scope == "project":
            return self.run.experiment.project.data_assets.get(name)
        elif scope == "workspace":
            return self.run.experiment.project.workspace.data_assets.get(name)
        raise ValueError(f"Unknown scope: {scope!r}")

    def find_asset(self, name: str):
        for scope in ("experiment", "project", "workspace"):
            asset = self.get_asset(name, scope=scope)
            if asset is not None:
                return asset
        return None

    # ── Actor message passing ───────────────────────────────────────────

    async def receive(self, channel: str) -> Any:
        if channel not in self._channels:
            raise KeyError(f"Channel '{channel}' not found")
        return await self._channels[channel].get()

    async def emit(self, channel: str, message: Any) -> None:
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

    def _register_channel(self, name: str, queue: Any) -> None:
        self._channels[name] = queue

    # ── Internal ────────────────────────────────────────────────────────

    @property
    def context(self) -> Context:
        return self._context

    def _load_existing_results(self):
        run_json = self.work_dir / "run.json"
        if not run_json.exists() or run_json.stat().st_size == 0:
            return
        with open(run_json) as f:
            data = json.load(f)
        for key, value in data.get("context", {}).get("results", {}).items():
            if key not in self._context.results:
                self._context.results[key] = value

    def _save_context(self):
        _atomic_write_json(
            self.work_dir / "run.json",
            {
                **self.run.metadata.model_dump(mode="json"),
                "context": self._context.model_dump(mode="json"),
            },
        )

    def _next_execution_id(self) -> str:
        """Return the execution_id for this attempt.

        Mirrors the logic in the workflow runtime so that the directory
        created by RunStorePersistence and the execution_history entry
        share the same identifier.
        """
        run_id = self.run.id
        base = f"exec-{run_id}"
        exec_root = self.work_dir / "execution"
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
        updates: dict[str, Any] = {
            "profile": cfg.name,
            "config": cfg.to_dict(),
            "config_hash": cfg.content_hash() if len(cfg) > 0 or cfg.name else None,
            "labels": dict(self.run.metadata.labels),
        }
        self.run._update_metadata(**updates)

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

    def _save_error_details(self, exc_type, exc_val, exc_tb):
        """Persist an ``ErrorTraceAsset`` for the current execution."""
        tb_lines = traceback.format_exception(exc_type, exc_val, exc_tb)
        exec_id = self._execution_id or "unbound"
        rel_path = Path("execution") / exec_id / "error.txt"
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


class Run:
    """Single execution instance within an experiment.

    Example::

        run = experiment.run(parameters={"lr": 0.001})
        with run.start() as ctx:
            result = my_workflow(ctx)
            ctx.set_result("output", result)
    """

    def __init__(
        self,
        experiment: Experiment,
        parameters: dict[str, Any] | None = None,
        id: str | None = None,
        workflow_snapshot: WorkflowSnapshotRef | None = None,
    ) -> None:
        self.experiment = experiment
        self.metadata = RunMetadata(
            id=id or generate_id(),
            parameters=parameters or {},
            workflow_snapshot=workflow_snapshot,
        )

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def id(self) -> str:
        return self.metadata.id

    @property
    def parameters(self) -> dict[str, Any]:
        return self.metadata.parameters

    @property
    def status(self) -> str:
        return self.metadata.status

    @property
    def run_dir(self) -> Path:
        return self.experiment.experiment_dir / "runs" / f"run-{self.id}"

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
                self.metadata.workflow_snapshot.model_dump(mode="json")
                if self.metadata.workflow_snapshot
                else None
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

    def start(self, profile_config: ProfileConfig | None = None) -> RunContext:
        """Return a context manager for executing this run.

        *profile_config* selects the active molcfg profile; when omitted
        the run executes with an empty (defaults-only) :class:`ProfileConfig`.
        """
        return RunContext(self, profile_config=profile_config)

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

    # ── Internal (frozen-metadata mutation helpers) ──────────────────────

    def _set_status(self, status: RunStatus) -> None:
        self.metadata = self.metadata.model_copy(update={"status": status.value})
        self.save()

    def _update_metadata(self, **updates: Any) -> None:
        self.metadata = self.metadata.model_copy(update=updates)
        self.save()
