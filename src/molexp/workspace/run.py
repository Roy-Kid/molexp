"""Run entity and RunContext execution lifecycle.

A **Run** represents a single execution instance within an experiment.
**RunContext** is the context manager that handles lifecycle, artifacts,
checkpoints, and asset access during execution.
"""

from __future__ import annotations

import json
import time
from mollog import get_logger
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from .experiment import Experiment

from .base import _atomic_write_json, _load_metadata, _reconstruct, _save_metadata
from .context import Context
from .models import ErrorInfo, ExecutionConfig, RunMetadata, WorkflowSnapshotRef
from .utils import generate_id

logger = get_logger(__name__)


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DRY_RUN = "dry_run"


# ── RunContext ──────────────────────────────────────────────────────────────


class RunContext:
    """Primary execution context.

    Entered via ``with run.start() as ctx:`` — manages lifecycle,
    result binding, checkpointing, artifact storage, and asset access.

    Execution mode (e.g. ``dry_run``) is fixed at construction time via
    ``execution_config``.  Late-binding after construction is not permitted.
    """

    def __init__(
        self,
        run: Run,
        *,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        self.run = run
        self.work_dir = run.run_dir
        self.artifacts_dir = self.work_dir / "artifacts"
        self.logs_dir = self.work_dir / "logs"
        self._execution_config = execution_config if execution_config is not None else ExecutionConfig()
        self._entered = False
        self._context = Context(
            run_id=run.id,
            experiment_id=run.experiment.id,
            project_id=run.experiment.project.id,
            work_dir=self.work_dir,
            artifacts_dir=self.artifacts_dir,
            logs_dir=self.logs_dir,
        )
        self._start_time: datetime | None = None
        # Actor message-passing infrastructure
        self._channels: dict[str, Any] = {}

    # ── Lifecycle ───────────────────────────────────────────────────────

    def __enter__(self) -> RunContext:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self._load_existing_results()
        self._apply_execution_mode_metadata()
        self.run._set_status(RunStatus.RUNNING)
        self._start_time = datetime.now()
        self._entered = True
        self._save_context()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        now = datetime.now()
        if exc_type is None:
            workflow_status = self._context.status.get("run")
            if workflow_status == RunStatus.FAILED:
                final = RunStatus.FAILED
            elif self.dry_run:
                final = RunStatus.DRY_RUN
            else:
                final = RunStatus.SUCCEEDED
            self.run._update_metadata(status=final, finished_at=now)
        else:
            self.run._update_metadata(
                status=RunStatus.FAILED,
                finished_at=now,
                error=ErrorInfo(
                    type=exc_type.__name__,
                    message=str(exc_val),
                    timestamp=now,
                ),
            )
            self._save_error_details(exc_type, exc_val, exc_tb)
        self._save_context()
        self._entered = False
        return False

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def params(self) -> dict[str, Any]:
        """Shortcut for ``self.run.parameters``."""
        return self.run.parameters

    @property
    def dry_run(self) -> bool:
        """Whether this execution is running in dry-run mode."""
        return self._execution_config.dry_run

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
        raise FileNotFoundError(
            f"Asset {asset_name!r} not found and no fallback specified."
        )

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

    def checkpoint(self, name: str | None = None) -> str:
        from .checkpoint import generate_checkpoint_id

        ckpt_id = generate_checkpoint_id()
        ckpt_dir = self.work_dir / ".ckpt"
        ckpt_dir.mkdir(exist_ok=True)
        _atomic_write_json(ckpt_dir / f"{ckpt_id}.json", {
            "ckpt_id": ckpt_id,
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "context": self._context.model_dump(mode="json"),
        })
        return ckpt_id

    def save_artifact(self, name: str, data: Any) -> Path:
        artifact_path = self.artifacts_dir / name
        if isinstance(data, dict):
            with open(artifact_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif isinstance(data, (bytes, bytearray)):
            with open(artifact_path, "wb") as f:
                f.write(data)
        elif isinstance(data, Path):
            import shutil
            shutil.copy2(data, artifact_path)
        else:
            with open(artifact_path, "w") as f:
                f.write(str(data))
        return artifact_path

    def get_artifact_path(self, name: str) -> Path:
        return self.artifacts_dir / name

    # ── Asset access ────────────────────────────────────────────────────

    def register_asset(
        self,
        name: str,
        src: Path | str,
        action: str = "copy",
        meta: dict | None = None,
    ):
        return self.run.experiment.assets.import_asset(
            name=name, src=src, action=action, meta=meta or {}
        )

    def get_asset(self, name: str, scope: str = "project"):
        if scope == "experiment":
            return self.run.experiment.assets.get_asset(name)
        elif scope == "project":
            return self.run.experiment.project.assets.get_asset(name)
        elif scope == "workspace":
            return self.run.experiment.project.workspace.assets.get_asset(name)
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
            raise FileNotFoundError(
                f"Run metadata not found at {run_json}"
            ) from last_exc
        run = _reconstruct(Run, {"experiment": experiment, "metadata": meta})
        exec_config = ExecutionConfig(dry_run=run.metadata.dry_run)
        return cls(run, execution_config=exec_config)

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
        _atomic_write_json(self.work_dir / "run.json", {
            **self.run.metadata.model_dump(mode="json"),
            "context": self._context.model_dump(mode="json"),
        })

    def _apply_execution_mode_metadata(self) -> None:
        labels = dict(self.run.metadata.labels)
        if self._execution_config.dry_run:
            labels["mode"] = "dry-run"
            updates: dict[str, Any] = {"dry_run": True, "labels": labels}
        else:
            if labels.get("mode") == "dry-run":
                labels.pop("mode")
            updates = {"dry_run": False, "labels": labels}
        self.run._update_metadata(**updates)

    def _save_error_details(self, exc_type, exc_val, exc_tb):
        tb_lines = traceback.format_exception(exc_type, exc_val, exc_tb)
        error_txt = self.logs_dir / "error.txt"
        with open(error_txt, "w") as f:
            f.write(f"Error: {datetime.now().isoformat()}\n")
            f.write(f"Type: {exc_type.__name__}\n")
            f.write(f"Message: {exc_val}\n\n")
            f.write("".join(tb_lines))


# ── Run ─────────────────────────────────────────────────────────────────────


class Run:
    """Single execution instance within an experiment.

    Example::

        run = experiment.create_run(parameters={"lr": 0.001})
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

    def save(self) -> None:
        _save_metadata(self.metadata, self.run_dir / "run.json")

    # ── Execution ───────────────────────────────────────────────────────

    def start(self) -> RunContext:
        """Return a context manager for normal-mode (non-dry-run) execution."""
        return RunContext(self, execution_config=ExecutionConfig(dry_run=False))

    def update_job_ids(
        self,
        *,
        slurm_job_id: str | None = None,
        molq_job_id: str | None = None,
    ) -> None:
        """Persist scheduler job IDs into run.json for cross-reference.

        Enables ``grep -r '"slurm_job_id": "..."' runs/*/run.json`` to locate
        a run from a SLURM job ID shown in ``squeue`` / ``sacct``.
        """
        updates: dict[str, Any] = {}
        if slurm_job_id is not None:
            updates["slurm_job_id"] = slurm_job_id
        if molq_job_id is not None:
            updates["molq_job_id"] = molq_job_id
        if updates:
            self._update_metadata(**updates)

    def cancel(self) -> None:
        """Mark this run as cancelled in the workspace.

        Updates ``run.json`` with ``status = "cancelled"``.  This is the
        workspace-side half of cancellation; call
        ``molq.Submitor(...).cancel(self.metadata.molq_job_id)`` first if
        the run was submitted to a scheduler.
        """
        self._set_status(RunStatus.CANCELLED)

    # ── Internal ─────────────────────────────────────────────────────────

    def _set_status(self, status: RunStatus) -> None:
        self.metadata = self.metadata.model_copy(update={"status": status.value})
        self.save()

    def _update_metadata(self, **updates: Any) -> None:
        self.metadata = self.metadata.model_copy(update=updates)
        self.save()
