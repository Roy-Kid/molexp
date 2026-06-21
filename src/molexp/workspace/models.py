"""Domain models for workspace entities.

This module is the single source of truth for workspace entity schemas.
The server, CLI, and Python API all derive from these models.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from molexp._typing import JSONValue

# ── Shared value objects ────────────────────────────────────────────────────


class RunStatus(StrEnum):
    """Lifecycle state of a run.

    Lives here (the workspace schema source of truth) rather than in
    ``run.py`` so the run-lifecycle collaborators can import it without a
    circular ``run.py`` dependency. Re-exported from ``run.py`` for
    backward compatibility (``from molexp.workspace.run import RunStatus``).
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorInfo(BaseModel, frozen=True):
    """Structured error summary attached to a failed run."""

    type: str
    message: str
    timestamp: datetime


# ``WorkflowSnapshotRef`` lives under ``molexp.workflow.snapshot_ref`` —
# workspace stores its on-disk shape as opaque JSON in
# ``RunMetadata.workflow_snapshot``. The relocation was part of the
# rectification spec (2026-05-09); see CLAUDE.md § Layer charters.

# ── Entity metadata ────────────────────────────────────────────────────────
#
# Design rules:
#   1. Every model is frozen.  Updates go through model_copy(update={...}).
#   2. Parents never store child lists.  ``list_projects()`` scans the
#      filesystem; directory presence is the only truth.
#   3. No ``updated_at``.  If you need "last modified", read the file mtime.
#      EXCEPTION: ``FolderMetadata`` (introduced by the
#      unify-folder-abstraction chain) carries an explicit ``updated_at``
#      because filesystem ``mtime`` is unreliable across rsync / git
#      checkout / cross-host copy and can't anchor the global folder
#      index. The deviation is scoped to the new ``FolderMetadata`` model;
#      legacy entity metadata above keeps the mtime-based design.


class FolderMetadata(BaseModel, frozen=True):
    """Lifecycle metadata for the unified ``Folder`` abstraction.

    Carries the minimum every folder needs regardless of business
    semantics: stable id (slugified ``name``), human-readable ``name``,
    dotted-ASCII ``kind`` for child filtering, and monotonic
    ``created_at``/``updated_at`` timestamps. The ``extra`` slot lets
    business subclasses (``SessionFolder`` / ``CacheFolder`` / ``PlanFolder``)
    stash custom fields without forking the schema.

    The ``updated_at`` field is a deliberate deviation from rule 3 above
    — see the comment block preceding this class for the rationale.
    """

    id: str
    name: str
    kind: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    extra: dict[str, JSONValue] = Field(default_factory=dict)


class ComputeTarget(BaseModel, frozen=True):
    """A registered execution destination — the cross product of two axes.

    The two-axis cluster model (Transport × Scheduler) treats *where* commands
    run as orthogonal to *how* jobs are dispatched.  ``host`` is the transport
    axis: ``None`` means run locally (``LocalTransport``); a non-empty value
    routes through SSH to that host.  ``scheduler`` is the dispatch axis: one
    of the molq scheduler names.

    Examples::

        # Today's `--local` path, just named.
        ComputeTarget(name="laptop", scratch_root="/tmp/molexp")

        # Remote SLURM cluster — the canonical HPC use case.
        ComputeTarget(
            name="hpc1",
            host="me@cluster.example.org",
            scheduler="slurm",
            scratch_root="/scratch/me/molexp",
        )

        # Run on a remote workstation directly, no batch system.
        ComputeTarget(
            name="desk", host="me@desk.lan", scheduler="local", scratch_root="/home/me/molexp-runs"
        )
    """  # noqa: RUF002

    name: str

    # ── Transport axis (where commands run) ─────────────────────────────────
    host: str | None = None  # None → LocalTransport, else SshTransport
    port: int | None = None
    identity_file: str | None = None
    ssh_opts: list[str] = Field(default_factory=list)

    # ── Scheduler axis (how jobs are dispatched) ────────────────────────────
    scheduler: Literal["local", "slurm", "pbs", "lsf"] = "local"

    # ── Working dir + defaults ──────────────────────────────────────────────
    scratch_root: str
    default_resources: dict[str, JSONValue] = Field(default_factory=dict)
    default_scheduling: dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_axes(self) -> ComputeTarget:
        if self.host is None and (self.port is not None or self.identity_file or self.ssh_opts):
            raise ValueError(
                "transport options (port, identity_file, ssh_opts) require host to be set"
            )
        if not self.scratch_root:
            raise ValueError("scratch_root is required")
        return self

    @property
    def is_remote(self) -> bool:
        return self.host is not None


class WorkspaceMetadata(BaseModel, frozen=True):
    """Top-level workspace."""

    id: str
    name: str
    created_at: datetime = Field(default_factory=datetime.now)
    targets: list[ComputeTarget] = Field(default_factory=list)


class ProjectMetadata(BaseModel, frozen=True):
    """Research project container."""

    id: str
    name: str
    description: str = ""
    owner: str = ""
    tags: list[str] = Field(default_factory=list)
    config: dict[str, JSONValue] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class ExperimentMetadata(BaseModel, frozen=True):
    """Repeatable experiment definition — a parameter-space container.

    An Experiment carries a concrete parameter dict (``parameter_space``)
    plus replica configuration (``n_replicas``, ``seeds``). Parameter
    combinations are expanded by the user at script level (e.g. via
    ``for p in GridSpace(...)``); each combination becomes a distinct
    Experiment. Replicas under a single Experiment share parameters but
    differ in random seed.

    Workspace does **not** know about workflows; the Experiment-to-
    Workflow pairing is the caller's concern (typically the agent layer
    or a user script). The ``workflow_source`` / ``workflow_type`` fields
    here are advisory free-form strings used by the UI for grouping —
    workspace itself never interprets them.

    ``model_config`` ignores extra fields so workspace.json files written
    by older molexp versions (which may carry now-removed keys like
    ``workflow``) still load cleanly.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    # Advisory free-form workflow metadata — used by the UI for
    # grouping; workspace never interprets it.
    workflow_source: str | None = None
    workflow_type: str | None = None
    parameter_space: dict[str, JSONValue] = Field(default_factory=dict)
    git_commit: str | None = None

    # Replica configuration
    n_replicas: int = 1
    seeds: list[int] | None = None

    # Default compute target for runs created under this experiment.
    # Validated against ``WorkspaceMetadata.targets`` at write time.
    default_target: str | None = None


class ExecutionRecord(BaseModel, frozen=True):
    """One attempt to execute a Run.

    A Run may be executed multiple times (e.g. retries after failure).
    Each attempt is recorded here so the full execution history is
    queryable from run.json without scanning execution sub-directories.
    """

    execution_id: str
    started_at: datetime
    finished_at: datetime | None = None
    status: str = "running"
    scheduler_job_id: str | None = None


class ExecutionMetadata(BaseModel, frozen=True):
    """Per-attempt metadata persisted to ``executions/<exec_id>/execution.json``.

    Mirrors the matching :class:`ExecutionRecord` entry in
    ``run.json.execution_history`` so a single ``executions/<exec_id>/``
    directory is self-describing without consulting the parent run.
    Fields beyond the record are populated from the active
    :class:`RunContext` (executor info, error summary).
    """

    execution_id: str
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    status: str = "running"
    scheduler_job_id: str | None = None
    executor_info: dict[str, JSONValue] = Field(default_factory=dict)
    error: ErrorInfo | None = None


class RunMetadata(BaseModel, frozen=True):
    """Single execution instance — identity / provenance only.

    ``RunMetadata`` (persisted to ``run.json``) carries the settled,
    human-meaningful identity of a run: its parameters, frozen config +
    hash, activated profile, workflow snapshot/version, source snapshot,
    intended compute target, executor info, and the terminal ``error``
    diagnostic. The framework treats profile/config contents as opaque
    user data — it persists them for reproducibility but never interprets
    them.

    Hot machine state — ``status`` / ``finished_at`` / execution history /
    ownership (pid/host/heartbeat) — does **not** live here. It is owned
    by the OKF ``_ops/run.json`` sidecar (:class:`RunOpsState`), the sole
    source of truth (the OKF identity-vs-runtime split; wsokf-07/wsokf-10).
    Read it through the ops-backed :class:`~molexp.workspace.run.Run`
    accessors (``run.status`` / ``run.finished_at`` / ``run.execution_history``)
    or :meth:`~molexp.workspace.run.Run.read_ops`.

    ``model_config`` uses ``extra="ignore"`` so a ``run.json`` written by
    an older molexp version (which still carried ``status`` / ``finished_at`` /
    ``execution_history`` / ``labels``) loads cleanly — the relocated keys
    are dropped on read (greenfield; no backfill migration).

    ``error`` remains here as a terminal failure diagnostic — it is not
    modelled by :class:`RunOpsState`.

    ``submit_cwd`` is the absolute working directory at the moment
    ``molexp run`` submitted this run.  The cluster worker chdirs here
    before importing the user's workflow file so cwd-relative paths in
    module-level code (e.g. ``Workspace("./lab")``) resolve to the same
    location they did at submit time, preventing nested duplicate
    workspaces under ``run_dir/``.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    parameters: dict[str, JSONValue] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    error: ErrorInfo | None = None
    # Opaque workflow-snapshot payload — the canonical type lives in
    # ``molexp.workflow.snapshot_ref.WorkflowSnapshotRef``; workspace
    # stores it as a plain dict to avoid an upward dependency.
    workflow_snapshot: dict[str, JSONValue] | None = None
    script: str | None = None
    # Source snapshot — the entrypoint script's bytes + its first-party
    # local-import closure, copied under ``run_dir/source/`` at dispatch (see
    # ``molexp.workspace.source_snapshot.snapshot_sources``). ``script`` records
    # the *path* and ``workflow_snapshot.code_hash`` an AST hash; this carries the
    # actual source manifest so a run is reproducible from its own directory even
    # if the live tree changes. ``None`` for runs dispatched before this existed.
    source_snapshot: dict[str, JSONValue] | None = None
    submit_cwd: str | None = None
    profile: str | None = None
    config: dict[str, JSONValue] = Field(default_factory=dict)
    config_hash: str | None = None
    executor_info: dict[str, JSONValue] = Field(default_factory=dict)

    # Intended compute target name (matches a ComputeTarget in the workspace
    # registry).  Captured at run-creation time so the UI can filter and the
    # actual submitter can pick the right SubmitHandler later.  Distinct from
    # ``executor_info.cluster_name`` which is populated post-submit by molq.
    target: str | None = None

    # Workflow versioning — populated by RunContext.bind_workflow_version().
    # ``workflow_id`` is the deterministic topology hash; ``workflow_version``
    # is the user-declared label.  Both are ``None`` when the run was started
    # without a bound Workflow (legacy / ad-hoc runs).
    workflow_id: str | None = None
    workflow_version: str | None = None
