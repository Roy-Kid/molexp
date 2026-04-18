"""Domain models for workspace entities.

This module is the single source of truth for workspace entity schemas.
The server, CLI, and Python API all derive from these models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ── Shared value objects ────────────────────────────────────────────────────


class ErrorInfo(BaseModel, frozen=True):
    """Structured error summary attached to a failed run."""

    type: str
    message: str
    timestamp: datetime


class WorkflowSnapshotRef(BaseModel, frozen=True):
    """Frozen reference to the workflow that produced a run.

    Captured at run-creation time so the exact code/config
    can always be traced back.
    """

    source: str
    git_commit: str | None = None
    code_hash: str | None = None
    config_hash: str | None = None


# ── Entity metadata ────────────────────────────────────────────────────────
#
# Design rules:
#   1. Every model is frozen.  Updates go through model_copy(update={...}).
#   2. Parents never store child lists.  ``list_projects()`` scans the
#      filesystem; directory presence is the only truth.
#   3. No ``updated_at``.  If you need "last modified", read the file mtime.


class WorkspaceMetadata(BaseModel, frozen=True):
    """Top-level workspace."""

    id: str
    name: str
    created_at: datetime = Field(default_factory=datetime.now)


class ProjectMetadata(BaseModel, frozen=True):
    """Research project container."""

    id: str
    name: str
    description: str = ""
    owner: str = ""
    tags: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class ExperimentMetadata(BaseModel, frozen=True):
    """Repeatable experiment definition bound to a workflow.

    An Experiment carries a concrete parameter dict (`parameter_space`)
    plus replica configuration (`n_replicas`, `seeds`).  Parameter sweeps
    are expanded by the user at script level (e.g. via ``for p in GridSpace(...)``);
    each combination becomes a distinct Experiment.  Replicas under a
    single Experiment share parameters but differ in random seed.
    """

    id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    # Workflow binding — what makes an Experiment more than just a folder
    workflow_source: str | None = None
    workflow_type: str | None = None
    parameter_space: dict[str, Any] = Field(default_factory=dict)
    git_commit: str | None = None

    # Replica configuration
    n_replicas: int = 1
    seeds: list[int] | None = None


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
    molq_job_id: str | None = None
    scheduler_job_id: str | None = None


class RunMetadata(BaseModel):
    """Single execution instance metadata.

    ``profile`` is the activated molcfg profile name (normalized,
    ``-`` → ``_``) or ``None`` when no profile was selected.  ``config``
    is the frozen merged configuration dict that the run executed
    against; ``config_hash`` is its sha256 digest, duplicated for fast
    queryability.  The framework treats profile contents as opaque
    user data — it persists them for reproducibility but never
    interprets them.

    ``execution_history`` indexes every attempt to execute this run,
    newest last.  Each entry points to the corresponding sub-directory
    under ``run_dir/execution/``.
    """

    id: str
    status: str = "pending"
    parameters: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    finished_at: datetime | None = None
    error: ErrorInfo | None = None
    workflow_snapshot: WorkflowSnapshotRef | None = None
    script: str | None = None
    profile: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    config_hash: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    executor_info: dict[str, Any] = Field(default_factory=dict)
    execution_history: list[ExecutionRecord] = Field(default_factory=list)
