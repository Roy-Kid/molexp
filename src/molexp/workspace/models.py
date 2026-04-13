"""Domain models for workspace entities.

All metadata models are frozen (immutable). To update, use
``model_copy(update={...})`` and persist the new instance.

This module is the single source of truth for workspace entity schemas.
The server, CLI, and Python API all derive from these models.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ── Execution configuration ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ExecutionConfig:
    """Immutable execution configuration set once before a workflow starts.

    Must be supplied at ``RunContext`` construction time.
    Late-binding (setting after construction) is not permitted.

    Attributes:
        dry_run: When ``True`` the execution is in dry-run mode.
            Tasks can inspect ``ctx.dry_run`` to skip side-effects.
    """

    dry_run: bool = False

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
    """Repeatable experiment definition bound to a workflow."""

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


class RunMetadata(BaseModel, frozen=True):
    """Single execution instance and its frozen snapshot."""

    id: str
    status: str = "pending"
    parameters: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    finished_at: datetime | None = None
    error: ErrorInfo | None = None
    workflow_snapshot: WorkflowSnapshotRef | None = None
    dry_run: bool = False
    labels: dict[str, str] = Field(default_factory=dict)
    executor_info: dict[str, Any] = Field(default_factory=dict)
