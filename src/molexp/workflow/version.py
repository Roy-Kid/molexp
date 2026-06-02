"""Workflow versioning — pure data records.

A :class:`WorkflowVersion` is the user-facing label for a particular
workflow topology. The ``workflow_id`` (topology hash, computed by
``_stable_workflow_id`` in :mod:`molexp.workflow._helpers` and carried on
:class:`~molexp.workflow.compiled.CompiledWorkflow`) is the cache and
execution identity; ``version`` is a free-form human label (typically a
semver string like ``"1.2.0"``).

This module owns only the **data shape** — :class:`WorkflowVersion`,
:class:`TaskTopologyEntry`, :class:`WorkflowVersionConflictError`, and the
schema-version constants. Persistence (writing to / reading from a
workspace's ``.versions/workflows/`` directory) is the workspace layer's
responsibility, not the workflow layer's.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

VERSION_SCHEMA_VERSION = 1
VERSIONS_DIRNAME = ".versions"
WORKFLOWS_SUBDIR = "workflows"


class WorkflowVersionConflictError(RuntimeError):
    """Raised when a ``workflow_id`` is re-registered with a different ``version``.

    The same code (identical topology hash) cannot wear two version
    labels — that defeats the point of pinning. Bump the topology
    (rename a task, add/remove a dependency, change the task class) or
    keep the version constant.
    """


class TaskTopologyEntry(BaseModel):
    """One row of a :class:`WorkflowVersion`'s topology snapshot.

    Attributes:
        name: Task name as registered on the workflow.
        qualname: ``__qualname__`` of the task callable / class — same
            input the topology hash uses.
        depends_on: Upstream task names, ordered as registered.
        code_hash: Optional AST-normalized code hash (from
            :class:`~molexp.workflow.snapshot.TaskSnapshot`); ``None``
            for callables that do not support snapshotting.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    qualname: str
    depends_on: tuple[str, ...] = ()
    code_hash: str | None = None


class WorkflowVersion(BaseModel):
    """Immutable record of one ``(workflow_id, version)`` pair.

    Carried as the ``version`` field of
    :class:`~molexp.workflow.compiled.CompiledWorkflow`. The workflow layer
    no longer writes this record to disk; persistence is a workspace-layer
    concern.

    Attributes:
        schema_version: On-disk format version for this record (current: 1).
        workflow_id: Deterministic topology hash from
            :func:`molexp.workflow._helpers._stable_workflow_id`.
        version: User-declared human label, e.g. ``"1.2.0"``.
        name: Workflow display name at registration time.
        topology: Ordered snapshot of every task (name, qualname,
            dependencies, optional code_hash).
        created_at: Record creation timestamp.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: int = VERSION_SCHEMA_VERSION
    workflow_id: str
    version: str
    name: str
    topology: tuple[TaskTopologyEntry, ...]
    created_at: datetime = Field(default_factory=datetime.now)
