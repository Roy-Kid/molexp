"""Workflow versioning — immutable on-disk version records.

A :class:`WorkflowVersion` is the user-facing label for a particular
workflow topology. The ``workflow_id`` (topology hash, computed by
``_stable_workflow_id`` in :mod:`molexp.workflow.spec`) is the cache and
execution identity; ``version`` is a free-form human label (typically a
semver string like ``"1.2.0"``).

Records are stored at::

    <workspace_root>/.versions/workflows/<workflow_id>.json

and are immutable: once a ``(workflow_id, version)`` pair has been
written, attempting to register the same ``workflow_id`` with a
different ``version`` raises :class:`WorkflowVersionConflictError`.

Same-pair re-registration is a no-op (the file is left untouched, mtime
unchanged) so that idle re-runs are cheap and safe.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from molexp.workspace import Workspace

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

    Persisted to ``<workspace>/.versions/workflows/<workflow_id>.json``
    via :meth:`molexp.workflow.spec.WorkflowSpec.register`.

    Attributes:
        schema_version: On-disk format version for this record (current: 1).
        workflow_id: Deterministic topology hash from
            :func:`molexp.workflow.spec._stable_workflow_id`.
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


def _versions_dir(workspace: Workspace) -> Path:
    return workspace.root / VERSIONS_DIRNAME / WORKFLOWS_SUBDIR


def _record_path(workspace: Workspace, workflow_id: str) -> Path:
    return _versions_dir(workspace) / f"{workflow_id}.json"


def write_record(workspace: Workspace, record: WorkflowVersion) -> Path:
    """Persist a :class:`WorkflowVersion` to disk.

    Conflict semantics:

    * If the file does not exist → write atomically and return its path.
    * If the file exists with the same ``(workflow_id, version)`` →
      no-op; mtime is preserved.
    * If the file exists with a different ``version`` → raise
      :class:`WorkflowVersionConflictError`.

    Args:
        workspace: Target :class:`~molexp.workspace.Workspace`.
        record: The version record to persist.

    Returns:
        Absolute path of the version file.

    Raises:
        WorkflowVersionConflictError: When ``workflow_id`` is already
            registered with a different ``version``.
    """
    from molexp.workspace.base import _atomic_write_json

    path = _record_path(workspace, record.workflow_id)
    if path.exists():
        with open(path) as fh:
            existing = WorkflowVersion(**json.load(fh))
        if existing.version == record.version:
            return path
        raise WorkflowVersionConflictError(
            f"workflow_id={record.workflow_id!r} is already registered as "
            f"version={existing.version!r}; cannot re-register as "
            f"version={record.version!r}. Either keep the version label or "
            f"change the workflow topology."
        )
    _atomic_write_json(path, record.model_dump(mode="json"))
    return path


def load_record(workspace: Workspace, workflow_id: str) -> WorkflowVersion | None:
    """Return the persisted :class:`WorkflowVersion` for ``workflow_id``.

    Args:
        workspace: Workspace whose ``.versions/workflows/`` directory to
            consult.
        workflow_id: Topology hash to look up.

    Returns:
        The parsed record, or ``None`` if the file does not exist.
    """
    path = _record_path(workspace, workflow_id)
    if not path.exists():
        return None
    with open(path) as fh:
        return WorkflowVersion(**json.load(fh))
