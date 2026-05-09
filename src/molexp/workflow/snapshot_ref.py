"""Workflow-snapshot reference type — on-disk shape stored in run.json.

Captured at run-creation time so the exact code/config that produced a
run can always be traced back. Owned by the workflow layer because the
fields (``entrypoint``, ``code_hash``, ``config_hash``) are workflow
concerns; workspace stores it as opaque JSON in
``RunMetadata.workflow_snapshot`` (typed as a generic mapping there)
to avoid an upward dependency.

History: this type used to live in ``molexp.workspace.models``,
which forced workspace to know about workflow concepts. The
rectification spec (2026-05-09) moved it here as part of the
workspace ← workflow ← agent dependency-direction fix.
"""

from __future__ import annotations

from pydantic import BaseModel


class WorkflowSnapshotRef(BaseModel, frozen=True):
    """Frozen reference to the workflow that produced a run.

    ``entrypoint`` is the worker's import coordinate, formatted as
    ``"<absolute_file_path>:<qualname>"``. The worker imports the file
    as a *non-``__main__``* module — any ``if __name__ == "__main__":``
    guard in the user script therefore skips workspace setup, so
    re-importing for workflow lookup has no side effects. ``source``
    is retained for human-readable audit (often the same path).
    """

    source: str
    entrypoint: str | None = None
    git_commit: str | None = None
    code_hash: str | None = None
    config_hash: str | None = None


__all__ = ["WorkflowSnapshotRef"]
