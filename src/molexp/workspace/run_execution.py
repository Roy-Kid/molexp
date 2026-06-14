"""``ExecutionStore`` — RunContext's execution-attempt persistence.

Tier-2 collaborator of :class:`~molexp.workspace.run.RunContext` (see the
``workspace-slim-03-runcontext`` decomposition). Owns the
``executions/<execution_id>/`` subtree (each attempt's ``execution.json``)
and the ``execution_history`` record maintenance. Stateless apart from the
``run`` + ``work_dir`` it is bound to; the active ``execution_id`` is
passed in per call by the lifecycle. Independent of :class:`ContextStore`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import ExecutionMetadata, ExecutionRecord
from .utils import derive_execution_id

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from .run import Run


class ExecutionStore:
    """Owns the ``executions/<id>/`` subtree and execution-history edits."""

    def __init__(self, run: Run, work_dir: Path) -> None:
        self._run = run
        self._work_dir = work_dir

    def metadata_path(self, execution_id: str) -> Path:
        return self._work_dir / "executions" / execution_id / "execution.json"

    def write_metadata(self, meta: ExecutionMetadata) -> None:
        from .schema_version import write_versioned_json

        target = self.metadata_path(meta.execution_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_versioned_json(target, meta.model_dump(mode="json"))

    def update_metadata(self, execution_id: str, **updates: object) -> None:
        """Merge *updates* into the on-disk ``execution.json`` (read-modify-write).

        Values flow through pydantic's per-field validators on
        :class:`ExecutionMetadata`; the parameter type is the structural
        top-type ``object`` because the values are forwarded as-is without
        inspection.
        """
        from .schema_version import read_versioned_json, write_versioned_json

        target = self.metadata_path(execution_id)
        if not target.exists():
            return
        current = ExecutionMetadata(**read_versioned_json(target))
        merged = current.model_copy(update=updates)
        write_versioned_json(target, merged.model_dump(mode="json"))

    def next_execution_id(self) -> str:
        """Return the execution_id for this attempt.

        Delegates to :func:`molexp.workspace.utils.derive_execution_id` — the
        single source of execution-id derivation shared with the workflow
        runtime — so the executions/<id>/ directory written at workflow start
        and the execution_history entry share the same identifier.
        """
        return derive_execution_id(self._run.id, self._work_dir / "executions")

    def close_record(
        self, execution_id: str, status: str, finished_at: datetime
    ) -> list[ExecutionRecord]:
        """Return execution_history with *execution_id*'s record closed."""
        history = list(self._run.metadata.execution_history)
        for i, entry in enumerate(history):
            if entry.execution_id == execution_id:
                history[i] = entry.model_copy(update={"finished_at": finished_at, "status": status})
                return history
        return history
