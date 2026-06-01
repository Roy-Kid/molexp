"""``DryRunExecutor`` — simulated executor with no real subprocess.

Persists empty stdout/stderr artifacts and returns ``exit_code=0``. Use to
smoke-test pipeline shape without burning compute.
"""

from __future__ import annotations

from datetime import UTC, datetime

from molexp.harness.schemas import CommandResult, CommandSpec
from molexp.harness.store.artifact_store import ArtifactStore

__all__ = ["DryRunExecutor"]


class DryRunExecutor:
    """No-op executor: returns success without running anything."""

    async def execute(
        self,
        spec: CommandSpec,
        *,
        artifact_store: ArtifactStore,
    ) -> CommandResult:
        started = datetime.now(tz=UTC)
        stdout_ref = artifact_store.put_text(
            kind="stdout",
            text=f"[DryRunExecutor] cmd={spec.cmd}\n",
            created_by="DryRunExecutor",
            parent_ids=[],
        )
        stderr_ref = artifact_store.put_text(
            kind="stderr",
            text="",
            created_by="DryRunExecutor",
            parent_ids=[],
        )
        ended = datetime.now(tz=UTC)
        return CommandResult(
            exit_code=0,
            started_at=started,
            ended_at=ended,
            stdout_artifact=stdout_ref,
            stderr_artifact=stderr_ref,
            output_artifacts=[],
            metadata={"executor": "DryRunExecutor", "dry_run": "true"},
        )
