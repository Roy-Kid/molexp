"""``ExecuteWorkflow`` — run the materialized driver through an executor.

Executes ``python run_workflow.py`` in the run's ``generated/`` directory via
the **injected** :class:`Executor` with ``expected_outputs=["outputs.json"]``
(the executor collects the file as an ``output_file`` artifact). The real
``molexp.workflow`` engine runs only inside that subprocess — this module
never imports it, keeping the harness import-guard green. The
:class:`CommandResult` is lifted into the new :class:`ExecutionResult`
schema and persisted as an ``execution_result`` artifact; nonzero exit
persists a failed result first and then raises
:class:`StagePersistedFailureError`.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any, ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError, StagePersistedFailureError
from molexp.harness.schemas import (
    ArtifactRef,
    CommandResult,
    CommandSpec,
    ExecutionResult,
    WorkflowSource,
)
from molexp.harness.stages._resolve import require_latest
from molexp.workspace.utils import generate_id

if TYPE_CHECKING:
    from molexp.harness.executors import Executor

__all__ = ["ExecuteWorkflow"]


class ExecuteWorkflow(Stage):
    """Run the materialized driver; persist an ExecutionResult; fail-stop on error."""

    name: ClassVar[str] = "execute_workflow"

    def __init__(self, executor: Executor, *, timeout_s: int = 3600) -> None:
        self._executor = executor
        self._timeout_s = timeout_s

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        ws_ref = require_latest(ctx, "workflow_source", stage=self.name)
        ws = self._parse_workflow_source(ctx, ws_ref.id)

        spec = CommandSpec(
            cmd=[sys.executable, "run_workflow.py"],
            cwd=str(ctx.workspace_root / "generated"),
            timeout_s=self._timeout_s,
            expected_outputs=["outputs.json"],
        )
        command = await self._executor.execute(spec, artifact_store=ctx.artifact_store)

        succeeded = command.exit_code == 0
        result = ExecutionResult(
            id=f"execution-result-{generate_id()}",
            bound_workflow_id=ws.bound_workflow_id,
            status="succeeded" if succeeded else "failed",
            exit_code=command.exit_code,
            started_at=command.started_at,
            ended_at=command.ended_at,
            outputs=self._parse_outputs(ctx, command),
            output_artifacts=command.output_artifacts,
            stdout=command.stdout_artifact,
            stderr=command.stderr_artifact,
            metadata=command.metadata,
        )
        result_ref = ctx.artifact_store.put_json(
            kind="execution_result",
            obj=json.loads(result.model_dump_json()),
            created_by="ExecuteWorkflow",
            parent_ids=[ws_ref.id],
        )
        if not succeeded:
            raise StagePersistedFailureError(
                result_ref,
                f"workflow driver exited {command.exit_code}; see the persisted "
                "execution_result for stdout/stderr artifacts",
            )
        return result_ref

    @staticmethod
    def _parse_outputs(ctx: HarnessRunContext, command: CommandResult) -> dict[str, Any]:
        """Parse the collected outputs.json artifact; degrade to {} quietly."""
        for ref in command.output_artifacts:
            if not ref.uri.endswith("outputs.json"):
                continue
            try:
                parsed = json.loads(ctx.artifact_store.get(ref.id))
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _parse_workflow_source(self, ctx: HarnessRunContext, artifact_id: str) -> WorkflowSource:
        raw = ctx.artifact_store.get(artifact_id)
        try:
            return WorkflowSource.model_validate_json(raw)
        except Exception as exc:
            raise StageExecutionError(
                f"stage {self.name!r} could not parse the 'workflow_source' artifact "
                f"{artifact_id!r}: {exc!r}"
            ) from exc
