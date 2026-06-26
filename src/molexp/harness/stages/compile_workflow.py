"""``CompileWorkflow`` — plan step 7: compile + validate inputs, no science.

Runs the materialized ``run_workflow.py --compile-only`` through the
**injected** :class:`Executor` in the run's ``generated/`` directory. The
compile-only driver branch does ``build_workflow().compile()`` (proving the
generated source compiles and the DAG builds) and confirms the embedded
params satisfy the root inputs — but executes **no** task bodies, so no real
compute runs. The real ``molexp.workflow`` engine loads only inside that
subprocess; this module never imports it.

The :class:`CommandResult` is lifted into an :class:`ExecutionResult`
artifact tagged ``metadata["mode"]="compile"`` (a dry result, distinct from
a real ``ExecuteWorkflow`` run). Nonzero exit persists the failed result
first, then raises — a workflow that fails to compile blocks the plan's
review gate.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError, StagePersistedFailureError
from molexp.harness.schemas import ArtifactRef, CommandSpec, ExecutionResult, WorkflowSource
from molexp.harness.stages._resolve import require_latest
from molexp.workspace.utils import generate_id

if TYPE_CHECKING:
    from molexp.harness.executors import Executor

__all__ = ["CompileWorkflow"]


class CompileWorkflow(Stage):
    """Compile the materialized workflow (no task execution); persist a dry ExecutionResult."""

    name: ClassVar[str] = "compile_workflow"

    def __init__(self, executor: Executor, *, timeout_s: int = 600) -> None:
        self._executor = executor
        self._timeout_s = timeout_s

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        ws_ref = require_latest(ctx, "workflow_source", stage=self.name)
        ws = self._parse_workflow_source(ctx, ws_ref.id)

        spec = CommandSpec(
            cmd=[sys.executable, "run_workflow.py", "--compile-only"],
            cwd=str(ctx.workspace_root / "generated"),
            timeout_s=self._timeout_s,
        )
        command = await self._executor.execute(spec, artifact_store=ctx.artifact_store)

        succeeded = command.exit_code == 0
        result = ExecutionResult(
            id=f"compile-result-{generate_id()}",
            bound_workflow_id=ws.bound_workflow_id,
            status="succeeded" if succeeded else "failed",
            exit_code=command.exit_code,
            started_at=command.started_at,
            ended_at=command.ended_at,
            outputs={},
            output_artifacts=command.output_artifacts,
            stdout=command.stdout_artifact,
            stderr=command.stderr_artifact,
            metadata={**command.metadata, "mode": "compile"},
        )
        result_ref = ctx.artifact_store.put_json(
            kind="execution_result",
            obj=json.loads(result.model_dump_json()),
            created_by="CompileWorkflow",
            parent_ids=[ws_ref.id],
        )
        if not succeeded:
            raise StagePersistedFailureError(
                result_ref,
                f"workflow compile exited {command.exit_code}; see the persisted "
                "execution_result for stdout/stderr artifacts",
            )
        return result_ref

    def _parse_workflow_source(self, ctx: HarnessRunContext, artifact_id: str) -> WorkflowSource:
        raw = ctx.artifact_store.get(artifact_id)
        try:
            return WorkflowSource.model_validate_json(raw)
        except Exception as exc:
            raise StageExecutionError(
                f"stage {self.name!r} could not parse the 'workflow_source' artifact "
                f"{artifact_id!r}: {exc!r}"
            ) from exc
