"""``ExecuteTests`` — run the materialized pytest module through an executor.

The generated tests are the gate in front of workflow execution: this stage
runs ``python -m pytest <test_module>.py -q`` in the run's ``generated/``
directory via the **injected** :class:`Executor` (runtime container →
constructor, never ctx), maps the :class:`CommandResult` onto the existing
:class:`TestResult` schema, persists it as a ``test_result`` artifact, and —
on any nonzero exit — raises :class:`StagePersistedFailureError` *after*
persisting. Red tests therefore unconditionally block the downstream
:class:`ExecuteWorkflow` stage; there is deliberately no
``raise_on_failure`` knob here (run-on-red would be a mode-level policy).
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError, StagePersistedFailureError
from molexp.harness.schemas import ArtifactRef, CommandSpec, TestResult, TestSource
from molexp.harness.stages._resolve import require_latest
from molexp.workspace.utils import generate_id

if TYPE_CHECKING:
    from molexp.harness.executors import Executor

__all__ = ["ExecuteTests"]


class ExecuteTests(Stage):
    """Run the generated pytest module; persist a TestResult; fail-stop on red."""

    name: ClassVar[str] = "execute_tests"

    def __init__(self, executor: Executor, *, timeout_s: int = 600) -> None:
        self._executor = executor
        self._timeout_s = timeout_s

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        ts_ref = require_latest(ctx, "test_source", stage=self.name)
        ts = self._parse_test_source(ctx, ts_ref.id)

        # Multi-file: collect every per-task test file (e.g. tests/test_*.py);
        # single-file: the one module. ``python -m pytest`` puts cwd (generated/)
        # on sys.path so the tests' ``from workflow import …`` resolves either way.
        targets = [f.path for f in ts.files] or [f"{ts.module_name}.py"]
        spec = CommandSpec(
            cmd=[sys.executable, "-m", "pytest", *targets, "-q"],
            cwd=str(ctx.workspace_root / "generated"),
            timeout_s=self._timeout_s,
        )
        command = await self._executor.execute(spec, artifact_store=ctx.artifact_store)

        passed = command.exit_code == 0
        result = TestResult(
            id=f"test-result-{generate_id()}",
            test_spec_id=ts.test_spec_id,
            status="passed" if passed else "failed",
            stdout=command.stdout_artifact,
            stderr=command.stderr_artifact,
            reason=None if passed else f"pytest exited {command.exit_code}",
        )
        result_ref = ctx.artifact_store.put_json(
            kind="test_result",
            obj=json.loads(result.model_dump_json()),
            created_by="ExecuteTests",
            parent_ids=[ts_ref.id],
        )
        if not passed:
            raise StagePersistedFailureError(
                result_ref,
                f"generated tests failed (pytest exit {command.exit_code}); "
                "workflow execution is blocked",
            )
        return result_ref

    def _parse_test_source(self, ctx: HarnessRunContext, artifact_id: str) -> TestSource:
        raw = ctx.artifact_store.get(artifact_id)
        try:
            return TestSource.model_validate_json(raw)
        except Exception as exc:
            raise StageExecutionError(
                f"stage {self.name!r} could not parse the 'test_source' artifact "
                f"{artifact_id!r}: {exc!r}"
            ) from exc
