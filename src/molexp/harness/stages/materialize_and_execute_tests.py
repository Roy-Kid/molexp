"""``MaterializeAndExecuteTests`` — step-7 dry-run with in-pipeline repair.

Design hole this closes: :class:`ExecuteTests` used to sit *outside* any
:class:`RepairLoop`. Structural validation of the test source could pass while
pytest still failed (wrong ``RegisterMetric`` field, bad mock, NameError). The
failure wrote ``test_code_feedback`` for a *future* plan re-run, but the same
``molexp plan`` invocation died after one red pytest — burning a full LLM
pipeline to re-learn the same mistake on resume.

This stage keeps Materialize + ExecuteTests in one place and, on a red run,
re-invokes :class:`GenerateTestCode` (which already threads
``test_code_feedback``) + :class:`ValidateTestSource`, rematerializes, and
retries pytest up to ``attempts`` times — same convergence contract as the
generate→validate repair loops above it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mollog import get_logger

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError, StagePersistedFailureError
from molexp.harness.schemas import PlanArtifactRef
from molexp.harness.stages.execute_tests import ExecuteTests
from molexp.harness.stages.generate_test_code import GenerateTestCode
from molexp.harness.stages.materialize_execution import MaterializeExecution
from molexp.harness.stages.validate_test_source import ValidateTestSource

if TYPE_CHECKING:
    from molexp.harness.executors import Executor

__all__ = ["MaterializeAndExecuteTests"]

_LOG = get_logger(__name__)


class MaterializeAndExecuteTests(Stage):
    """Materialize generated programs, run pytest, regenerate tests on red."""

    name: ClassVar[str] = "materialize_and_execute_tests"

    def __init__(self, executor: Executor, *, attempts: int = 4, timeout_s: int = 600) -> None:
        if attempts < 1:
            raise ValueError("MaterializeAndExecuteTests attempts must be >= 1")
        self._executor = executor
        self._attempts = attempts
        self._timeout_s = timeout_s
        self._materialize = MaterializeExecution()
        self._execute = ExecuteTests(executor, timeout_s=timeout_s)
        self._generate = GenerateTestCode()
        self._validate = ValidateTestSource()

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        last: StageExecutionError | None = None
        for attempt in range(1, self._attempts + 1):
            if attempt > 1:
                _LOG.warning(
                    f"[materialize_and_execute_tests] attempt {attempt}/{self._attempts}: "
                    "regenerating test_source from test_code_feedback"
                )
                try:
                    await self._generate.run(ctx)
                    await self._validate.run(ctx)
                except StageExecutionError as exc:
                    last = exc
                    _LOG.warning(
                        f"[materialize_and_execute_tests] regenerate/validate failed: {exc}"
                    )
                    continue
            try:
                await self._materialize.run(ctx)
                return await self._execute.run(ctx)
            except StageExecutionError as exc:
                last = exc
                # ExecuteTests already wrote test_code_feedback on red pytest.
                _LOG.warning(
                    f"[materialize_and_execute_tests] attempt {attempt}/{self._attempts} "
                    f"failed dry-run — {exc}"
                )
                if not isinstance(exc, StagePersistedFailureError) and attempt >= self._attempts:
                    raise
                continue
        assert last is not None
        raise last
