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

import importlib.util
import json
import re
import sys
from typing import TYPE_CHECKING, ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError, StagePersistedFailureError
from molexp.harness.schemas import (
    CommandResult,
    CommandSpec,
    PlanArtifactRef,
    TestResult,
    TestSource,
)
from molexp.harness.stages._resolve import require_latest
from molexp.workspace.utils import generate_id

if TYPE_CHECKING:
    from molexp.harness.executors import Executor

__all__ = ["ExecuteTests"]

#: CPython's ModuleNotFoundError spelling in pytest output.
_MISSING_MODULE_RE = re.compile(r"No module named '([^']+)'")

#: pytest's generic complaint when async tests run without an async plugin.
_ASYNC_UNSUPPORTED_RE = re.compile(r"async def functions are not natively supported")


class ExecuteTests(Stage):
    """Run the generated pytest module; persist a TestResult; fail-stop on red."""

    name: ClassVar[str] = "execute_tests"

    def __init__(self, executor: Executor, *, timeout_s: int = 600) -> None:
        self._executor = executor
        self._timeout_s = timeout_s

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        self._require_pytest()
        ts_ref = require_latest(ctx, "test_source", stage=self.name)
        ts = self._parse_test_source(ctx, ts_ref.id)

        # Multi-file: collect every per-task test file (e.g. tests/test_*.py);
        # single-file: the one module. ``python -m pytest`` puts cwd (generated/)
        # on sys.path so the tests' ``from workflow import …`` resolves either way.
        targets = [f.path for f in ts.files] or [f"{ts.module_name}.py"]
        cmd = [sys.executable, "-m", "pytest", *targets, "-q"]
        # molexp task bodies are async-first, so the generated tests are async
        # `def`s. pytest-asyncio (a declared runtime dep of the plan path) is
        # strict-mode by default and REJECTS bare async tests; the materialized
        # tree carries no pytest config, so the mode is set at the invocation.
        if importlib.util.find_spec("pytest_asyncio") is not None:
            cmd += ["-o", "asyncio_mode=auto"]
        spec = CommandSpec(
            cmd=cmd,
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
            detail = self._environment_detail(ctx, command)
            # Leave the captured pytest output as test_code_feedback so the
            # NEXT GenerateTestCode regeneration (repair loop and the
            # post-eviction re-run both thread feedback_inputs(...,
            # "test_code_feedback")) learns from this failure instead of
            # re-rolling blind — three consecutive production re-runs each
            # invented a different wrong assertion.
            ctx.artifact_store.put_text(
                kind="test_code_feedback",
                text=(
                    "The previously generated tests FAILED when executed "
                    f"(pytest exit {command.exit_code}). Fix the tests (or the "
                    "assertions' expectations) so they pass against the actual "
                    "workflow source. Captured pytest output:\n\n"
                    + self._captured_output(ctx, command)
                ),
                created_by="ExecuteTests",
                parent_ids=[ts_ref.id, result_ref.id],
            )
            raise StagePersistedFailureError(
                result_ref,
                f"generated tests failed (pytest exit {command.exit_code}){detail}; "
                "workflow execution is blocked",
            )
        return result_ref

    def _environment_detail(self, ctx: HarnessRunContext, command: CommandResult) -> str:
        """One ``" — …"`` clause naming an environment gap behind the red run.

        Two gaps routinely masquerade as plain test failures — an experiment
        dependency the interpreter cannot import (e.g. numpy), and async
        generated tests running without pytest-asyncio (molexp task bodies
        are async-first, so ``@pytest.mark.asyncio`` is the norm). Name the
        gap and the fix; return ``""`` for a genuinely red test run.
        """
        output = self._captured_output(ctx, command)
        missing = sorted(set(_MISSING_MODULE_RE.findall(output)))
        if missing:
            return (
                f" — the test interpreter cannot import: {', '.join(missing)}; "
                f"install the experiment's dependencies into {sys.executable}"
            )
        if _ASYNC_UNSUPPORTED_RE.search(output):
            return (
                " — the generated tests are async but the test interpreter lacks "
                'pytest-asyncio (carried by `pip install "molexp[agent]"`)'
            )
        return ""

    @staticmethod
    def _captured_output(ctx: HarnessRunContext, command: CommandResult) -> str:
        """The run's combined stdout + stderr text, best-effort."""
        chunks: list[str] = []
        for ref in (command.stdout_artifact, command.stderr_artifact):
            try:
                raw = ctx.artifact_store.get(ref.id)
            except Exception:  # pragma: no cover — output artifact gone; skip
                continue
            chunks.append(raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw)
        return "\n".join(chunks)

    def _require_pytest(self) -> None:
        """Fail fast — with the real reason — when the test runner is absent.

        The command below targets ``sys.executable``, so an in-process
        ``find_spec`` check is a faithful precondition for the exact
        interpreter that will run it. Without this guard a missing pytest
        surfaced as ``generated tests failed (pytest exit 1)`` — blaming the
        LLM-generated tests for an environment gap the operator must fix.
        """
        if importlib.util.find_spec("pytest") is None:
            raise StageExecutionError(
                f"stage {self.name!r} needs pytest to run the generated per-task "
                f"tests, but {sys.executable} cannot import it — install it into "
                'that environment (`pip install "molexp[agent]"` carries it, or '
                "`pip install pytest`). The generated tests were NOT run."
            )

    def _parse_test_source(self, ctx: HarnessRunContext, artifact_id: str) -> TestSource:
        raw = ctx.artifact_store.get(artifact_id)
        try:
            return TestSource.model_validate_json(raw)
        except Exception as exc:
            raise StageExecutionError(
                f"stage {self.name!r} could not parse the 'test_source' artifact "
                f"{artifact_id!r}: {exc!r}"
            ) from exc
