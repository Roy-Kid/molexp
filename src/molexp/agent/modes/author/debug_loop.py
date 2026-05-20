"""Per-task subprocess debug loop — run, diagnose, repair, re-run.

After a task's implementation and test land on disk, :func:`run_task_debug_loop`
runs that test in an *isolated subprocess* — via the harness
:class:`~molexp.agent.harness.execution_env.ExecutionEnv`, never raw
:mod:`subprocess` — with a hard timeout and a ``cwd`` confined to the
generated ``src/`` tree. On a non-zero exit or a timeout, the pytest
traceback plus the current impl + test source are fed back to the LLM
(via the harness :class:`~molexp.agent.router.Router`) for a targeted
fix; the impl is rewritten, the test re-run, bounded by an attempt
budget. Every iteration produces a
:class:`~molexp.agent.modes._planning.PlanDiff` (see :mod:`repair`); on
budget exhaustion the loop plants a final repair signal.

Isolation here is deliberately the pragmatic floor — a separate
subprocess + hard timeout + confined ``cwd``. A container/VM sandbox is
a future spec.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from molexp.agent.harness.execution_env import ExecutionEnv, ExecutionError
from molexp.agent.modes._planning import PlanDiff, PlanGraph
from molexp.agent.modes.author.codegen import GeneratedModule
from molexp.agent.modes.author.repair import build_repair_diff
from molexp.agent.router import ModelTier, Router

__all__ = [
    "DebugLoopResult",
    "TaskTestOutcome",
    "run_subprocess_test",
    "run_task_debug_loop",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")

_DEFAULT_TIMEOUT_SECONDS = 30.0
"""Per-test hard timeout — a runaway generated test can never wedge a run."""


class TaskTestOutcome(BaseModel):
    """The outcome of one isolated pytest run for a generated task.

    Attributes:
        passed: Whether pytest exited 0.
        timed_out: Whether the run was killed by the hard timeout.
        exit_code: The subprocess exit code (``-1`` on timeout).
        traceback: The captured stdout+stderr (the failure detail).
    """

    model_config = _FROZEN

    passed: bool
    timed_out: bool
    exit_code: int
    traceback: str = ""


class DebugLoopResult(BaseModel):
    """The outcome of one :func:`run_task_debug_loop` call.

    Attributes:
        task_id: The task the loop ran for.
        converged: Whether the test passed within the attempt budget.
        attempts: How many run attempts were made.
        diffs: One :class:`PlanDiff` per failed iteration (audit trail).
        final_outcome: The last :class:`TaskTestOutcome` observed.
    """

    model_config = _FROZEN

    task_id: str
    converged: bool
    attempts: int
    diffs: tuple[PlanDiff, ...] = ()
    final_outcome: TaskTestOutcome


def run_subprocess_test(
    *,
    execution_env: ExecutionEnv,
    test_path: Path,
    cwd: Path,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> TaskTestOutcome:
    """Run one generated pytest module in an isolated subprocess.

    The subprocess runs ``python -m pytest <test_path> -q`` with ``cwd``
    confined to ``cwd`` (the generated ``src/`` tree) and a hard
    ``timeout``. A timeout is captured as a failed
    :class:`TaskTestOutcome` (``timed_out=True``), not raised — the
    debug loop treats it as a normal failure.
    """
    import sys

    command = [sys.executable, "-m", "pytest", str(test_path), "-q", "-p", "no:cacheprovider"]
    env = {"PYTHONPATH": str(cwd), "PATH": _inherited_path()}
    try:
        result = execution_env.exec(command, cwd=cwd, env=env, timeout=timeout)
    except ExecutionError as exc:
        message = str(exc)
        if message.startswith("timeout:"):
            return TaskTestOutcome(passed=False, timed_out=True, exit_code=-1, traceback=message)
        return TaskTestOutcome(passed=False, timed_out=False, exit_code=-1, traceback=message)
    return TaskTestOutcome(
        passed=result.exit_code == 0,
        timed_out=False,
        exit_code=result.exit_code,
        traceback=(result.stdout + "\n" + result.stderr).strip(),
    )


def _inherited_path() -> str:
    """Return the parent ``PATH`` so the child can find ``python`` deps."""
    import os

    return os.environ.get("PATH", "")


_REPAIR_SYSTEM_PROMPT = (
    "A generated task implementation failed its generated pytest test. "
    "Given the current implementation source, the test source, and the "
    "pytest traceback, rewrite ONLY the implementation module so the "
    "test passes. Keep it a molexp.workflow.Task subclass. Return the "
    "full corrected module source."
)


async def run_task_debug_loop(
    *,
    task_id: str,
    impl_path: Path,
    test_path: Path,
    plan_graph: PlanGraph,
    router: Router,
    execution_env: ExecutionEnv,
    src_root: Path,
    debug_attempts: int,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    tier: ModelTier = ModelTier.DEFAULT,
) -> DebugLoopResult:
    """Run one task's test, repairing the impl until it passes or the budget runs out.

    On the first run the test is executed as-is. On a failure (non-zero
    exit or timeout) the LLM is asked for a targeted fix, the impl is
    rewritten, and the test re-run — up to ``debug_attempts`` total
    runs. Every failed iteration records a
    :class:`~molexp.agent.modes._planning.PlanDiff`. The loop never
    mutates the ``PlanGraph``; it returns its audit trail in
    :attr:`DebugLoopResult.diffs`.
    """
    diffs: list[PlanDiff] = []
    outcome = run_subprocess_test(
        execution_env=execution_env, test_path=test_path, cwd=src_root, timeout=timeout
    )
    attempts = 1
    while not outcome.passed and attempts < max(debug_attempts, 1):
        diff = build_repair_diff(
            plan_graph=plan_graph,
            step_id=task_id,
            traceback=outcome.traceback,
            attempt=attempts,
        )
        diffs.append(diff)
        await _apply_targeted_fix(
            task_id=task_id,
            impl_path=impl_path,
            test_path=test_path,
            traceback=outcome.traceback,
            router=router,
            tier=tier,
        )
        outcome = run_subprocess_test(
            execution_env=execution_env, test_path=test_path, cwd=src_root, timeout=timeout
        )
        attempts += 1

    if not outcome.passed:
        # Budget exhausted — plant a final repair diff so the caller can
        # emit a repair_proposed event and fail the plan.
        diffs.append(
            build_repair_diff(
                plan_graph=plan_graph,
                step_id=task_id,
                traceback=outcome.traceback,
                attempt=attempts,
            )
        )
    return DebugLoopResult(
        task_id=task_id,
        converged=outcome.passed,
        attempts=attempts,
        diffs=tuple(diffs),
        final_outcome=outcome,
    )


async def _apply_targeted_fix(
    *,
    task_id: str,
    impl_path: Path,
    test_path: Path,
    traceback: str,
    router: Router,
    tier: ModelTier,
) -> None:
    """Ask the LLM for a corrected impl and rewrite ``impl_path`` in place."""
    from molexp.workspace import atomic_write_text

    impl_source = impl_path.read_text(encoding="utf-8") if impl_path.exists() else ""
    test_source = test_path.read_text(encoding="utf-8") if test_path.exists() else ""
    user = (
        f"task_id: {task_id}\n\n"
        f"--- current implementation ---\n{impl_source}\n\n"
        f"--- test ---\n{test_source}\n\n"
        f"--- pytest traceback ---\n{traceback}\n"
    )
    fixed = await router.complete_structured(
        tier=tier,
        system=_REPAIR_SYSTEM_PROMPT,
        user=user,
        schema=GeneratedModule,
        node_id=f"RunTaskDebugLoop/{task_id}",
    )
    impl_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(impl_path, fixed.source)
