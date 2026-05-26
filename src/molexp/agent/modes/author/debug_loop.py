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

The repair path uses the same constrained
:class:`~molexp.agent.modes.author.codegen.TaskImplDraft` schema as the
initial codegen — the LLM fills in class name + imports + execute body;
:func:`~molexp.agent.modes.author.codegen.assemble_impl_module` wraps
them in the canonical ``Task`` subclass shape. Before writing the
repaired impl to disk, the assembled source must pass the same gate
the initial codegen does
(:func:`~molexp.agent.modes.author.codegen.validate_assembled_impl`).
A failed gate is fed back into the next repair attempt inside the same
debug-loop iteration; only test failures advance the outer attempt
counter.

Isolation here is deliberately the pragmatic floor — a separate
subprocess + hard timeout + confined ``cwd``. A container/VM sandbox is
a future spec.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.harness.execution_env import ExecutionEnv, ExecutionError
from molexp.agent.modes._planning import PlanDiff, PlanGraph
from molexp.agent.modes.author.codegen import (
    RepairDecision,
    assemble_impl_module,
    validate_assembled_impl,
    validate_test_source,
)
from molexp.agent.modes.author.repair import build_repair_diff
from molexp.agent.router import ModelTier, Router

__all__ = [
    "DebugLoopResult",
    "TaskTestOutcome",
    "run_subprocess_test",
    "run_task_debug_loop",
]

_LOG = get_logger(__name__)

_FROZEN = ConfigDict(frozen=True, extra="forbid")

_DEFAULT_TIMEOUT_SECONDS = 30.0
"""Per-test hard timeout — a runaway generated test can never wedge a run."""

_REPAIR_GATE_RETRY_BUDGET = 3
"""How many TaskImplDraft attempts ``_apply_targeted_fix`` makes within a
single debug-loop iteration before giving up. Each attempt feeds the
gate verdict (syntax / shape / evidence) back into the next prompt, so
recovery from a one-off mistake is cheap. Shape-level errors don't
advance the outer attempt counter — that budget is reserved for
test-failure-driven repair iterations."""

_FALLBACK_REPAIR_TIMEOUT_SECONDS = 180.0
"""Wall-clock budget for the no-MCP fallback repair (one
``router.complete_structured`` call). The MCP-attached repair has its
own ``timeout_seconds`` inside ``build_repair_callable``; this constant
gives the fallback path the same upper bound so a hung provider can't
wedge the debug loop indefinitely when ``repair_model`` is ``None``."""


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


class _RepairGateExhausted(Exception):
    """Raised by ``_apply_targeted_fix`` when every gate-retry slot is spent.

    Carries the last gate verdict so the caller can surface it in the
    final ``TaskTestOutcome.traceback``.
    """

    def __init__(self, last_issue: str) -> None:
        super().__init__(last_issue)
        self.last_issue = last_issue


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
    env = {
        "PYTHONPATH": str(cwd),
        "PATH": _inherited_path(),
        # Write no .pyc files: a repaired implementation can land with the
        # same byte size and same integer-second mtime as the original,
        # which would let Python reuse a stale cached bytecode and poison
        # the re-run. A hermetic subprocess always sees the current source.
        "PYTHONDONTWRITEBYTECODE": "1",
    }
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
    "Either side may carry the bug. Diagnose the root cause first, "
    "then emit a RepairDecision targeting whichever file (or both) "
    "needs to change.\n"
    "\n"
    "WHICH FILE IS BROKEN — look at the traceback:\n"
    "  - the failure happens BEFORE the task function is called "
    "(error in a fixture, setUp, stub-class instantiation, the test "
    "module's imports, …) → the TEST has the bug → emit "
    "`test_source`.\n"
    "  - the failure is the impl raising an exception or returning the "
    "wrong shape, and the test's assertion is reasonable given the "
    "PlanStep's declared inputs/outputs → the IMPL has the bug → "
    "emit `impl`.\n"
    "  - the test asserts content the PlanStep doesn't actually "
    "promise (exact whitespace, exact numeric value, specific "
    "ordering, …) → the TEST over-specifies → emit `test_source` "
    "with a tighter, shape-level assertion.\n"
    "  - both files need work → emit both.\n"
    "\n"
    "ALWAYS write the `diagnosis` field first (one sentence naming the "
    "root-cause file and the underlying mistake). Both fields go "
    "through the same gates that initial codegen does: the IMPL "
    "through shape + evidence, the TEST through syntax + evidence.\n"
    "\n"
    "The codegen layer assembles the impl function wrapper around your "
    "draft (function name, docstring, input bindings, return shape "
    "from the PlanStep) — you contribute the imports and body for the "
    "impl; for the test you contribute the full pytest module source."
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
    repair: Callable[[str], Awaitable[RepairDecision]] | None = None,
) -> DebugLoopResult:
    """Run one task's test, repairing impl or test until it passes or the budget runs out.

    On the first run the test is executed as-is. On a failure (non-zero
    exit or timeout) the LLM is asked for a
    :class:`~molexp.agent.modes.author.codegen.RepairDecision` —
    diagnose the root cause, then patch the impl, the test, or both.
    Each patched file is gated identically to the initial codegen
    (impl through shape + evidence, test through syntax + evidence)
    before landing on disk. The test is re-run; up to ``debug_attempts``
    iterations total. Every failed iteration records a
    :class:`~molexp.agent.modes._planning.PlanDiff`. The loop never
    mutates the ``PlanGraph``; it returns its audit trail in
    :attr:`DebugLoopResult.diffs`.

    ``repair`` is an optional source-grounded repair callable (typically
    built by ``_pydanticai/debug_repair.build_repair_callable`` — an
    MCP-attached agent that can search the project source before
    drafting). When supplied, every repair step routes through it;
    when ``None`` the loop falls back to the no-tool
    ``router.complete_structured`` path so existing callers keep
    working.
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
        try:
            await _apply_targeted_fix(
                task_id=task_id,
                impl_path=impl_path,
                test_path=test_path,
                traceback=outcome.traceback,
                plan_graph=plan_graph,
                router=router,
                tier=tier,
                repair=repair,
            )
        except _RepairGateExhausted as exc:
            _LOG.warning(
                f"[debug-loop] {task_id}: gate retry budget exhausted — {exc.last_issue}"
            )
            outcome = TaskTestOutcome(
                passed=False,
                timed_out=False,
                exit_code=-1,
                traceback=(
                    f"repair gate exhausted after {_REPAIR_GATE_RETRY_BUDGET} attempts: "
                    f"{exc.last_issue}"
                ),
            )
            attempts += 1
            break
        except TimeoutError as exc:
            _LOG.warning(f"[debug-loop] {task_id}: repair wall-clock exceeded — {exc!r}")
            outcome = TaskTestOutcome(
                passed=False,
                timed_out=True,
                exit_code=-1,
                traceback=f"repair wall-clock exceeded: {exc!r}",
            )
            attempts += 1
            break
        except Exception as exc:
            # Any other exception from the repair callable (pydantic-ai
            # validation failure surfaced through retries, MCP transport
            # error, network blip) is isolated to this task — convert to
            # a failed iteration so ``asyncio.gather`` doesn't cancel the
            # sibling debug loops and abort the whole AuthorMode run.
            _LOG.warning(
                f"[debug-loop] {task_id}: repair raised "
                f"{type(exc).__name__}: {exc}"
            )
            outcome = TaskTestOutcome(
                passed=False,
                timed_out=False,
                exit_code=-1,
                traceback=f"repair raised {type(exc).__name__}: {exc!r}",
            )
            attempts += 1
            break
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
    plan_graph: PlanGraph,
    router: Router,
    tier: ModelTier,
    repair: Callable[[str], Awaitable[RepairDecision]] | None = None,
) -> None:
    """Ask the LLM for a :class:`RepairDecision`, gate it, write changed files.

    Drives an inline gate-retry loop inside one debug-loop iteration:
    request a :class:`RepairDecision`, run the impl/test through their
    respective gates (impl: shape + evidence; test: syntax + evidence),
    and only write the file(s) the decision targets. On gate failure
    the verdict is fed into the next prompt up to
    :data:`_REPAIR_GATE_RETRY_BUDGET` attempts. On exhaustion raise
    :class:`_RepairGateExhausted` so the caller surfaces it as a failed
    iteration rather than writing a broken patch to disk.

    Either ``decision.impl`` or ``decision.test_source`` may be ``None``
    (the on-disk file is preserved) but at least one must be set — an
    empty decision is treated as a gate rejection so the model retries
    with a real patch.
    """
    from molexp.workspace import atomic_write_text

    step = plan_graph.step_by_id(task_id)
    if step is None:
        # The repair loop was asked to fix a task that the current plan
        # doesn't know about — leave the files untouched and bubble up
        # via the exhausted path so the caller records the failure.
        raise _RepairGateExhausted(
            f"plan has no step for task_id={task_id!r}; cannot repair"
        )

    from molexp.agent.modes.author.renderers import module_id

    impl_source = impl_path.read_text(encoding="utf-8") if impl_path.exists() else ""
    test_source = test_path.read_text(encoding="utf-8") if test_path.exists() else ""
    allowed_refs = sorted({ref for s in plan_graph.steps for ref in s.api_refs})
    refs_block = "\n".join(f"  - {ref}" for ref in allowed_refs) or "  (none)"
    input_locals = [
        f"  - {module_id(inp.name)}  (from PlanStep.io.inputs[{i}], "
        f"original name {inp.name!r})"
        for i, inp in enumerate(step.io.inputs)
        if inp.source_step is not None
    ]
    output_locals = [
        f"  - {module_id(name)}  (returned under key {name!r})"
        for name in step.io.outputs
    ]
    func_name = module_id(step.id)
    bindings_block = (
        f"TASK FUNCTION: ``async def {func_name}(ctx)``\n"
        f"TESTS IMPORT THE TASK VIA:\n"
        f"  ``from experiment.tasks.{func_name} import {func_name}``\n"
        f"(the only correct import path — the test subprocess runs with "
        f"`src/` on PYTHONPATH; never use a bare ``from {func_name} "
        f"import …`` or `sys.path` hacks.)\n\n"
        "INPUT LOCALS — auto-bound for the impl body before it runs:\n"
        + ("\n".join(input_locals) if input_locals else "  (none — root task)")
        + "\n\nOUTPUT LOCALS — the impl body must bind each of these:\n"
        + ("\n".join(output_locals) if output_locals else "  (none — return None)")
    )
    base_prompt = (
        "ALLOWED PROJECT IMPORTS — every project symbol referenced by "
        "the impl's `imports` OR the test's import statements MUST be "
        "drawn from this list (re-exports of the same symbol at a "
        "shorter path are fine; nothing else is). The codegen-evidence "
        "gate runs on both and rejects anything outside it.\n"
        f"{refs_block}\n\n"
        f"{bindings_block}\n\n"
        f"PlanStep:\n{step.model_dump_json(indent=2)}\n\n"
        f"--- current implementation ---\n{impl_source}\n\n"
        f"--- current test ---\n{test_source}\n\n"
        f"--- pytest traceback ---\n{traceback}\n"
    )

    last_issue: str | None = None
    for _attempt in range(_REPAIR_GATE_RETRY_BUDGET):
        prompt = base_prompt
        if last_issue is not None:
            prompt = (
                f"{base_prompt}\n\n"
                f"PREVIOUS REPAIR REJECTED BY CODEGEN GATE:\n  {last_issue}\n\n"
                "Re-emit the RepairDecision fixing that specific issue. "
                "The ALLOWED PROJECT IMPORTS list above still applies."
            )
        if repair is not None:
            decision = await repair(prompt)
        else:
            # Fallback router path — wrap in ``asyncio.wait_for`` so the
            # MCP-absent deployment has the same wall-clock bound as the
            # MCP-attached one. Without this a hung provider can wedge
            # the debug loop indefinitely.
            decision = await asyncio.wait_for(
                router.complete_structured(
                    tier=tier,
                    system=_REPAIR_SYSTEM_PROMPT,
                    user=prompt,
                    schema=RepairDecision,
                    node_id=f"RunTaskDebugLoop/{task_id}",
                ),
                timeout=_FALLBACK_REPAIR_TIMEOUT_SECONDS,
            )

        if decision.impl is None and decision.test_source is None:
            last_issue = (
                "RepairDecision is empty — at least one of `impl` or "
                "`test_source` must carry the patch"
            )
            continue
        if not decision.diagnosis.strip():
            last_issue = "RepairDecision.diagnosis is empty — name the root cause"
            continue

        # Gate impl (if changed)
        assembled_impl: str | None = None
        if decision.impl is not None:
            assembled_impl = assemble_impl_module(decision.impl, step)
            impl_issue = validate_assembled_impl(assembled_impl, plan_graph)
            if impl_issue is not None:
                last_issue = f"impl repair rejected: {impl_issue}"
                continue

        # Gate test (if changed)
        if decision.test_source is not None:
            test_issue = validate_test_source(decision.test_source, plan_graph)
            if test_issue is not None:
                last_issue = f"test repair rejected: {test_issue}"
                continue

        # Both gates passed — write the patched files atomically.
        if assembled_impl is not None:
            impl_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(impl_path, assembled_impl)
        if decision.test_source is not None:
            test_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(test_path, decision.test_source)
        _LOG.info(f"[debug-loop] {task_id}: {decision.diagnosis}")
        return

    assert last_issue is not None
    raise _RepairGateExhausted(last_issue)
