"""Runtime-failure classification + ``PlanDiff``-centric repair for RunMode.

When a materialized workflow step fails at *execution* time (not codegen
time — that is AuthorMode's debug loop), RunMode does not blindly re-run.
It first classifies the failure:

- :data:`RuntimeFailureKind.transient` — a flake worth retrying (a timed
  out subprocess, a temporarily unavailable resource). Retried per the
  step's :class:`~molexp.agent.modes._planning.RetryPolicy`.
- :data:`RuntimeFailureKind.structural` — a defect in the plan / generated
  code that retrying cannot fix. Escalated immediately.

:func:`apply_retry_policy` honours one step's ``RetryPolicy`` (its
``max_attempts`` and the ``on`` failure-tag allow-list). On retry
exhaustion *or* a structural classification, :func:`build_repair_diff`
emits a typed :class:`~molexp.agent.modes._planning.PlanDiff` against the
``PlanGraph``. A diff whose repair needs a *plan-shape* change — a step
added, removed, or structurally replaced — is wrapped in a
:class:`RepairEscalation` flagged ``requires_rematerialization=True`` and
``target_mode="author"``: the contract toward AuthorMode. RunMode emits
the escalation; it never re-runs AuthorMode itself.

Pure data + pure functions; no LLM, no I/O. Frozen-pydantic value types.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from molexp.agent.modes._planning import (
    DiffOpKind,
    PlanDiff,
    PlanGraph,
    PlanNodeOp,
    PlanStep,
    RetryPolicy,
)

__all__ = [
    "RepairEscalation",
    "RetryOutcome",
    "RuntimeFailure",
    "RuntimeFailureKind",
    "apply_retry_policy",
    "build_repair_diff",
    "classify_runtime_failure",
]

_RUNTIME_FAILURE_INVARIANT = "materialized_step_executes_successfully"
"""The invariant a step failure at execution time violates."""

# Substrings in an error message / type that mark a transient (retryable)
# failure. Anything else is treated as structural.
_TRANSIENT_MARKERS: tuple[str, ...] = (
    "timeout",
    "timederror",
    "temporarily unavailable",
    "connection reset",
    "connection refused",
    "resource temporarily",
)


class RuntimeFailureKind(StrEnum):
    """Coarse classification of a step's execution-time failure."""

    transient = "transient"
    structural = "structural"


class RuntimeFailure(BaseModel):
    """A typed description of one step's execution-time failure.

    Attributes:
        step_id: ``id`` of the failing :class:`PlanStep`.
        error_type: The exception class name (e.g. ``TimeoutError``).
        message: The exception message.
        kind: The classified :class:`RuntimeFailureKind`.
        attempts: How many attempts were spent before giving up.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: str
    error_type: str
    message: str
    kind: RuntimeFailureKind
    attempts: int = Field(default=1, ge=1)


class RetryOutcome(BaseModel):
    """The result of running one step's :class:`RetryPolicy`.

    Attributes:
        succeeded: Whether an attempt eventually succeeded.
        attempts: Total attempts spent (``1 <= attempts <= max_attempts``).
        last_failure: The final :class:`RuntimeFailure` when every attempt
            failed, else ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    succeeded: bool
    attempts: int = Field(ge=1)
    last_failure: RuntimeFailure | None = None


class RepairEscalation(BaseModel):
    """The contract RunMode emits when a repair needs AuthorMode re-entry.

    A :class:`PlanDiff` that only re-orders / no-ops within the existing
    plan shape can in principle be handled in place; a diff that *adds*
    or *removes* steps needs the workspace re-materialized — that is
    AuthorMode's job. RunMode emits this record and stops; the caller (an
    orchestrator) re-enters AuthorMode.

    Attributes:
        plan_id: The plan whose execution failed.
        diff: The structured :class:`PlanDiff` describing the repair.
        requires_rematerialization: ``True`` when the diff needs AuthorMode
            to regenerate source before another run.
        target_mode: The mode the escalation routes to — always
            ``"author"`` for a re-materialization escalation.
        rationale: Why the escalation is needed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str
    diff: PlanDiff
    requires_rematerialization: bool
    target_mode: str = "author"
    rationale: str = ""


def classify_runtime_failure(step: PlanStep, error: BaseException) -> RuntimeFailureKind:
    """Classify a step's execution-time failure as transient or structural.

    A failure is transient when its exception type or message matches a
    known-flaky marker *and* the step's :class:`RetryPolicy` opts into
    that failure tag (``RetryPolicy.on``). An empty ``on`` tuple means
    "retry any transient-looking failure". Everything else — and any
    failure on a step with ``max_attempts == 1`` — is structural.

    Args:
        step: The failing :class:`PlanStep` (its ``retry_policy`` gates
            whether a transient-looking failure is actually retried).
        error: The exception the step raised.

    Returns:
        The :class:`RuntimeFailureKind`.
    """
    haystack = f"{type(error).__name__} {error}".lower()
    looks_transient = any(marker in haystack for marker in _TRANSIENT_MARKERS)
    if not looks_transient:
        return RuntimeFailureKind.structural
    policy = step.retry_policy
    if policy.max_attempts <= 1:
        return RuntimeFailureKind.structural
    if policy.on and not _failure_tag_allowed(haystack, policy.on):
        return RuntimeFailureKind.structural
    return RuntimeFailureKind.transient


def _failure_tag_allowed(haystack: str, on: tuple[str, ...]) -> bool:
    """Return whether any ``RetryPolicy.on`` tag matches the failure text."""
    return any(tag.lower() in haystack for tag in on)


async def apply_retry_policy(
    step: PlanStep,
    attempt: Callable[[int], Awaitable[None]],
) -> RetryOutcome:
    """Run ``attempt`` under ``step``'s :class:`RetryPolicy`.

    ``attempt(attempt_number)`` is awaited once per try (1-based). A try
    that returns normally succeeds; a try that raises is classified by
    :func:`classify_runtime_failure`. A *structural* failure stops the
    loop immediately — retrying cannot help. A *transient* failure is
    retried until :attr:`RetryPolicy.max_attempts` is reached.

    Args:
        step: The :class:`PlanStep` whose ``retry_policy`` governs the loop.
        attempt: An async callable run once per attempt; it raises on
            failure.

    Returns:
        A :class:`RetryOutcome` recording success / attempts / final
        failure.
    """
    policy: RetryPolicy = step.retry_policy
    last_failure: RuntimeFailure | None = None
    for attempt_number in range(1, policy.max_attempts + 1):
        try:
            await attempt(attempt_number)
        except Exception as exc:
            kind = classify_runtime_failure(step, exc)
            last_failure = RuntimeFailure(
                step_id=step.id,
                error_type=type(exc).__name__,
                message=str(exc),
                kind=kind,
                attempts=attempt_number,
            )
            if kind is RuntimeFailureKind.structural:
                return RetryOutcome(
                    succeeded=False, attempts=attempt_number, last_failure=last_failure
                )
            continue
        else:
            return RetryOutcome(succeeded=True, attempts=attempt_number, last_failure=None)
    return RetryOutcome(succeeded=False, attempts=policy.max_attempts, last_failure=last_failure)


def build_repair_diff(
    *,
    plan_graph: PlanGraph,
    failed_step: PlanStep,
    failure: RuntimeFailure,
) -> PlanDiff:
    """Build a typed :class:`PlanDiff` for one unrecoverable runtime failure.

    The diff names the violated invariant
    (``materialized_step_executes_successfully``), the failed step as the
    sole affected node, a ``replace`` op carrying the same step (its
    materialized code is to be regenerated), the surviving steps as
    ``reused``, and the failed step's transitive dependents as
    ``invalidated``.

    Args:
        plan_graph: The plan being executed.
        failed_step: The :class:`PlanStep` that failed unrecoverably.
        failure: The classified :class:`RuntimeFailure`.

    Returns:
        A frozen :class:`PlanDiff`.
    """
    step_id = failed_step.id
    operations = (PlanNodeOp(kind=DiffOpKind.replace, node_id=step_id, step=failed_step),)
    rationale = (
        f"step {step_id!r} failed at runtime after {failure.attempts} attempt(s) "
        f"({failure.kind.value}: {failure.error_type}: {failure.message}); "
        f"regenerate its implementation."
    )
    return PlanDiff(
        failed_invariant=_RUNTIME_FAILURE_INVARIANT,
        affected_nodes=(step_id,),
        operations=operations,
        rationale=rationale,
        reused=tuple(s.id for s in plan_graph.steps if s.id != step_id),
        invalidated=plan_graph.downstream_of(step_id),
    )


def diff_requires_rematerialization(diff: PlanDiff) -> bool:
    """Return whether a :class:`PlanDiff` needs AuthorMode re-materialization.

    Any ``add`` / ``remove`` op changes the plan *shape* — the experiment
    workspace must be regenerated. A ``replace`` op rewrites one step's
    code, which also needs AuthorMode codegen. A diff with no ops at all
    is purely advisory and needs no re-materialization.
    """
    return bool(diff.operations)


def build_repair_escalation(
    *,
    plan_graph: PlanGraph,
    diff: PlanDiff,
) -> RepairEscalation:
    """Wrap a :class:`PlanDiff` into the :class:`RepairEscalation` contract.

    The escalation flags whether AuthorMode must re-materialize the
    workspace and always targets ``"author"``.
    """
    needs_remat = diff_requires_rematerialization(diff)
    return RepairEscalation(
        plan_id=plan_graph.plan_id,
        diff=diff,
        requires_rematerialization=needs_remat,
        target_mode="author",
        rationale=diff.rationale,
    )
