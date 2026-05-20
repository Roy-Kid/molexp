"""Tests for runtime-failure classification + repair (ac-007 / ac-008 / ac-009).

Covers:

- ``classify_runtime_failure`` — transient vs. structural.
- ``apply_retry_policy`` — a transient step is retried exactly
  ``RetryPolicy.max_attempts`` times before escalation.
- ``build_repair_diff`` — an unrecoverable failure becomes a well-formed
  :class:`PlanDiff`.
- ``build_repair_escalation`` — a diff needing re-materialization yields a
  :class:`RepairEscalation` with ``requires_rematerialization=True`` and
  ``target_mode="author"``.
"""

from __future__ import annotations

import pytest

from molexp.agent.modes._planning import RetryPolicy
from molexp.agent.modes.run.repair import (
    RuntimeFailure,
    RuntimeFailureKind,
    apply_retry_policy,
    build_repair_diff,
    build_repair_escalation,
    classify_runtime_failure,
    diff_requires_rematerialization,
)

from .conftest import make_plan_graph, make_step

# ── classify_runtime_failure ─────────────────────────────────────────────


def test_classify_transient_failure_with_retry_budget() -> None:
    step = make_step("flaky", retry_policy=RetryPolicy(max_attempts=3, on=()))
    kind = classify_runtime_failure(step, TimeoutError("step timeout"))
    assert kind is RuntimeFailureKind.transient


def test_classify_structural_when_no_retry_budget() -> None:
    # max_attempts == 1: even a transient-looking failure is structural.
    step = make_step("once", retry_policy=RetryPolicy(max_attempts=1, on=()))
    kind = classify_runtime_failure(step, TimeoutError("step timeout"))
    assert kind is RuntimeFailureKind.structural


def test_classify_structural_for_non_transient_error() -> None:
    step = make_step("broken", retry_policy=RetryPolicy(max_attempts=3, on=()))
    kind = classify_runtime_failure(step, ValueError("bad config value"))
    assert kind is RuntimeFailureKind.structural


def test_classify_respects_retry_policy_on_allowlist() -> None:
    # `on` restricts retries to matching tags; a non-matching transient
    # failure is treated as structural.
    step = make_step("gated", retry_policy=RetryPolicy(max_attempts=3, on=("connection",)))
    assert (
        classify_runtime_failure(step, TimeoutError("step timeout"))
        is RuntimeFailureKind.structural
    )
    assert (
        classify_runtime_failure(step, ConnectionError("connection reset by peer"))
        is RuntimeFailureKind.transient
    )


# ── apply_retry_policy ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_apply_retry_policy_retries_transient_to_max_attempts() -> None:
    step = make_step("flaky", retry_policy=RetryPolicy(max_attempts=3, on=()))
    attempts: list[int] = []

    async def _always_timeout(attempt_number: int) -> None:
        attempts.append(attempt_number)
        raise TimeoutError("transient timeout")

    outcome = await apply_retry_policy(step, _always_timeout)

    assert attempts == [1, 2, 3]  # exactly max_attempts tries
    assert outcome.succeeded is False
    assert outcome.attempts == 3
    assert outcome.last_failure is not None
    assert outcome.last_failure.kind is RuntimeFailureKind.transient


@pytest.mark.asyncio
async def test_apply_retry_policy_succeeds_on_later_attempt() -> None:
    step = make_step("flaky", retry_policy=RetryPolicy(max_attempts=3, on=()))
    attempts: list[int] = []

    async def _fail_then_succeed(attempt_number: int) -> None:
        attempts.append(attempt_number)
        if attempt_number < 2:
            raise TimeoutError("transient timeout")

    outcome = await apply_retry_policy(step, _fail_then_succeed)

    assert attempts == [1, 2]
    assert outcome.succeeded is True
    assert outcome.attempts == 2
    assert outcome.last_failure is None


@pytest.mark.asyncio
async def test_apply_retry_policy_stops_immediately_on_structural() -> None:
    step = make_step("broken", retry_policy=RetryPolicy(max_attempts=3, on=()))
    attempts: list[int] = []

    async def _structural(attempt_number: int) -> None:
        attempts.append(attempt_number)
        raise ValueError("structural defect")

    outcome = await apply_retry_policy(step, _structural)

    # A structural failure does not consume the retry budget.
    assert attempts == [1]
    assert outcome.succeeded is False
    assert outcome.attempts == 1
    assert outcome.last_failure is not None
    assert outcome.last_failure.kind is RuntimeFailureKind.structural


@pytest.mark.asyncio
async def test_apply_retry_policy_single_attempt_no_retry() -> None:
    step = make_step("once", retry_policy=RetryPolicy(max_attempts=1, on=()))
    attempts: list[int] = []

    async def _fail(attempt_number: int) -> None:
        attempts.append(attempt_number)
        raise TimeoutError("timeout")

    outcome = await apply_retry_policy(step, _fail)
    assert attempts == [1]
    assert outcome.succeeded is False


# ── build_repair_diff ────────────────────────────────────────────────────


def test_build_repair_diff_is_well_formed_for_retry_exhaustion() -> None:
    plan = make_plan_graph()
    failed_step = plan.step_by_id("prepare")
    assert failed_step is not None
    failure = RuntimeFailure(
        step_id="prepare",
        error_type="TimeoutError",
        message="exhausted retries",
        kind=RuntimeFailureKind.transient,
        attempts=3,
    )
    diff = build_repair_diff(plan_graph=plan, failed_step=failed_step, failure=failure)

    assert diff.affected_nodes == ("prepare",)
    assert diff.failed_invariant == "materialized_step_executes_successfully"
    assert diff.rationale  # non-empty
    assert "prepare" in diff.rationale
    assert diff.operations  # carries a replace op
    # `run` depends on `prepare` -> it is invalidated.
    assert "run" in diff.invalidated


def test_build_repair_diff_is_well_formed_for_structural_failure() -> None:
    plan = make_plan_graph()
    failed_step = plan.step_by_id("run")
    assert failed_step is not None
    failure = RuntimeFailure(
        step_id="run",
        error_type="ValueError",
        message="bad output shape",
        kind=RuntimeFailureKind.structural,
        attempts=1,
    )
    diff = build_repair_diff(plan_graph=plan, failed_step=failed_step, failure=failure)

    assert diff.affected_nodes == ("run",)
    assert diff.failed_invariant
    assert diff.rationale
    assert "prepare" in diff.reused  # the surviving step is reused


# ── escalation toward AuthorMode ─────────────────────────────────────────


def test_repair_escalation_requires_rematerialization() -> None:
    plan = make_plan_graph()
    failed_step = plan.step_by_id("prepare")
    assert failed_step is not None
    failure = RuntimeFailure(
        step_id="prepare",
        error_type="ValueError",
        message="structural",
        kind=RuntimeFailureKind.structural,
        attempts=1,
    )
    diff = build_repair_diff(plan_graph=plan, failed_step=failed_step, failure=failure)
    escalation = build_repair_escalation(plan_graph=plan, diff=diff)

    assert escalation.requires_rematerialization is True
    assert escalation.target_mode == "author"
    assert escalation.plan_id == plan.plan_id
    assert escalation.diff is diff
    assert escalation.rationale


def test_diff_requires_rematerialization_flags_ops() -> None:
    plan = make_plan_graph()
    failed_step = plan.step_by_id("run")
    assert failed_step is not None
    failure = RuntimeFailure(
        step_id="run",
        error_type="ValueError",
        message="x",
        kind=RuntimeFailureKind.structural,
        attempts=1,
    )
    diff = build_repair_diff(plan_graph=plan, failed_step=failed_step, failure=failure)
    # The diff carries a replace op -> needs AuthorMode regeneration.
    assert diff_requires_rematerialization(diff) is True
