"""Tests for :mod:`molexp.agent._pydanticai.retry`.

Acceptance criterion ac-003: ``sleep_for`` matches
``backoff * 2 ** (attempt-1)``; ``should_retry`` returns ``False`` once
attempt reaches the cap or the kind is outside ``retry_on``.
"""

from __future__ import annotations

import pytest

from molexp.agent._pydanticai.errors import ErrorKind
from molexp.agent._pydanticai.retry import RetryPolicy, should_retry, sleep_for

# ── RetryPolicy defaults ───────────────────────────────────────────────────


def test_retry_policy_default_construction() -> None:
    policy = RetryPolicy()
    assert policy.max_attempts == 3
    assert policy.backoff_seconds == 0.5
    assert ErrorKind.model_unavailable in policy.retry_on
    assert ErrorKind.timeout in policy.retry_on
    # schema_parse is deliberately NOT in the default retry_on after
    # ``plan-mode-pydanticai-rewrite``: pydantic-ai's Agent(retries={"output": N})
    # retries schema-parse failures at the model level with the
    # validation error fed back as a cheap follow-up turn; the router
    # used to re-issue the full prompt on the same kind of failure and
    # burned 14:30 min on one call in production.
    assert ErrorKind.schema_parse not in policy.retry_on
    assert ErrorKind.validation not in policy.retry_on
    assert ErrorKind.unknown not in policy.retry_on


# ── should_retry ───────────────────────────────────────────────────────────


def test_should_retry_returns_false_when_attempt_at_cap() -> None:
    policy = RetryPolicy(max_attempts=3)
    # The 3rd attempt (attempt=3) is the last one; should not retry.
    assert should_retry(ErrorKind.timeout, policy, attempt=3) is False


def test_should_retry_true_when_kind_in_retry_on_and_attempts_left() -> None:
    policy = RetryPolicy(max_attempts=3)
    assert should_retry(ErrorKind.timeout, policy, attempt=1) is True
    assert should_retry(ErrorKind.timeout, policy, attempt=2) is True


def test_should_retry_false_when_kind_not_in_retry_on() -> None:
    policy = RetryPolicy(max_attempts=3)
    assert should_retry(ErrorKind.unknown, policy, attempt=1) is False
    assert should_retry(ErrorKind.validation, policy, attempt=1) is False


# ── sleep_for ──────────────────────────────────────────────────────────────


def test_sleep_for_exponential_backoff() -> None:
    policy = RetryPolicy(backoff_seconds=0.5)
    assert sleep_for(policy, attempt=1) == pytest.approx(0.5)
    assert sleep_for(policy, attempt=2) == pytest.approx(1.0)
    assert sleep_for(policy, attempt=3) == pytest.approx(2.0)
    assert sleep_for(policy, attempt=4) == pytest.approx(4.0)
