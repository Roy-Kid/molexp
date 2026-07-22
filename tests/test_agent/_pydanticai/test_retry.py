"""Tests for :mod:`molexp.agent._pydanticai.retry`.

Covers the transport-only retry policy the router applies:
``should_retry`` (decision) and ``sleep_for`` (exponential backoff),
plus the load-bearing default that keeps ``schema_parse`` out of the
router-level retry loop.
"""

from __future__ import annotations

import pytest

from molexp.agent._pydanticai.errors import ErrorKind
from molexp.agent._pydanticai.retry import RetryPolicy, should_retry, sleep_for


class TestRetryPolicy:
    def test_schema_parse_excluded_from_default_retry_on(self) -> None:
        """Regression (14:30-min prod call): schema_parse must NOT be retried
        at the router level — pydantic-ai's ``Agent(retries={"output": N})``
        retries it cheaply as a follow-up turn, whereas the router re-sends the
        full multi-thousand-token prompt. The default is exactly the two
        transient transport classes."""
        assert RetryPolicy().retry_on == (ErrorKind.model_unavailable, ErrorKind.timeout)


class TestShouldRetry:
    def test_retries_retryable_kind_below_cap(self) -> None:
        assert should_retry(ErrorKind.timeout, RetryPolicy(max_attempts=3), attempt=1) is True

    def test_stops_at_attempt_cap(self) -> None:
        # max_attempts is the total budget; the 3rd attempt is the last.
        assert should_retry(ErrorKind.timeout, RetryPolicy(max_attempts=3), attempt=3) is False

    def test_does_not_retry_non_retryable_kind(self) -> None:
        assert should_retry(ErrorKind.validation, RetryPolicy(max_attempts=3), attempt=1) is False


class TestSleepFor:
    def test_exponential_backoff(self) -> None:
        policy = RetryPolicy(backoff_seconds=0.5)
        assert sleep_for(policy, attempt=1) == pytest.approx(0.5)
        assert sleep_for(policy, attempt=2) == pytest.approx(1.0)
        assert sleep_for(policy, attempt=3) == pytest.approx(2.0)
