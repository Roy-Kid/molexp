"""Retry policy + helpers for the provider's invoke loop.

The provider applies the policy via :func:`should_retry` (decision)
and :func:`sleep_for` (delay computation); the awaiting itself stays
inside the provider so tests can monkey-patch :func:`asyncio.sleep`
to drive the loop deterministically.

This module imports neither ``pydantic_ai`` nor ``asyncio`` — it is
pure data + arithmetic. ``RetryPolicy`` is a frozen pydantic model
with construction-time validation rejecting nonsensical values.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from .errors import ErrorKind

__all__ = [
    "RetryPolicy",
    "should_retry",
    "sleep_for",
]


class RetryPolicy(BaseModel):
    """How aggressively the provider retries classified failures.

    Attributes:
        max_attempts: Total attempts including the first call.
            Validated ``>= 1``.
        backoff_seconds: Base delay between retries; the actual sleep
            is ``backoff_seconds * 2 ** (attempt - 1)``. Validated
            ``>= 0``.
        retry_on: Tuple of :class:`ErrorKind` members that count as
            retryable. Defaults to the two truly-transient classes
            (``model_unavailable``, ``timeout``). ``schema_parse`` is
            **deliberately excluded** — pydantic-ai's
            ``Agent(output_retries=N)`` retries schema-parse failures
            at the model level with the validation error fed back as a
            short follow-up turn (cheap). Re-sending the full
            multi-thousand-token prompt at the router level on the
            same kind of failure burned 14:30 min on one call in
            production; once is enough. ``validation`` and ``unknown``
            also stay excluded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = 3
    backoff_seconds: float = 0.5
    retry_on: tuple[ErrorKind, ...] = (
        ErrorKind.model_unavailable,
        ErrorKind.timeout,
    )

    @field_validator("max_attempts")
    @classmethod
    def _check_max_attempts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_attempts must be >= 1")
        return value

    @field_validator("backoff_seconds")
    @classmethod
    def _check_backoff(cls, value: float) -> float:
        if value < 0:
            raise ValueError("backoff_seconds must be >= 0")
        return value


def should_retry(kind: ErrorKind, policy: RetryPolicy, attempt: int) -> bool:
    """Return whether the loop should run another attempt.

    Args:
        kind: Classified failure of the just-completed attempt.
        policy: Active retry policy.
        attempt: 1-indexed count of the attempt that just failed.

    Returns:
        ``True`` iff ``kind`` is in ``policy.retry_on`` AND
        ``attempt < policy.max_attempts``. The strict inequality
        means once the failed attempt's index reaches the cap, no
        further retry happens (``max_attempts`` is the **total**
        attempt budget, including the first call).
    """
    if attempt >= policy.max_attempts:
        return False
    return kind in policy.retry_on


def sleep_for(policy: RetryPolicy, attempt: int) -> float:
    """Return the delay (seconds) before the next attempt.

    Exponential backoff: ``backoff_seconds * 2 ** (attempt - 1)``.
    For ``attempt=1`` (first failure), the delay is the base
    ``backoff_seconds``; each subsequent attempt doubles.
    """
    return policy.backoff_seconds * 2 ** (attempt - 1)
